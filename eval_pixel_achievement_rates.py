"""Full-episode pixel eval with per-achievement success rates."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import chex
import dm_env
from dm_env import specs
import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax_classic.constants import Achievement
from craftax.craftax_env import make_craftax_env_from_name
from disco_rl import types
from eval_paper_pixels import keep_old_when_done, summarize
from train_paper_pixels import ENV_NAME, FRAME_STACK, Config, make_agent, stack_frames


@chex.dataclass(mappable_dataclass=False)
class EvalState:
    rng: chex.PRNGKey
    env_state: chex.ArrayTree
    frames: chex.Array
    done: chex.Array
    returns: chex.Array
    lengths: chex.Array
    actor_state: chex.ArrayTree


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--num_batches", type=int, default=4)
    p.add_argument("--num_envs", type=int, default=512)
    p.add_argument("--seed", type=int, default=4000)
    p.add_argument("--action_mode", choices=("greedy", "sample"), default="sample")
    p.add_argument("--channels", type=str, default="256,384,384,256")
    p.add_argument("--fc_size", type=int, default=768)
    p.add_argument("--model_lstm_size", type=int, default=1024)
    p.add_argument("--model_head_size", type=int, default=1024)
    return p.parse_args()


def load_params(path: Path):
    with path.open("rb") as f:
        payload = pickle.load(f)
    return payload["learner_state"].params, payload


def make_eval(args: argparse.Namespace):
    channels = tuple(int(x) for x in args.channels.split(",") if x.strip())
    env = make_craftax_env_from_name(ENV_NAME, auto_reset=False)
    env_params = env.default_params
    action_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    fake_env = type(
        "EvalEnvSpec",
        (),
        {
            "obs_spec": {
                "observation": specs.Array(
                    (*obs_space.shape[:2], obs_space.shape[2] * FRAME_STACK),
                    obs_space.dtype,
                )
            },
            "action_spec": specs.BoundedArray((), np.int32, 0, action_space.n - 1),
        },
    )()
    cfg = Config(
        checkpoint_dir=args.out_dir,
        channels=channels,
        fc_size=args.fc_size,
        model_lstm_size=args.model_lstm_size,
        model_head_size=args.model_head_size,
    )
    agent = make_agent(fake_env, cfg)
    actor_state0 = agent.initial_actor_state(jax.random.PRNGKey(args.seed + 1))
    reset_many = jax.vmap(lambda rng: env.reset(rng, env_params))
    step_many = jax.vmap(lambda rng, state, action: env.step(rng, state, action, env_params))
    max_steps = int(env_params.max_timesteps) + 1

    def timestep_from_frames(frames, done):
        return types.EnvironmentTimestep(
            observation={"observation": stack_frames(frames).astype(jnp.float32)},
            step_type=jnp.where(
                done,
                jnp.asarray(dm_env.StepType.LAST, jnp.int32),
                jnp.asarray(dm_env.StepType.MID, jnp.int32),
            ),
            reward=jnp.zeros((args.num_envs,), jnp.float32),
        )

    def one_batch(params, rng):
        rng, reset_rng = jax.random.split(rng)
        raw_obs, env_state = reset_many(jax.random.split(reset_rng, args.num_envs))
        frames = jnp.repeat(raw_obs[:, None, ...], FRAME_STACK, axis=1)
        state = EvalState(
            rng=rng,
            env_state=env_state,
            frames=frames,
            done=jnp.zeros((args.num_envs,), bool),
            returns=jnp.zeros((args.num_envs,), jnp.float32),
            lengths=jnp.zeros((args.num_envs,), jnp.int32),
            actor_state=actor_state0,
        )

        def cond(s):
            return (~jnp.all(s.done)) & (jnp.max(s.lengths) < max_steps)

        def body(s):
            rng, action_rng, step_rng = jax.random.split(s.rng, 3)
            ts = timestep_from_frames(s.frames, s.done)
            should_reset = ts.step_type == dm_env.StepType.LAST
            if args.action_mode == "sample":
                actor_ts, actor_state = agent.actor_step(params, action_rng, ts, s.actor_state)
                actions = actor_ts.actions
            else:
                outs, actor_state = agent._network.one_step(
                    params, s.actor_state, ts.observation, should_reset
                )
                actions = jnp.argmax(outs["logits"], axis=-1).astype(jnp.int32)
            next_obs, next_env_state, reward, env_done, _ = step_many(
                jax.random.split(step_rng, args.num_envs), s.env_state, actions
            )
            active = ~s.done
            next_frames = jnp.concatenate([s.frames[:, 1:], next_obs[:, None, ...]], axis=1)
            return EvalState(
                rng=rng,
                env_state=jax.tree.map(
                    lambda old, new: keep_old_when_done(old, new, s.done),
                    s.env_state,
                    next_env_state,
                ),
                frames=keep_old_when_done(s.frames, next_frames, s.done),
                done=s.done | env_done,
                returns=s.returns + jnp.where(active, reward, 0.0),
                lengths=s.lengths + active.astype(jnp.int32),
                actor_state=actor_state,
            )

        state = jax.lax.while_loop(cond, body, state)
        achievements = state.env_state.achievements.astype(jnp.float32)
        achievement_pct = achievements * state.done[:, None].astype(jnp.float32) * 100.0
        score = jnp.exp(jnp.mean(jnp.log1p(achievement_pct), axis=-1)) - 1.0
        return state.returns, state.lengths, achievements, score, state.done

    return jax.jit(one_batch), max_steps


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    params, payload = load_params(args.checkpoint.expanduser().resolve())
    eval_batch, max_steps = make_eval(args)

    all_returns, all_lengths, all_achievements, all_scores, all_done = [], [], [], [], []
    batch_rows = []
    start = time.time()
    for batch in range(args.num_batches):
        rng = jax.random.PRNGKey(args.seed + batch)
        returns, lengths, achievements, scores, done = jax.device_get(eval_batch(params, rng))
        all_returns.append(returns)
        all_lengths.append(lengths)
        all_achievements.append(achievements)
        all_scores.append(scores)
        all_done.append(done)
        row = {
            "batch": batch,
            "episodes": int(args.num_envs),
            "mean_return": float(np.mean(returns)),
            "median_return": float(np.median(returns)),
            "max_return": float(np.max(returns)),
            "mean_length": float(np.mean(lengths)),
            "mean_achievements": float(np.mean(achievements.sum(axis=-1))),
            "mean_score": float(np.mean(scores)),
            "done_fraction": float(np.mean(done)),
            "elapsed_seconds": time.time() - start,
        }
        batch_rows.append(row)
        print(
            f"batch={batch + 1}/{args.num_batches} mean_return={row['mean_return']:.3f} "
            f"max_return={row['max_return']:.3f} mean_achievements={row['mean_achievements']:.2f} "
            f"done={row['done_fraction']:.3f} elapsed={row['elapsed_seconds']:.1f}s",
            flush=True,
        )

    returns = np.concatenate(all_returns)
    lengths = np.concatenate(all_lengths)
    achievements = np.concatenate(all_achievements).astype(bool)
    scores = np.concatenate(all_scores)
    done = np.concatenate(all_done).astype(bool)
    ach_counts = achievements.sum(axis=0)
    achievement_rows = [
        {
            "achievement": achievement.name.lower(),
            "index": achievement.value,
            "successes": int(ach_counts[achievement.value]),
            "episodes": int(returns.size),
            "success_rate": float(ach_counts[achievement.value] / returns.size),
            "success_pct": float(100.0 * ach_counts[achievement.value] / returns.size),
        }
        for achievement in Achievement
    ]
    achievement_rows.sort(key=lambda r: (-r["success_rate"], r["index"]))

    raw_path = out_dir / "eval_raw.npz"
    np.savez_compressed(
        raw_path,
        returns=returns,
        lengths=lengths,
        achievements=achievements,
        scores=scores,
        done=done,
        checkpoint=str(args.checkpoint),
    )
    for name, rows in [
        ("eval_batches.csv", batch_rows),
        ("achievement_success_rates.csv", achievement_rows),
    ]:
        with (out_dir / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_total_steps": int(payload.get("total_steps", -1)),
        "action_mode": args.action_mode,
        "num_episodes": int(returns.size),
        "num_batches": args.num_batches,
        "num_envs": args.num_envs,
        "max_steps": max_steps,
        "completed_fraction": float(np.mean(done)),
        "return": summarize(returns),
        "length": summarize(lengths),
        "achievements_per_episode": summarize(achievements.sum(axis=-1)),
        "score": summarize(scores),
        "achievement_success_rates": achievement_rows,
        "raw_path": str(raw_path),
        "achievement_csv_path": str(out_dir / "achievement_success_rates.csv"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
