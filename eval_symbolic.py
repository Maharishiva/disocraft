"""Minimal full-episode eval for symbolic Craftax DiscoRL checkpoints."""

from __future__ import annotations

import argparse
import csv
import pickle
import time
from pathlib import Path

import chex
import dm_env
import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name
from disco_rl import agent as agent_lib
from disco_rl import types
from dm_env import specs


@chex.dataclass(mappable_dataclass=False)
class EvalState:
    rng: chex.PRNGKey
    env_state: chex.ArrayTree
    obs: chex.Array
    actor_state: chex.ArrayTree
    done: chex.Array
    returns: chex.Array
    lengths: chex.Array


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", action="append", required=True, help="name=/path/to.pkl")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--num_rollouts", type=int, default=100)
    p.add_argument("--num_envs", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def make_agent(obs_spec: specs.Array, action_spec: specs.BoundedArray) -> agent_lib.Agent:
    settings = agent_lib.get_settings_disco()
    settings.learning_rate = 3e-4
    settings.net_settings.name = "mlp"
    settings.net_settings.net_args = dict(
        dense=(512, 512),
        model_arch_name="lstm",
        head_w_init_std=1e-2,
        model_kwargs=dict(head_mlp_hiddens=(128,), lstm_size=128),
    )
    return agent_lib.Agent(
        agent_settings=settings,
        single_observation_spec={"observation": obs_spec},
        single_action_spec=action_spec,
        batch_axis_name=None,
    )


def load_params(path: Path):
    with path.open("rb") as f:
        payload = pickle.load(f)
    return payload["learner_state"].params


def named_checkpoints(raw: list[str]) -> list[tuple[str, Path]]:
    out = []
    for item in raw:
        name, sep, path = item.partition("=")
        if not sep:
            path = name
            name = Path(path).parent.name
        out.append((name, Path(path).expanduser().resolve()))
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_craftax_env_from_name("Craftax-Symbolic-v1", auto_reset=False)
    env_params = env.default_params
    action_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    action_spec = specs.BoundedArray((), np.int32, 0, action_space.n - 1)
    obs_spec = specs.Array(obs_space.shape, obs_space.dtype)
    agent = make_agent(obs_spec, action_spec)
    actor_state0 = agent.initial_actor_state(jax.random.PRNGKey(args.seed + 1))
    reset_many = jax.vmap(lambda rng: env.reset(rng, env_params))
    step_many = jax.vmap(lambda rng, state, action: env.step(rng, state, action, env_params))

    def keep_old_when_done(old, new, done):
        mask = done.reshape(done.shape + (1,) * (old.ndim - 1))
        return jnp.where(mask, old, new)

    def eval_one_rollout(params, rng):
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = reset_many(jax.random.split(reset_rng, args.num_envs))
        state = EvalState(
            rng=rng,
            env_state=env_state,
            obs=jnp.asarray(obs, jnp.float32),
            actor_state=actor_state0,
            done=jnp.zeros((args.num_envs,), bool),
            returns=jnp.zeros((args.num_envs,), jnp.float32),
            lengths=jnp.zeros((args.num_envs,), jnp.int32),
        )

        def cond(s):
            return ~jnp.all(s.done)

        def body(s):
            rng, action_rng, step_rng = jax.random.split(s.rng, 3)
            timestep = types.EnvironmentTimestep(
                observation={"observation": s.obs},
                step_type=jnp.where(
                    s.done,
                    jnp.asarray(dm_env.StepType.LAST, jnp.int32),
                    jnp.asarray(dm_env.StepType.MID, jnp.int32),
                ),
                reward=jnp.zeros((args.num_envs,), jnp.float32),
            )
            actor_ts, actor_state = agent.actor_step(params, action_rng, timestep, s.actor_state)
            next_obs, next_env_state, reward, env_done, _ = step_many(
                jax.random.split(step_rng, args.num_envs), s.env_state, actor_ts.actions
            )
            active = ~s.done
            done = s.done | env_done
            return EvalState(
                rng=rng,
                env_state=jax.tree.map(
                    lambda old, new: keep_old_when_done(old, new, s.done),
                    s.env_state,
                    next_env_state,
                ),
                obs=keep_old_when_done(s.obs, jnp.asarray(next_obs, jnp.float32), s.done),
                actor_state=actor_state,
                done=done,
                returns=s.returns + jnp.where(active, reward, 0.0),
                lengths=s.lengths + active.astype(jnp.int32),
            )

        state = jax.lax.while_loop(cond, body, state)
        return state.returns, state.lengths

    eval_one_rollout = jax.jit(eval_one_rollout)
    rows = []
    for ckpt_name, ckpt_path in named_checkpoints(args.checkpoint):
        params = load_params(ckpt_path)
        returns_by_rollout, lengths_by_rollout = [], []
        start = time.time()
        for i in range(args.num_rollouts):
            rng = jax.random.PRNGKey(args.seed + i)
            returns, lengths = jax.device_get(eval_one_rollout(params, rng))
            returns_by_rollout.append(returns)
            lengths_by_rollout.append(lengths)
            row = {
                "checkpoint": ckpt_name,
                "rollout": i,
                "episodes": args.num_envs,
                "mean_return": float(np.mean(returns)),
                "median_return": float(np.median(returns)),
                "std_return": float(np.std(returns)),
                "min_return": float(np.min(returns)),
                "max_return": float(np.max(returns)),
                "mean_length": float(np.mean(lengths)),
                "max_length": int(np.max(lengths)),
                "seconds": time.time() - start,
            }
            rows.append(row)
            print(
                f"{ckpt_name} rollout={i + 1}/{args.num_rollouts} "
                f"mean={row['mean_return']:.3f} max={row['max_return']:.3f} "
                f"mean_len={row['mean_length']:.1f} elapsed={row['seconds']:.1f}s",
                flush=True,
            )
        np.savez_compressed(
            out_dir / f"{ckpt_name}_raw.npz",
            returns=np.asarray(returns_by_rollout),
            lengths=np.asarray(lengths_by_rollout),
            checkpoint=str(ckpt_path),
        )

    csv_path = out_dir / "eval_rollouts.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
