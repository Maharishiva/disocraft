"""Quantify policy-output variability for a Craftax pixel checkpoint."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import distrax
import dm_env
from dm_env import specs
import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax_classic.constants import Action
from craftax.craftax_env import make_craftax_env_from_name
from disco_rl import types
from train_paper_pixels import ENV_NAME, FRAME_STACK, Config, make_agent, stack_frames


ACTION_NAMES = [a.name.lower() for a in Action]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--num_steps", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pair_samples", type=int, default=20000)
    return p.parse_args()


def load_params(path: Path):
    with path.open("rb") as f:
        payload = pickle.load(f)
    return payload["learner_state"].params, payload


def setup_agent(out_dir: Path):
    env = make_craftax_env_from_name(ENV_NAME, auto_reset=False)
    env_params = env.default_params
    action_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    fake_env = type(
        "AnalyzeEnvSpec",
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
    return env, env_params, make_agent(fake_env, Config(checkpoint_dir=out_dir))


def summarize(x: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "p05": float(np.percentile(x, 5)),
        "p95": float(np.percentile(x, 95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def pairwise_stats(probs: np.ndarray, n: int, seed: int) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    m = len(probs)
    i = rng.integers(0, m, size=min(n, m * 4))
    j = rng.integers(0, m, size=len(i))
    p = np.clip(probs[i], 1e-8, 1.0)
    q = np.clip(probs[j], 1e-8, 1.0)
    kl = np.sum(p * (np.log(p) - np.log(q)), axis=-1)
    mprob = 0.5 * (p + q)
    js = 0.5 * np.sum(p * (np.log(p) - np.log(mprob)), axis=-1)
    js += 0.5 * np.sum(q * (np.log(q) - np.log(mprob)), axis=-1)
    l1 = np.sum(np.abs(p - q), axis=-1)
    return {"kl_pq": summarize(kl), "js": summarize(js), "l1": summarize(l1)}


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    params, payload = load_params(args.checkpoint.expanduser().resolve())
    env, env_params, agent = setup_agent(out_dir)
    actor_state0 = agent.initial_actor_state(jax.random.PRNGKey(args.seed + 1))
    reset_many = jax.vmap(lambda rng: env.reset(rng, env_params))
    step_many = jax.vmap(lambda rng, state, action: env.step(rng, state, action, env_params))

    def rollout(rng):
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = reset_many(jax.random.split(reset_rng, args.num_envs))
        frames = jnp.repeat(obs[:, None, ...], FRAME_STACK, axis=1)
        done = jnp.zeros((args.num_envs,), bool)
        returns = jnp.zeros((args.num_envs,), jnp.float32)
        actor_state = actor_state0

        def step(carry, _):
            rng, env_state, frames, done, returns, actor_state = carry
            rng, action_rng, step_rng = jax.random.split(rng, 3)
            ts = types.EnvironmentTimestep(
                observation={"observation": stack_frames(frames).astype(jnp.float32)},
                step_type=jnp.where(
                    done,
                    jnp.asarray(dm_env.StepType.LAST, jnp.int32),
                    jnp.asarray(dm_env.StepType.MID, jnp.int32),
                ),
                reward=jnp.zeros((args.num_envs,), jnp.float32),
            )
            outs, actor_state = agent._network.one_step(
                params, actor_state, ts.observation, ts.step_type == dm_env.StepType.LAST
            )
            probs = jax.nn.softmax(outs["logits"])
            actions = distrax.Softmax(logits=outs["logits"]).sample(seed=action_rng)
            next_obs, next_env_state, reward, env_done, _ = step_many(
                jax.random.split(step_rng, args.num_envs), env_state, actions
            )
            active = ~done
            frames = jnp.concatenate([frames[:, 1:], next_obs[:, None, ...]], axis=1)
            done = done | env_done
            returns = returns + jnp.where(active, reward, 0.0)
            y_probs = jax.nn.softmax(outs["y"])
            z_a = jnp.take_along_axis(outs["z"], actions[:, None, None], axis=1).squeeze(axis=1)
            z_probs = jax.nn.softmax(z_a)
            sample = {
                "probs": probs,
                "logits": outs["logits"],
                "actions": actions,
                "greedy": jnp.argmax(probs, axis=-1),
                "active": active,
                "returns": returns,
                "y_entropy": -jnp.sum(y_probs * jnp.log(jnp.clip(y_probs, 1e-8)), axis=-1),
                "z_entropy": -jnp.sum(z_probs * jnp.log(jnp.clip(z_probs, 1e-8)), axis=-1),
                "y_logit_std": jnp.std(outs["y"], axis=-1),
                "z_logit_std": jnp.std(z_a, axis=-1),
            }
            carry = (rng, next_env_state, frames, done, returns, actor_state)
            return carry, sample

        carry = (rng, env_state, frames, done, returns, actor_state)
        carry, samples = jax.lax.scan(step, carry, None, length=args.num_steps)
        return samples, carry[4]

    samples, final_returns = jax.device_get(jax.jit(rollout)(jax.random.PRNGKey(args.seed)))
    active = samples["active"].reshape(-1).astype(bool)
    probs = samples["probs"].reshape((-1, len(ACTION_NAMES)))[active]
    logits = samples["logits"].reshape((-1, len(ACTION_NAMES)))[active]
    actions = samples["actions"].reshape(-1)[active]
    greedy = samples["greedy"].reshape(-1)[active]
    entropy = -np.sum(np.clip(probs, 1e-8, 1.0) * np.log(np.clip(probs, 1e-8, 1.0)), axis=-1)
    mean_probs = probs.mean(axis=0)
    std_probs = probs.std(axis=0)
    greedy_counts = np.bincount(greedy, minlength=len(ACTION_NAMES)) / len(greedy)
    sample_counts = np.bincount(actions, minlength=len(ACTION_NAMES)) / len(actions)
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_total_steps": int(payload.get("total_steps", -1)),
        "num_active_states": int(len(probs)),
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "policy_entropy": summarize(entropy),
        "log_uniform_17": float(np.log(len(ACTION_NAMES))),
        "policy_pairwise": pairwise_stats(probs, args.pair_samples, args.seed),
        "logit_per_state_std": summarize(np.std(logits, axis=-1)),
        "logit_across_state_std_mean": float(np.mean(np.std(logits, axis=0))),
        "prob_across_state_std_mean": float(np.mean(std_probs)),
        "y_entropy": summarize(samples["y_entropy"].reshape(-1)[active]),
        "z_chosen_entropy": summarize(samples["z_entropy"].reshape(-1)[active]),
        "y_logit_std": summarize(samples["y_logit_std"].reshape(-1)[active]),
        "z_chosen_logit_std": summarize(samples["z_logit_std"].reshape(-1)[active]),
        "rollout_return": summarize(np.asarray(final_returns)),
        "actions": [
            {
                "id": i,
                "name": name,
                "mean_prob": float(mean_probs[i]),
                "std_prob": float(std_probs[i]),
                "greedy_fraction": float(greedy_counts[i]),
                "sample_fraction": float(sample_counts[i]),
            }
            for i, name in enumerate(ACTION_NAMES)
        ],
    }
    (out_dir / "policy_static_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    np.savez_compressed(
        out_dir / "policy_static_raw.npz",
        probs=probs,
        logits=logits,
        actions=actions,
        greedy=greedy,
        entropy=entropy,
        final_returns=np.asarray(final_returns),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
