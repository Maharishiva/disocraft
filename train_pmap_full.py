"""Multi-GPU Disco103 training on Craftax with policy checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np

from disco_rl import agent as agent_lib
from disco_rl import types
from train import (
    NOTEBOOK_DEFAULT_GLOBAL_BATCH_SIZE,
    NOTEBOOK_DEFAULT_GLOBAL_NUM_ENVS,
    CraftaxBatchedEnvironment,
    PAPER_CRAFTER_BATCH_SIZE,
    PAPER_CRAFTER_BUFFER_TRANSITIONS,
    PAPER_CRAFTER_ENV_STEPS,
    PAPER_CRAFTER_REPLAY_FRACTION,
    PAPER_CRAFTER_ROLLOUT_LEN,
    ReplayBufferState,
    TrainLoopState,
    accumulate_rewards,
    clamp_replay_fraction,
    init_replay_buffer,
    load_disco_103_params,
    replay_buffer_add,
    resolve_num_iterations,
    resolve_weights_path,
    sample_fifo_replay_batch,
    sample_mixed_batch,
    strip_rollout_for_replay,
    swap_time_batch,
)


AXIS_NAME = "devices"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Disco103 on Craftax with data-parallel local GPUs."
    )
    parser.add_argument("--env_name", default="Craftax-Symbolic-v1")
    parser.add_argument("--num_devices", type=int, default=None)
    parser.add_argument("--global_num_envs", type=int, default=NOTEBOOK_DEFAULT_GLOBAL_NUM_ENVS)
    parser.add_argument("--local_num_envs", type=int, default=None)
    parser.add_argument("--global_batch_size", type=int, default=NOTEBOOK_DEFAULT_GLOBAL_BATCH_SIZE)
    parser.add_argument("--local_batch_size", type=int, default=None)
    parser.add_argument("--num_iterations", type=int, default=None)
    parser.add_argument("--target_env_steps", type=int, default=PAPER_CRAFTER_ENV_STEPS)
    parser.add_argument(
        "--iteration_rounding", choices=("nearest", "floor", "ceil"), default="nearest"
    )
    parser.add_argument("--chunk_iterations", type=int, default=1000)
    parser.add_argument("--rollout_len", type=int, default=PAPER_CRAFTER_ROLLOUT_LEN)
    parser.add_argument("--updates_per_iter", type=int, default=1)
    parser.add_argument("--replay_mode", choices=("mixed", "fifo"), default="fifo")
    parser.add_argument("--replay_fraction", type=float, default=PAPER_CRAFTER_REPLAY_FRACTION)
    parser.add_argument("--min_buffer_size", type=int, default=None)
    parser.add_argument("--local_buffer_capacity", type=int, default=None)
    parser.add_argument("--local_buffer_capacity_transitions", type=int, default=None)
    parser.add_argument(
        "--global_buffer_capacity_transitions",
        type=int,
        default=PAPER_CRAFTER_BUFFER_TRANSITIONS,
    )
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--dense", type=str, default="512,512")
    parser.add_argument("--lstm_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--checkpoint_every_chunks", type=int, default=10)
    parser.add_argument("--keep_chunk_checkpoints", action="store_true")
    parser.add_argument("--metrics_csv", type=str, default=None)
    return parser.parse_args()


def resolve_local_count(
    local_value: int | None,
    global_value: int | None,
    num_devices: int,
    name: str,
) -> int:
    if local_value is not None:
        return int(local_value)
    if global_value is None:
        raise ValueError(f"Either local_{name} or global_{name} must be provided.")
    if global_value <= 0:
        raise ValueError(f"global_{name} must be positive.")
    if global_value % num_devices != 0:
        raise ValueError(
            f"global_{name}={global_value} must divide evenly across num_devices={num_devices}."
        )
    return int(global_value // num_devices)


def create_agent(
    env: CraftaxBatchedEnvironment,
    learning_rate: float,
    dense: tuple[int, ...],
    lstm_size: int,
) -> agent_lib.Agent:
    agent_settings = agent_lib.get_settings_disco()
    agent_settings.learning_rate = learning_rate
    agent_settings.net_settings.name = "mlp"
    agent_settings.net_settings.net_args = dict(
        dense=dense,
        model_arch_name="lstm",
        head_w_init_std=1e-2,
        model_kwargs=dict(
            head_mlp_hiddens=(128,),
            lstm_size=lstm_size,
        ),
    )
    return agent_lib.Agent(
        agent_settings=agent_settings,
        single_observation_spec=env.single_observation_spec(),
        single_action_spec=env.single_action_spec(),
        batch_axis_name=AXIS_NAME,
    )


def summarize_returns_parts(
    returns: chex.Array, discounts: chex.Array
) -> tuple[chex.Array, chex.Array]:
    episode_mask = 1.0 - discounts
    return jnp.sum(returns * episode_mask), jnp.sum(episode_mask)


def flatten_params(params: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    flat = {}
    for module_name, module_params in params.items():
        for param_name, value in module_params.items():
            flat[f"{module_name}/{param_name}"] = np.asarray(value)
    return flat


def take_replica_zero(tree: Any) -> Any:
    return jax.tree.map(lambda x: np.asarray(jax.device_get(x[0])), tree)


def atomic_pickle_dump(payload: Any, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def atomic_npz_dump(arrays: dict[str, np.ndarray], path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez(f, **arrays)
    tmp_path.replace(path)


def save_checkpoint(
    state: TrainLoopState,
    args: argparse.Namespace,
    checkpoint_dir: Path,
    chunk_idx: int,
    final: bool,
) -> tuple[Path, Path]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    learner_state = take_replica_zero(state.learner_state)
    total_steps = int(np.asarray(jax.device_get(state.total_steps[0])))
    payload = {
        "format": "disocraft_train_pmap_checkpoint_v1",
        "chunk_idx": int(chunk_idx),
        "final": bool(final),
        "total_steps": total_steps,
        "args": vars(args),
        "learner_state": learner_state,
    }

    latest_pkl = checkpoint_dir / "checkpoint_latest.pkl"
    latest_npz = checkpoint_dir / "policy_params_latest.npz"
    atomic_pickle_dump(payload, latest_pkl)
    atomic_npz_dump(flatten_params(learner_state.params), latest_npz)

    if final:
        final_pkl = checkpoint_dir / "checkpoint_final.pkl"
        final_npz = checkpoint_dir / "policy_params_final.npz"
        atomic_pickle_dump(payload, final_pkl)
        atomic_npz_dump(flatten_params(learner_state.params), final_npz)
        return final_pkl, final_npz

    if args.keep_chunk_checkpoints:
        chunk_pkl = checkpoint_dir / f"checkpoint_chunk_{chunk_idx:06d}.pkl"
        chunk_npz = checkpoint_dir / f"policy_params_chunk_{chunk_idx:06d}.npz"
        atomic_pickle_dump(payload, chunk_pkl)
        atomic_npz_dump(flatten_params(learner_state.params), chunk_npz)

    return latest_pkl, latest_npz


def append_metric_row(metrics_csv: Path, row: dict[str, Any]) -> None:
    fieldnames = [
        "iter",
        "steps",
        "avg_return",
        "loss",
        "grad_norm",
        "episodes",
    ]
    exists = metrics_csv.exists()
    with metrics_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def validate_args(args: argparse.Namespace, num_devices: int) -> None:
    if num_devices <= 0:
        raise ValueError("num_devices must be positive.")
    if args.local_num_envs <= 0:
        raise ValueError("local_num_envs must be positive.")
    if args.local_batch_size <= 0:
        raise ValueError("local_batch_size must be positive.")
    if args.rollout_len <= 0:
        raise ValueError("rollout_len must be positive.")
    if args.num_iterations is not None and args.num_iterations <= 0:
        raise ValueError("num_iterations must be positive when provided.")
    if args.target_env_steps is not None and args.target_env_steps <= 0:
        raise ValueError("target_env_steps must be positive when provided.")
    if args.chunk_iterations <= 0:
        raise ValueError("chunk_iterations must be positive.")
    if args.updates_per_iter <= 0:
        raise ValueError("updates_per_iter must be positive.")
    if args.checkpoint_every_chunks <= 0:
        raise ValueError("checkpoint_every_chunks must be positive.")


def train(args: argparse.Namespace) -> None:
    local_devices = jax.local_devices()
    num_devices = args.num_devices or len(local_devices)
    if num_devices > len(local_devices):
        raise ValueError(
            f"Requested {num_devices} devices but only {len(local_devices)} are visible."
        )
    devices = local_devices[:num_devices]
    args.local_num_envs = resolve_local_count(
        args.local_num_envs, args.global_num_envs, num_devices, "num_envs"
    )
    args.local_batch_size = resolve_local_count(
        args.local_batch_size, args.global_batch_size, num_devices, "batch_size"
    )
    validate_args(args, num_devices)

    replay_fraction = clamp_replay_fraction(args.replay_fraction)
    if args.local_buffer_capacity is None:
        if args.local_buffer_capacity_transitions is None:
            if args.global_buffer_capacity_transitions is None:
                raise ValueError(
                    "Either local_buffer_capacity_transitions or "
                    "global_buffer_capacity_transitions must be provided."
                )
            if args.global_buffer_capacity_transitions <= 0:
                raise ValueError("global_buffer_capacity_transitions must be positive.")
            args.local_buffer_capacity_transitions = int(
                math.ceil(args.global_buffer_capacity_transitions / num_devices)
            )
        requested_capacity = int(
            math.ceil(args.local_buffer_capacity_transitions / args.rollout_len)
        )
    else:
        requested_capacity = int(args.local_buffer_capacity)
    local_steps_per_iter = args.local_num_envs * args.rollout_len
    global_steps_per_iter = num_devices * local_steps_per_iter
    num_iterations = resolve_num_iterations(
        args.num_iterations,
        args.target_env_steps,
        global_steps_per_iter,
        args.iteration_rounding,
    )
    args.num_iterations = num_iterations
    args.global_num_envs = args.local_num_envs * num_devices
    args.global_batch_size = args.local_batch_size * num_devices

    max_chunks_needed = args.local_num_envs * num_iterations
    local_buffer_capacity = int(min(requested_capacity, max_chunks_needed))
    if local_buffer_capacity <= 0:
        raise ValueError("local buffer capacity must be positive.")

    min_buffer_size = (
        int(args.min_buffer_size) if args.min_buffer_size is not None else args.local_batch_size
    )
    if min_buffer_size > local_buffer_capacity:
        raise ValueError(
            f"min_buffer_size ({min_buffer_size}) cannot exceed local buffer "
            f"capacity ({local_buffer_capacity})."
        )

    dense = tuple(int(x) for x in args.dense.split(",") if x.strip())
    env = CraftaxBatchedEnvironment(
        env_name=args.env_name, batch_size=args.local_num_envs, auto_reset=True
    )
    agent = create_agent(env, args.learning_rate, dense, args.lstm_size)

    weights_path = resolve_weights_path(args.weights_path)
    disco_103_params = load_disco_103_params(weights_path)
    random_update_rule_params, _ = agent.update_rule.init_params(jax.random.PRNGKey(0))
    chex.assert_trees_all_equal_shapes_and_dtypes(random_update_rule_params, disco_103_params)
    update_rule_params = disco_103_params

    target_steps = num_iterations * global_steps_per_iter
    local_buffer_transitions = local_buffer_capacity * args.rollout_len
    global_buffer_transitions = local_buffer_transitions * num_devices
    expected_local_on_policy = (1.0 - replay_fraction) * args.local_batch_size
    expected_local_replay = replay_fraction * args.local_batch_size
    fifo_replay_ratio = args.global_batch_size * args.updates_per_iter / args.global_num_envs
    fifo_replay_fraction = 1.0 - (1.0 / fifo_replay_ratio)

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    metrics_csv = (
        Path(args.metrics_csv).resolve() if args.metrics_csv else checkpoint_dir / "metrics.csv"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_csv.unlink(missing_ok=True)
    (checkpoint_dir / "run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "num_devices": num_devices,
                "devices": [str(d) for d in devices],
                "global_num_envs": args.global_num_envs,
                "global_batch_size": args.global_batch_size,
                "global_steps_per_iter": global_steps_per_iter,
                "target_steps": target_steps,
                "local_buffer_capacity_chunks": local_buffer_capacity,
                "local_buffer_transitions": local_buffer_transitions,
                "global_buffer_transitions": global_buffer_transitions,
            },
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "Pmap config: "
        f"devices={num_devices} "
        f"local_envs={args.local_num_envs} "
        f"global_envs={args.global_num_envs} "
        f"local_batch={args.local_batch_size} "
        f"global_batch={args.global_batch_size} "
        f"rollout_len={args.rollout_len} "
        f"steps_per_iter={global_steps_per_iter} "
        f"num_iterations={num_iterations} "
        f"target_env_steps={args.target_env_steps} "
        f"actual_env_steps={target_steps}"
    )
    print(
        "Replay/checkpoint config: "
        f"mode={args.replay_mode} "
        f"replay_fraction={replay_fraction:.3f} "
        f"expected_local_on_policy={expected_local_on_policy:.2f} "
        f"expected_local_replay={expected_local_replay:.2f} "
        f"fifo_replay_ratio={fifo_replay_ratio:.2f} "
        f"fifo_replay_fraction={fifo_replay_fraction:.4f} "
        f"local_buffer_chunks={local_buffer_capacity} "
        f"local_buffer_transitions={local_buffer_transitions} "
        f"global_buffer_transitions={global_buffer_transitions} "
        f"checkpoint_dir={checkpoint_dir} "
        f"metrics_csv={metrics_csv}"
    )

    def init_one_device(device_rng):
        rng, reset_rng = jax.random.split(device_rng)
        learner_rng = jax.random.PRNGKey(args.seed + 2027)
        actor_rng = jax.random.PRNGKey(args.seed + 2028)
        learner_state = agent.initial_learner_state(learner_rng)
        actor_state = agent.initial_actor_state(actor_rng)
        env_state, ts = env.reset(reset_rng)
        acc_rewards = jnp.zeros((args.local_num_envs,), dtype=jnp.float32)

        dummy_obs = jax.tree.map(
            lambda spec: jnp.zeros((args.local_num_envs,) + spec.shape, dtype=spec.dtype),
            env.single_observation_spec(),
        )
        dummy_ts = types.EnvironmentTimestep(
            observation=dummy_obs,
            step_type=jnp.zeros((args.local_num_envs,), dtype=jnp.int32),
            reward=jnp.zeros((args.local_num_envs,), dtype=jnp.float32),
        )
        dummy_timestep, _ = agent.actor_step(
            learner_state.params, jax.random.PRNGKey(0), dummy_ts, actor_state
        )
        dummy_rollout = types.ActorRollout.from_timestep(
            jax.tree.map(
                lambda x: jnp.zeros((args.rollout_len,) + x.shape, dtype=x.dtype),
                dummy_timestep,
            )
        )
        dummy_rollout = strip_rollout_for_replay(dummy_rollout)
        buffer = init_replay_buffer(dummy_rollout, local_buffer_capacity)
        return TrainLoopState(
            rng=rng,
            env_state=env_state,
            timestep=ts,
            learner_state=learner_state,
            actor_state=actor_state,
            buffer=buffer,
            acc_rewards=acc_rewards,
            total_steps=jnp.array(0, dtype=jnp.int64),
        )

    def unroll_actor(params, actor_state, ts, env_state, rng):
        def _single_step(carry, step_rng):
            env_state, ts, actor_state = carry
            actor_timestep, actor_state = agent.actor_step(params, step_rng, ts, actor_state)
            env_state, ts = env.step(env_state, actor_timestep.actions)
            return (env_state, ts, actor_state), actor_timestep

        (env_state, ts, actor_state), actor_rollout = jax.lax.scan(
            _single_step,
            (env_state, ts, actor_state),
            jax.random.split(rng, args.rollout_len),
        )
        actor_rollout = types.ActorRollout.from_timestep(actor_rollout)
        actor_rollout = strip_rollout_for_replay(actor_rollout)
        return actor_rollout, actor_state, ts, env_state

    def _log_callback(values):
        step, total_steps, avg_return, loss, grad_norm, episodes = values
        row = {
            "iter": int(step),
            "steps": int(total_steps),
            "avg_return": float(avg_return),
            "loss": float(loss),
            "grad_norm": float(grad_norm),
            "episodes": float(episodes),
        }
        append_metric_row(metrics_csv, row)
        print(
            f"iter={row['iter']} steps={row['steps']} "
            f"avg_return={row['avg_return']:.3f} "
            f"loss={row['loss']:.4f} "
            f"grad_norm={row['grad_norm']:.4f} "
            f"episodes={row['episodes']:.0f}"
        )

    log_every = int(args.log_every)
    enable_logging = log_every > 0

    def train_step(state: TrainLoopState, _):
        rng, env_state, ts, learner_state, actor_state, buffer, acc_rewards, total_steps = (
            state.rng,
            state.env_state,
            state.timestep,
            state.learner_state,
            state.actor_state,
            state.buffer,
            state.acc_rewards,
            state.total_steps,
        )
        rng, rollout_rng, update_rng = jax.random.split(rng, 3)

        actor_rollout, actor_state, ts, env_state = unroll_actor(
            learner_state.params, actor_state, ts, env_state, rollout_rng
        )
        rollout_bt = swap_time_batch(actor_rollout)
        buffer = replay_buffer_add(buffer, rollout_bt, local_buffer_capacity)

        acc_rewards, returns = accumulate_rewards(
            acc_rewards,
            (actor_rollout.rewards, actor_rollout.discounts),
        )

        update_rngs = jax.random.split(update_rng, args.updates_per_iter)
        zero_metric = jnp.array(0.0, dtype=jnp.float32)

        def _update_step(learner_state, rng):
            sample_rng, step_rng = jax.random.split(rng)
            if args.replay_mode == "fifo":
                learner_rollout = sample_fifo_replay_batch(
                    sample_rng,
                    buffer,
                    args.local_batch_size,
                )
            else:
                learner_rollout = sample_mixed_batch(
                    sample_rng,
                    rollout_bt,
                    buffer,
                    args.local_batch_size,
                    replay_fraction,
                )
            learner_agent_state = learner_rollout.first_state(time_axis=0)
            learner_state, _, metrics = agent.learner_step(
                step_rng,
                learner_rollout,
                learner_state,
                learner_agent_state,
                update_rule_params,
                False,
            )
            loss = metrics.get("total_loss", zero_metric)
            grad_norm = metrics.get("global_gradient_norm", zero_metric)
            return learner_state, (loss, grad_norm)

        def _do_updates(ls):
            new_ls, (losses, grad_norms) = jax.lax.scan(_update_step, ls, update_rngs)
            return new_ls, (losses[-1], grad_norms[-1])

        def _skip_updates(ls):
            return ls, (zero_metric, zero_metric)

        learner_state, (loss, grad_norm) = jax.lax.cond(
            buffer.size >= min_buffer_size,
            _do_updates,
            _skip_updates,
            learner_state,
        )

        total_steps = total_steps + global_steps_per_iter
        iter_idx = total_steps // global_steps_per_iter
        return_sum, episode_count = summarize_returns_parts(returns, actor_rollout.discounts)
        global_return_sum = jax.lax.psum(return_sum, AXIS_NAME)
        global_episode_count = jax.lax.psum(episode_count, AXIS_NAME)
        avg_return = jnp.where(
            global_episode_count > 0,
            global_return_sum / global_episode_count,
            0.0,
        )
        avg_loss = jax.lax.pmean(loss, AXIS_NAME)
        avg_grad_norm = jax.lax.pmean(grad_norm, AXIS_NAME)

        if enable_logging:
            do_log = (iter_idx % log_every == 0) & (jax.lax.axis_index(AXIS_NAME) == 0)

            def _log(_):
                jax.debug.callback(
                    _log_callback,
                    (
                        iter_idx,
                        total_steps,
                        avg_return,
                        avg_loss,
                        avg_grad_norm,
                        global_episode_count,
                    ),
                )
                return None

            jax.lax.cond(do_log, _log, lambda _: None, operand=None)

        new_state = TrainLoopState(
            rng=rng,
            env_state=env_state,
            timestep=ts,
            learner_state=learner_state,
            actor_state=actor_state,
            buffer=buffer,
            acc_rewards=acc_rewards,
            total_steps=total_steps,
        )
        return new_state, None

    def train_chunk(state):
        state, _ = jax.lax.scan(train_step, state, xs=None, length=args.chunk_iterations)
        return state

    init_pmap = jax.pmap(init_one_device, axis_name=AXIS_NAME, devices=devices)
    train_chunk_pmaps = {
        args.chunk_iterations: jax.pmap(
            train_chunk,
            axis_name=AXIS_NAME,
            devices=devices,
            donate_argnums=(0,),
        )
    }

    def get_train_chunk_pmap(chunk_iterations: int):
        if chunk_iterations not in train_chunk_pmaps:

            def train_sized_chunk(state):
                state, _ = jax.lax.scan(train_step, state, xs=None, length=chunk_iterations)
                return state

            train_chunk_pmaps[chunk_iterations] = jax.pmap(
                train_sized_chunk,
                axis_name=AXIS_NAME,
                devices=devices,
                donate_argnums=(0,),
            )
        return train_chunk_pmaps[chunk_iterations]

    init_rng = jax.random.PRNGKey(args.seed)
    device_rngs = jax.random.split(init_rng, num_devices)
    state = init_pmap(device_rngs)
    jax.block_until_ready(state.total_steps)

    total_chunks = math.ceil(num_iterations / args.chunk_iterations)
    completed_iterations = 0
    for chunk_idx in range(1, total_chunks + 1):
        chunk_iterations = min(args.chunk_iterations, num_iterations - completed_iterations)
        state = get_train_chunk_pmap(chunk_iterations)(state)
        completed_iterations += chunk_iterations
        jax.block_until_ready(state.total_steps)
        if chunk_idx % args.checkpoint_every_chunks == 0:
            pkl_path, npz_path = save_checkpoint(
                state, args, checkpoint_dir, chunk_idx, final=False
            )
            print(f"checkpoint chunk={chunk_idx} pkl={pkl_path} params={npz_path}")

    final_pkl, final_npz = save_checkpoint(state, args, checkpoint_dir, total_chunks, final=True)
    print(f"final_checkpoint pkl={final_pkl} params={final_npz}")


if __name__ == "__main__":
    train(parse_args())
