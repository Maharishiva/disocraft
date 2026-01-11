"""Train DiscoRL (Disco103) on Craftax-Symbolic-v1."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import chex
import dm_env
from dm_env import specs as dm_specs
import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax_env import make_craftax_env_from_name
from disco_rl import agent as agent_lib
from disco_rl import types
from disco_rl.environments import base as disco_base_env


@chex.dataclass(mappable_dataclass=False)
class CraftaxEnvState:
    state: chex.ArrayTree
    rng: chex.PRNGKey


class CraftaxBatchedEnvironment(disco_base_env.Environment):
    """Batched Craftax environment adapter for DiscoRL."""

    def __init__(self, env_name: str, batch_size: int, auto_reset: bool = True):
        self.batch_size = batch_size
        self._env = make_craftax_env_from_name(env_name, auto_reset=False)
        self._env_params = self._env.default_params
        self._auto_reset = auto_reset

        action_space = self._env.action_space(self._env_params)
        self._single_action_spec = dm_specs.BoundedArray(
            (), np.int32, 0, action_space.n - 1
        )

        obs_space = self._env.observation_space(self._env_params)
        self._single_observation_spec = {
            "observation": dm_specs.Array(shape=obs_space.shape, dtype=obs_space.dtype)
        }

        self._batched_env_step = jax.vmap(self._single_env_step)
        self._batched_env_reset = jax.vmap(self._single_env_reset)

    def _single_env_step(
        self, env_state: CraftaxEnvState, action: chex.Array
    ) -> tuple[CraftaxEnvState, types.EnvironmentTimestep]:
        new_rng, step_rng, reset_rng = jax.random.split(env_state.rng, 3)
        obs, new_state, reward, done, _ = self._env.step(
            step_rng, env_state.state, action, self._env_params
        )

        if self._auto_reset:
            reset_obs, reset_state = self._env.reset(reset_rng, self._env_params)
            new_state = jax.tree.map(
                lambda reset_x, x: jax.lax.select(done, reset_x, x),
                reset_state,
                new_state,
            )
            obs = jax.tree.map(
                lambda reset_x, x: jax.lax.select(done, reset_x, x),
                reset_obs,
                obs,
            )

        timestep = types.EnvironmentTimestep(
            observation={"observation": jnp.asarray(obs, dtype=jnp.float32)},
            step_type=jax.lax.select(
                done, dm_env.StepType.LAST, dm_env.StepType.MID
            ),
            reward=jnp.asarray(reward, dtype=jnp.float32),
        )
        return CraftaxEnvState(state=new_state, rng=new_rng), timestep

    def _single_env_reset(
        self, rng_key: chex.PRNGKey
    ) -> tuple[CraftaxEnvState, types.EnvironmentTimestep]:
        new_rng, reset_rng = jax.random.split(rng_key)
        obs, state = self._env.reset(reset_rng, self._env_params)
        timestep = types.EnvironmentTimestep(
            observation={"observation": jnp.asarray(obs, dtype=jnp.float32)},
            step_type=jnp.asarray(dm_env.StepType.MID),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
        )
        return CraftaxEnvState(state=state, rng=new_rng), timestep

    def step(
        self, state: CraftaxEnvState, actions: chex.Array
    ) -> tuple[CraftaxEnvState, types.EnvironmentTimestep]:
        return self._batched_env_step(state, actions)

    def reset(
        self, rng_key: chex.PRNGKey
    ) -> tuple[CraftaxEnvState, types.EnvironmentTimestep]:
        rngs = jax.random.split(rng_key, self.batch_size)
        return self._batched_env_reset(rngs)

    def single_action_spec(self) -> types.ActionSpec:
        return self._single_action_spec

    def single_observation_spec(self) -> types.Specs:
        return self._single_observation_spec


@chex.dataclass(mappable_dataclass=False)
class ReplayBufferState:
    data: types.ActorRollout
    idx: chex.Array
    size: chex.Array


@chex.dataclass(mappable_dataclass=False)
class TrainLoopState:
    rng: chex.PRNGKey
    env_state: CraftaxEnvState
    timestep: types.EnvironmentTimestep
    learner_state: agent_lib.LearnerState
    actor_state: chex.ArrayTree
    buffer: ReplayBufferState
    acc_rewards: chex.Array
    total_steps: chex.Array


def unflatten_params(flat_params: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    params = {}
    for key_wb in flat_params:
        key = "/".join(key_wb.split("/")[:-1])
        params[key] = {
            "b": flat_params[f"{key}/b"],
            "w": flat_params[f"{key}/w"],
        }
    return params


def resolve_weights_path(path: str | None) -> Path:
    if path:
        return Path(path)

    local = (
        Path(__file__).resolve().parent
        / "external/disco_rl/disco_rl/update_rules/weights/disco_103.npz"
    )
    if local.exists():
        return local

    import disco_rl

    pkg = (
        Path(disco_rl.__file__).resolve().parent
        / "update_rules/weights/disco_103.npz"
    )
    if pkg.exists():
        return pkg

    raise FileNotFoundError(
        "Could not find disco_103.npz. Pass --weights_path or clone disco_rl."
    )


def load_disco_103_params(weights_path: Path) -> dict[str, dict[str, np.ndarray]]:
    with np.load(weights_path) as flat_params:
        flat_params = dict(flat_params)
    return unflatten_params(flat_params)


def accumulate_rewards(acc_rewards, x):
    rewards, discounts = x

    def _step_fn(acc_rewards, x):
        rewards, discounts = x
        acc_rewards += rewards
        return acc_rewards * discounts, acc_rewards

    return jax.lax.scan(_step_fn, acc_rewards, (rewards, discounts))

def swap_time_batch(rollout: types.ActorRollout) -> types.ActorRollout:
    return jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), rollout)

def strip_rollout_for_replay(rollout: types.ActorRollout) -> types.ActorRollout:
    """Keep only what Disco103 training needs from the behaviour policy.

    Storing full `agent_outs` (in particular `q` with shape [A, num_bins]) makes
    a 400k-transition buffer infeasible on GPU memory.
    """
    behaviour_logits = rollout.agent_outs["logits"]
    return types.ActorRollout(
        observations=rollout.observations,
        actions=rollout.actions,
        rewards=rollout.rewards,
        discounts=rollout.discounts,
        agent_outs={"logits": behaviour_logits},
        states=rollout.states,
        logits=rollout.logits,
    )


def init_replay_buffer(
    example_rollout: types.ActorRollout, capacity: int
) -> ReplayBufferState:
    rollout_bt = swap_time_batch(example_rollout)
    data = jax.tree.map(
        lambda x: jnp.zeros((capacity,) + x.shape[1:], dtype=x.dtype),
        rollout_bt,
    )
    return ReplayBufferState(
        data=data,
        idx=jnp.array(0, dtype=jnp.int32),
        size=jnp.array(0, dtype=jnp.int32),
    )


def replay_buffer_add(
    buffer: ReplayBufferState,
    rollout_bt: types.ActorRollout,
    capacity: int,
) -> ReplayBufferState:
    batch_size = rollout_bt.rewards.shape[0]
    indices = (buffer.idx + jnp.arange(batch_size)) % capacity
    data = jax.tree.map(lambda buf, x: buf.at[indices].set(x), buffer.data, rollout_bt)
    idx = (buffer.idx + batch_size) % capacity
    size = jnp.minimum(capacity, buffer.size + batch_size)
    return ReplayBufferState(data=data, idx=idx, size=size)


def replay_buffer_sample(
    buffer: ReplayBufferState, rng: chex.PRNGKey, batch_size: int
) -> types.ActorRollout:
    indices = jax.random.randint(rng, (batch_size,), 0, buffer.size)
    return jax.tree.map(lambda buf: buf[indices], buffer.data)


def sample_mixed_batch(
    rng: chex.PRNGKey,
    rollout_bt: types.ActorRollout,
    buffer: ReplayBufferState,
    batch_size: int,
    replay_fraction: float,
) -> types.ActorRollout:
    rng, on_rng, replay_rng, mix_rng, perm_rng = jax.random.split(rng, 5)

    on_indices = jax.random.randint(
        on_rng, (batch_size,), 0, rollout_bt.rewards.shape[0]
    )
    on_policy = jax.tree.map(lambda x: x[on_indices], rollout_bt)
    replay = replay_buffer_sample(buffer, replay_rng, batch_size)

    use_replay = jax.random.bernoulli(
        mix_rng, p=replay_fraction, shape=(batch_size,)
    )

    def _mix(on, rep):
        mask = use_replay.reshape((batch_size,) + (1,) * (on.ndim - 1))
        return jnp.where(mask, rep, on)

    mixed = jax.tree.map(_mix, on_policy, replay)
    perm = jax.random.permutation(perm_rng, batch_size)
    mixed = jax.tree.map(lambda x: x[perm], mixed)
    return swap_time_batch(mixed)


def summarize_returns_jax(returns: chex.Array, discounts: chex.Array) -> chex.Array:
    total_returns = jnp.sum(returns * (1.0 - discounts))
    total_episodes = jnp.sum(1.0 - discounts)
    return jnp.where(total_episodes > 0, total_returns / total_episodes, 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Disco103 update rule on Craftax-Symbolic-v1."
    )
    parser.add_argument("--env_name", default="Craftax-Symbolic-v1")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--rollout_len", type=int, default=29)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--updates_per_iter", type=int, default=1)
    parser.add_argument("--replay_fraction", type=float, default=0.99)
    parser.add_argument("--buffer_capacity", type=int, default=None)
    parser.add_argument("--buffer_capacity_transitions", type=int, default=400000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--dense", type=str, default="512,512")
    parser.add_argument("--lstm_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--weights_path", type=str, default=None)
    return parser.parse_args()


def clamp_replay_fraction(replay_fraction: float) -> float:
    return float(np.clip(replay_fraction, 0.0, 1.0))


def train(args: argparse.Namespace) -> None:
    if not jax.devices():
        raise RuntimeError("No JAX devices available.")

    replay_fraction = clamp_replay_fraction(args.replay_fraction)
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.updates_per_iter <= 0:
        raise ValueError("updates_per_iter must be positive.")

    if args.buffer_capacity is None:
        requested_capacity = int(
            math.ceil(args.buffer_capacity_transitions / args.rollout_len)
        )
    else:
        requested_capacity = int(args.buffer_capacity)
    if requested_capacity <= 0:
        raise ValueError("buffer_capacity must be positive.")

    max_chunks_needed = args.num_envs * args.num_iterations
    buffer_capacity = int(min(requested_capacity, max_chunks_needed))
    if buffer_capacity <= 0:
        raise ValueError("buffer_capacity must be positive.")

    steps_per_iter = args.num_envs * args.rollout_len
    dense = tuple(int(x) for x in args.dense.split(",") if x.strip())
    agent_settings = agent_lib.get_settings_disco()
    agent_settings.learning_rate = args.learning_rate
    agent_settings.net_settings.name = "mlp"
    agent_settings.net_settings.net_args = dict(
        dense=dense,
        model_arch_name="lstm",
        head_w_init_std=1e-2,
        model_kwargs=dict(
            head_mlp_hiddens=(128,),
            lstm_size=args.lstm_size,
        ),
    )

    env = CraftaxBatchedEnvironment(
        env_name=args.env_name, batch_size=args.num_envs, auto_reset=True
    )
    agent = agent_lib.Agent(
        agent_settings=agent_settings,
        single_observation_spec=env.single_observation_spec(),
        single_action_spec=env.single_action_spec(),
        batch_axis_name=None,
    )

    weights_path = resolve_weights_path(args.weights_path)
    disco_103_params = load_disco_103_params(weights_path)

    random_update_rule_params, _ = agent.update_rule.init_params(
        jax.random.PRNGKey(0)
    )
    chex.assert_trees_all_equal_shapes_and_dtypes(
        random_update_rule_params, disco_103_params
    )

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_rng, learner_rng, actor_rng = jax.random.split(rng, 4)
    env_state, ts = env.reset(reset_rng)
    acc_rewards = jnp.zeros((args.num_envs,), dtype=jnp.float32)
    learner_state = agent.initial_learner_state(learner_rng)
    actor_state = agent.initial_actor_state(actor_rng)

    dummy_obs = jax.tree.map(
        lambda spec: jnp.zeros((args.num_envs,) + spec.shape, dtype=spec.dtype),
        env.single_observation_spec(),
    )
    dummy_ts = types.EnvironmentTimestep(
        observation=dummy_obs,
        step_type=jnp.zeros((args.num_envs,), dtype=jnp.int32),
        reward=jnp.zeros((args.num_envs,), dtype=jnp.float32),
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
    buffer = init_replay_buffer(dummy_rollout, buffer_capacity)
    update_rule_params = disco_103_params

    expected_on_policy = (1.0 - replay_fraction) * args.batch_size
    expected_replay = replay_fraction * args.batch_size
    buffer_transitions = buffer_capacity * args.rollout_len
    print(
        "Replay mix: "
        f"target_r={replay_fraction:.3f} "
        f"expected_on_policy={expected_on_policy:.2f} "
        f"expected_replay={expected_replay:.2f} "
        f"updates_per_iter={args.updates_per_iter} "
        f"buffer_chunks={buffer_capacity} "
        f"(requested={requested_capacity}) "
        f"buffer_transitions={buffer_transitions}"
    )

    def unroll_actor(params, actor_state, ts, env_state, rng):
        def _single_step(carry, step_rng):
            env_state, ts, actor_state = carry
            actor_timestep, actor_state = agent.actor_step(
                params, step_rng, ts, actor_state
            )
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

    def _log_callback(args):
        step, total_steps, avg_return, loss, grad_norm = args
        print(
            f"iter={int(step)} steps={int(total_steps)} "
            f"avg_return={float(avg_return):.3f} "
            f"loss={float(loss):.4f} grad_norm={float(grad_norm):.4f}"
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
        buffer = replay_buffer_add(buffer, rollout_bt, buffer_capacity)

        acc_rewards, returns = accumulate_rewards(
            acc_rewards,
            (actor_rollout.rewards, actor_rollout.discounts),
        )

        update_rngs = jax.random.split(update_rng, args.updates_per_iter)

        def _update_step(learner_state, rng):
            sample_rng, step_rng = jax.random.split(rng)
            learner_rollout = sample_mixed_batch(
                sample_rng,
                rollout_bt,
                buffer,
                args.batch_size,
                replay_fraction,
            )
            learner_state, _, metrics = agent.learner_step(
                step_rng,
                learner_rollout,
                learner_state,
                actor_state,
                update_rule_params,
                False,
            )
            return learner_state, metrics

        learner_state, metrics_stack = jax.lax.scan(
            _update_step, learner_state, update_rngs
        )
        metrics = jax.tree.map(lambda x: x[-1], metrics_stack)

        total_steps = total_steps + steps_per_iter
        iter_idx = total_steps // steps_per_iter
        avg_return = summarize_returns_jax(returns, actor_rollout.discounts)
        loss = metrics.get("total_loss", jnp.array(0.0))
        grad_norm = metrics.get("global_gradient_norm", jnp.array(0.0))

        if enable_logging:
            do_log = iter_idx % log_every == 0

            def _log(_):
                jax.debug.callback(
                    _log_callback,
                    (iter_idx, total_steps, avg_return, loss, grad_norm),
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
        metrics_out = dict(
            steps=total_steps,
            avg_return=avg_return,
            loss=loss,
            grad_norm=grad_norm,
        )
        return new_state, metrics_out

    init_state = TrainLoopState(
        rng=rng,
        env_state=env_state,
        timestep=ts,
        learner_state=learner_state,
        actor_state=actor_state,
        buffer=buffer,
        acc_rewards=acc_rewards,
        total_steps=jnp.array(0, dtype=jnp.int64),
    )

    def train_loop(state):
        return jax.lax.scan(train_step, state, xs=None, length=args.num_iterations)

    train_loop_jit = jax.jit(train_loop)
    final_state, _ = train_loop_jit(init_state)
    jax.block_until_ready(final_state.total_steps)


if __name__ == "__main__":
    train(parse_args())
