"""Train DiscoRL (Disco103) on Craftax-Symbolic-v1."""
from __future__ import annotations

import argparse
import collections
from pathlib import Path

import chex
import dm_env
from dm_env import specs as dm_specs
import jax
import jax.numpy as jnp
import numpy as np
import rlax

from craftax.craftax_env import make_craftax_env_from_name
from disco_rl import agent as agent_lib
from disco_rl import types
from disco_rl import utils
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


class SimpleReplayBuffer:
    """A simple FIFO replay buffer for JAX arrays."""

    def __init__(self, capacity: int, seed: int, batch_axis: int = 1):
        self.buffer = collections.deque(maxlen=capacity)
        self.capacity = capacity
        self.np_rng = np.random.default_rng(seed)
        self.batch_axis = batch_axis

    def add(self, rollout: types.ActorRollout) -> None:
        rollout = jax.device_get(rollout)
        split_tree = rlax.tree_split_leaves(rollout, axis=self.batch_axis)
        self.buffer.extend(split_tree)

    def sample(self, batch_size: int) -> types.ActorRollout | None:
        buffer_size = len(self.buffer)
        if buffer_size == 0:
            print("Warning: Trying to sample from an empty buffer.")
            return None

        indices = self.np_rng.integers(buffer_size, size=batch_size)
        batched_samples = utils.tree_stack(
            [self.buffer[i] for i in indices], axis=self.batch_axis
        )
        return batched_samples

    def __len__(self) -> int:
        return len(self.buffer)


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


def summarize_returns(returns: np.ndarray, discounts: np.ndarray) -> float:
    axes = tuple(range(returns.ndim))
    total_returns = (returns * (1 - discounts)).sum(axis=axes)
    total_episodes = (1 - discounts).sum(axis=axes)
    return float(total_returns / max(total_episodes, 1))


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
    parser.add_argument("--buffer_capacity", type=int, default=400000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.3)
    parser.add_argument("--dense", type=str, default="512,512")
    parser.add_argument("--lstm_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--weights_path", type=str, default=None)
    return parser.parse_args()


def clamp_replay_fraction(replay_fraction: float) -> float:
    return float(np.clip(replay_fraction, 0.0, 1.0))


def sample_mixed_rollout(
    actor_rollout: types.ActorRollout,
    buffer: SimpleReplayBuffer,
    batch_size: int,
    replay_fraction: float,
) -> types.ActorRollout:
    on_policy_count = buffer.np_rng.binomial(
        batch_size, 1.0 - replay_fraction
    )
    replay_count = batch_size - on_policy_count
    if on_policy_count == 0:
        return buffer.sample(batch_size)

    actor_rollout = jax.device_get(actor_rollout)
    on_policy_pool = rlax.tree_split_leaves(actor_rollout, axis=1)
    on_policy_indices = buffer.np_rng.integers(
        len(on_policy_pool), size=on_policy_count
    )
    on_policy_samples = [on_policy_pool[i] for i in on_policy_indices]

    replay_samples: list[types.ActorRollout] = []
    if replay_count > 0:
        replay_rollout = buffer.sample(replay_count)
        replay_samples = rlax.tree_split_leaves(replay_rollout, axis=1)

    combined = on_policy_samples + replay_samples
    buffer.np_rng.shuffle(combined)
    return utils.tree_stack(combined, axis=1)


def train(args: argparse.Namespace) -> None:
    devices = tuple(jax.devices())
    if not devices:
        raise RuntimeError("No JAX devices available.")
    device = devices[0]

    dense = tuple(int(x) for x in args.dense.split(",") if x.strip())
    agent_settings = agent_lib.get_settings_disco()
    agent_settings.learning_rate = args.learning_rate
    agent_settings.weight_decay = args.weight_decay
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

    replay_fraction = clamp_replay_fraction(args.replay_fraction)
    buffer = SimpleReplayBuffer(
        capacity=args.buffer_capacity, seed=args.seed, batch_axis=1
    )
    min_buffer_size = 1 if replay_fraction > 0.0 else 0

    def unroll_jittable_actor(params, actor_state, ts, env_state, rng):
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
        return actor_rollout, actor_state, ts, env_state

    learner_step_fn = jax.jit(agent.learner_step, static_argnums=(5,))
    unroll_actor_fn = jax.jit(unroll_jittable_actor)
    acc_rewards_fn = jax.jit(accumulate_rewards)

    learner_state = jax.device_put(learner_state, device)
    actor_state = jax.device_put(actor_state, device)
    update_rule_params = jax.device_put(disco_103_params, device)
    env_state = jax.device_put(env_state, device)
    ts = jax.device_put(ts, device)
    acc_rewards = jax.device_put(acc_rewards, device)

    total_steps = 0
    last_log_step = 0
    expected_on_policy = (1.0 - replay_fraction) * args.batch_size
    expected_replay = replay_fraction * args.batch_size
    print(
        "Replay mix: "
        f"target_r={replay_fraction:.3f} "
        f"expected_on_policy={expected_on_policy:.2f} "
        f"expected_replay={expected_replay:.2f} "
        f"updates_per_iter={args.updates_per_iter}"
    )

    try:
        from tqdm import trange
    except ImportError:
        trange = range

    for step in trange(args.num_iterations):
        rng, rng_actor, rng_learner = jax.random.split(rng, 3)

        actor_rollout, actor_state, ts, env_state = unroll_actor_fn(
            learner_state.params, actor_state, ts, env_state, rng_actor
        )
        buffer.add(actor_rollout)

        total_steps += np.prod(actor_rollout.rewards.shape)
        acc_rewards, returns = acc_rewards_fn(
            acc_rewards,
            (actor_rollout.rewards, actor_rollout.discounts),
        )

        metrics = None
        if len(buffer) >= min_buffer_size:
            learner_rngs = jax.random.split(
                rng_learner, max(args.updates_per_iter, 1)
            )
            for update_rng in learner_rngs:
                learner_rollout = sample_mixed_rollout(
                    actor_rollout,
                    buffer,
                    args.batch_size,
                    replay_fraction,
                )
                learner_state, _, metrics = learner_step_fn(
                    update_rng,
                    learner_rollout,
                    learner_state,
                    actor_state,
                    update_rule_params,
                    False,
                )

        if (step + 1) % args.log_every == 0:
            returns_host = np.array(jax.device_get(returns))
            discounts_host = np.array(jax.device_get(actor_rollout.discounts))
            avg_return = summarize_returns(returns_host, discounts_host)

            msg = (
                f"iter={step + 1} steps={total_steps} "
                f"avg_return={avg_return:.3f}"
            )
            if metrics is not None:
                metrics_host = jax.tree.map(
                    lambda x: float(np.mean(np.array(x))),
                    jax.device_get(metrics),
                )
                msg += (
                    f" loss={metrics_host.get('total_loss', 0.0):.4f} "
                    f"grad_norm={metrics_host.get('global_gradient_norm', 0.0):.4f}"
                )
            print(msg)
            last_log_step = total_steps

    if last_log_step == 0:
        print("Finished without logging; increase --num_iterations or reduce --log_every.")


if __name__ == "__main__":
    train(parse_args())
