"""Paper-shaped Disco103 training on Craftax Classic pixels."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chex
import dm_env
from dm_env import specs
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from craftax.craftax_env import make_craftax_env_from_name
from disco_rl import agent as agent_lib
from disco_rl import optimizers
from disco_rl import types
from disco_rl.networks import nets as disco_nets


ENV_NAME = "Craftax-Classic-Pixels-v1"
ROLLOUT = 29
FRAME_STACK = 32
LR = 3e-4
WEIGHT_DECAY = 0.3
MAX_ABS_UPDATE = 1.0
BATCH_SIZE = 24
NUM_ENVS = 8
REPLAY_FRACTION = 0.99
BUFFER_TRANSITIONS = 400_000


@dataclass(frozen=True)
class Config:
    checkpoint_dir: Path
    target_steps: int = 1_000_000
    seed: int = 0
    num_envs: int = NUM_ENVS
    batch_size: int = BATCH_SIZE
    rollout_len: int = ROLLOUT
    frame_stack: int = FRAME_STACK
    replay_fraction: float = REPLAY_FRACTION
    buffer_transitions: int = BUFFER_TRANSITIONS
    min_buffer_size: int | None = None
    learning_rate: float = LR
    weight_decay: float = WEIGHT_DECAY
    max_abs_update: float = MAX_ABS_UPDATE
    channels: tuple[int, ...] = (256, 384, 384, 256)
    fc_size: int = 768
    model_lstm_size: int = 1024
    model_head_size: int = 1024
    log_every: int = 50
    chunk_iterations: int = 100
    checkpoint_every_chunks: int = 1
    snapshot_every_steps: int = 0


@chex.dataclass(mappable_dataclass=False)
class EnvState:
    state: Any
    rng: chex.PRNGKey
    frames: chex.Array


@chex.dataclass(mappable_dataclass=False)
class Replay:
    obs_history: chex.Array
    actions: chex.Array
    rewards: chex.Array
    discounts: chex.Array
    logits: chex.Array
    idx: chex.Array
    size: chex.Array


@chex.dataclass(mappable_dataclass=False)
class LoopState:
    rng: chex.PRNGKey
    env_state: EnvState
    timestep: types.EnvironmentTimestep
    learner_state: agent_lib.LearnerState
    actor_state: Any
    replay: Replay
    frame_tail: chex.Array
    acc_rewards: chex.Array
    total_steps: chex.Array


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", type=Path, required=True)
    p.add_argument("--target_steps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_envs", type=int, default=NUM_ENVS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--replay_fraction", type=float, default=REPLAY_FRACTION)
    p.add_argument("--buffer_transitions", type=int, default=BUFFER_TRANSITIONS)
    p.add_argument("--min_buffer_size", type=int, default=None)
    p.add_argument("--channels", type=str, default="256,384,384,256")
    p.add_argument("--fc_size", type=int, default=768)
    p.add_argument("--model_lstm_size", type=int, default=1024)
    p.add_argument("--model_head_size", type=int, default=1024)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--chunk_iterations", type=int, default=100)
    p.add_argument("--checkpoint_every_chunks", type=int, default=1)
    p.add_argument("--snapshot_every_steps", type=int, default=0)
    args = p.parse_args()
    channels = tuple(int(x) for x in args.channels.split(",") if x.strip())
    return Config(
        checkpoint_dir=args.checkpoint_dir.resolve(),
        target_steps=args.target_steps,
        seed=args.seed,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        replay_fraction=float(np.clip(args.replay_fraction, 0.0, 1.0)),
        buffer_transitions=args.buffer_transitions,
        min_buffer_size=args.min_buffer_size,
        channels=channels,
        fc_size=args.fc_size,
        model_lstm_size=args.model_lstm_size,
        model_head_size=args.model_head_size,
        log_every=args.log_every,
        chunk_iterations=args.chunk_iterations,
        checkpoint_every_chunks=args.checkpoint_every_chunks,
        snapshot_every_steps=args.snapshot_every_steps,
    )


def unflatten_params(flat: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    params = {}
    for key in flat:
        if not key.endswith("/w"):
            continue
        module = key[:-2]
        params[module] = {"w": flat[key], "b": flat[f"{module}/b"]}
    return params


def load_update_rule() -> dict[str, dict[str, np.ndarray]]:
    path = Path(__file__).parent / "external/disco_rl/disco_rl/update_rules/weights/disco_103.npz"
    with np.load(path) as f:
        return unflatten_params(dict(f))


def flatten_params(params: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {f"{m}/{k}": np.asarray(v) for m, p in params.items() for k, v in p.items()}


def stack_frames(frames: chex.Array) -> chex.Array:
    # [..., K, H, W, C] -> [..., H, W, K*C]
    prefix = frames.shape[:-4]
    k, h, w, c = frames.shape[-4:]
    return jnp.reshape(jnp.moveaxis(frames, -4, -2), (*prefix, h, w, k * c))


def to_u8(obs: chex.Array) -> chex.Array:
    return jnp.asarray(jnp.clip(jnp.rint(obs * 255.0), 0, 255), jnp.uint8)


class CraftaxPixelsBatch:
    def __init__(self, batch_size: int, frame_stack: int):
        self.batch_size = batch_size
        self.frame_stack = frame_stack
        self.env = make_craftax_env_from_name(ENV_NAME, auto_reset=False)
        self.params = self.env.default_params
        a = self.env.action_space(self.params)
        o = self.env.observation_space(self.params)
        self.raw_shape = o.shape
        self.action_spec = specs.BoundedArray((), np.int32, 0, a.n - 1)
        self.obs_spec = {
            "observation": specs.Array(
                (*o.shape[:2], o.shape[2] * frame_stack),
                o.dtype,
            )
        }
        self.vstep = jax.vmap(self._step)
        self.vreset = jax.vmap(self._reset)

    def _repeat(self, obs):
        frames = jnp.repeat(obs[None, ...], self.frame_stack, axis=0)
        return frames, stack_frames(frames)

    def _step(self, s: EnvState, action):
        rng, step_rng, reset_rng = jax.random.split(s.rng, 3)
        obs, state, reward, done, _ = self.env.step(step_rng, s.state, action, self.params)
        reset_obs, reset_state = self.env.reset(reset_rng, self.params)
        state = jax.tree.map(lambda r, x: jax.lax.select(done, r, x), reset_state, state)
        obs = jax.tree.map(lambda r, x: jax.lax.select(done, r, x), reset_obs, obs)
        reset_frames, reset_stacked = self._repeat(obs)
        next_frames = jnp.concatenate([s.frames[1:], obs[None, ...]], axis=0)
        next_frames = jax.lax.select(done, reset_frames, next_frames)
        stacked = jax.lax.select(done, reset_stacked, stack_frames(next_frames))
        ts = types.EnvironmentTimestep(
            observation={"observation": jnp.asarray(stacked, jnp.float32)},
            step_type=jax.lax.select(done, dm_env.StepType.LAST, dm_env.StepType.MID),
            reward=jnp.asarray(reward, jnp.float32),
        )
        return EnvState(state, rng, next_frames), ts

    def _reset(self, rng):
        rng, reset_rng = jax.random.split(rng)
        obs, state = self.env.reset(reset_rng, self.params)
        frames, stacked = self._repeat(obs)
        ts = types.EnvironmentTimestep(
            observation={"observation": jnp.asarray(stacked, jnp.float32)},
            step_type=jnp.asarray(dm_env.StepType.MID),
            reward=jnp.asarray(0.0, jnp.float32),
        )
        return EnvState(state, rng, frames), ts

    def reset(self, rng):
        return self.vreset(jax.random.split(rng, self.batch_size))

    def step(self, state, actions):
        return self.vstep(state, actions)


class ImpalaCNN(disco_nets.MLPHeadNet):
    def __init__(self, *args, channels: tuple[int, ...], fc_size: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.fc_size = fc_size

    def _residual(self, x, channels: int, name: str):
        y = jax.nn.relu(x)
        y = hk.Conv2D(channels, 3, padding="SAME", name=f"{name}_conv1")(y)
        y = jax.nn.relu(y)
        y = hk.Conv2D(channels, 3, padding="SAME", name=f"{name}_conv2")(y)
        return x + y

    def _embedding_pass(self, inputs, should_reset=None):
        del should_reset
        x = inputs["observation"].astype(jnp.float32)
        for i, channels in enumerate(self.channels):
            x = hk.Conv2D(channels, 3, padding="SAME", name=f"impala_{i}_conv")(x)
            x = hk.max_pool(x, (1, 3, 3, 1), (1, 2, 2, 1), padding="SAME")
            x = self._residual(x, channels, f"impala_{i}_res0")
            x = self._residual(x, channels, f"impala_{i}_res1")
        x = hk.Flatten()(jax.nn.relu(x))
        x = hk.Linear(self.fc_size, name="impala_fc")(x)
        return jax.nn.relu(x)


def install_impala_network() -> None:
    original = disco_nets.get_network

    def get_network(name: str, *args, **kwargs):
        if name != "impala_cnn":
            return original(name, *args, **kwargs)

        def make_net():
            return ImpalaCNN(*args, **kwargs)

        def step_fn(*call_args, **call_kwargs):
            return make_net()(*call_args, **call_kwargs)

        def unroll_fn(*call_args, **call_kwargs):
            return make_net().unroll(*call_args, **call_kwargs)

        init_fn, one_step = hk.without_apply_rng(hk.transform_with_state(step_fn))
        _, unroll = hk.without_apply_rng(hk.transform_with_state(unroll_fn))
        return types.PolicyNetwork(init=init_fn, one_step=one_step, unroll=unroll)

    disco_nets.get_network = get_network


def make_agent(env: CraftaxPixelsBatch, cfg: Config) -> agent_lib.Agent:
    install_impala_network()
    s = agent_lib.get_settings_disco()
    s.learning_rate = cfg.learning_rate
    s.max_abs_update = cfg.max_abs_update
    s.net_settings.name = "impala_cnn"
    s.net_settings.net_args = dict(
        channels=cfg.channels,
        fc_size=cfg.fc_size,
        model_arch_name="lstm",
        head_w_init_std=1e-2,
        model_kwargs=dict(
            head_mlp_hiddens=(cfg.model_head_size,),
            lstm_size=cfg.model_lstm_size,
        ),
    )
    agent = agent_lib.Agent(
        single_observation_spec=env.obs_spec,
        single_action_spec=env.action_spec,
        agent_settings=s,
        batch_axis_name=None,
    )
    agent._optimizer = optax.chain(
        optimizers.scale_by_adam_sg_denom(),
        optax.clip(max_delta=cfg.max_abs_update),
        optax.add_decayed_weights(cfg.weight_decay),
        optax.scale(-cfg.learning_rate),
    )
    return agent


def strip_rollout(r: types.ActorRollout) -> types.ActorRollout:
    return types.ActorRollout(
        observations=r.observations,
        actions=r.actions,
        rewards=r.rewards,
        discounts=r.discounts,
        agent_outs={"logits": r.agent_outs["logits"]},
        states={},
        logits=r.agent_outs["logits"],
    )


def swap_tb(x):
    return jax.tree.map(lambda y: jnp.swapaxes(y, 0, 1), x)


def init_replay(
    cap: int, hist_len: int, raw_shape: tuple[int, ...], rollout_len: int, actions: int
):
    return Replay(
        obs_history=jnp.zeros((cap, hist_len, *raw_shape), dtype=jnp.uint8),
        actions=jnp.zeros((cap, rollout_len), dtype=jnp.int32),
        rewards=jnp.zeros((cap, rollout_len), dtype=jnp.float32),
        discounts=jnp.zeros((cap, rollout_len), dtype=jnp.float32),
        logits=jnp.zeros((cap, rollout_len, actions), dtype=jnp.float32),
        idx=jnp.array(0, dtype=jnp.int32),
        size=jnp.array(0, dtype=jnp.int32),
    )


def add_replay(buf: Replay, rollout_bt: types.ActorRollout, frame_tail, cap: int):
    raw = to_u8(rollout_bt.observations["observation"][..., -3:])
    hist = jnp.concatenate([frame_tail, raw], axis=1)
    n = raw.shape[0]
    idx = (buf.idx + jnp.arange(n)) % cap
    return Replay(
        obs_history=buf.obs_history.at[idx].set(hist),
        actions=buf.actions.at[idx].set(rollout_bt.actions),
        rewards=buf.rewards.at[idx].set(rollout_bt.rewards),
        discounts=buf.discounts.at[idx].set(rollout_bt.discounts.astype(jnp.float32)),
        logits=buf.logits.at[idx].set(rollout_bt.agent_outs["logits"]),
        idx=(buf.idx + n) % cap,
        size=jnp.minimum(cap, buf.size + n),
    ), hist[:, -(frame_tail.shape[1]) :]


def history_to_obs(hist: chex.Array, rollout_len: int, frame_stack: int):
    obs = []
    for t in range(rollout_len):
        obs.append(stack_frames(hist[:, t : t + frame_stack]).astype(jnp.float32) / 255.0)
    return jnp.stack(obs, axis=0)


def sample_replay(buf: Replay, rng, batch_size: int, rollout_len: int, frame_stack: int):
    idx = jax.random.randint(rng, (batch_size,), 0, buf.size)
    hist = buf.obs_history[idx]
    obs = {"observation": history_to_obs(hist, rollout_len, frame_stack)}
    logits = jnp.swapaxes(buf.logits[idx], 0, 1)
    return types.ActorRollout(
        observations=obs,
        actions=jnp.swapaxes(buf.actions[idx], 0, 1),
        rewards=jnp.swapaxes(buf.rewards[idx], 0, 1),
        discounts=jnp.swapaxes(buf.discounts[idx], 0, 1),
        agent_outs={"logits": logits},
        states={},
        logits=logits,
    )


def sample_on_policy(rollout_bt, rng, batch_size: int):
    idx = jax.random.randint(rng, (batch_size,), 0, rollout_bt.rewards.shape[0])
    return swap_tb(jax.tree.map(lambda x: x[idx], rollout_bt))


def mix_batches(replay, fresh, rng, p_replay: float, batch_size: int):
    mask = jax.random.bernoulli(rng, p_replay, (batch_size,))

    def mix(a, b):
        shape = (1, batch_size) + (1,) * (a.ndim - 2)
        return jnp.where(mask.reshape(shape), a, b)

    return jax.tree.map(mix, replay, fresh)


def reward_scan(acc, rollout):
    def step(a, x):
        r, d = x
        a = a + r
        return a * d, a

    return jax.lax.scan(step, acc, (rollout.rewards, rollout.discounts))


def take_host(x):
    return jax.tree.map(lambda y: np.asarray(jax.device_get(y)), x)


def dump(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def save(state: LoopState, cfg: Config, chunk: int, final: bool):
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    learner = take_host(state.learner_state)
    total_steps = int(np.asarray(jax.device_get(state.total_steps)))
    payload = dict(
        format="disocraft_paper_pixels_v1",
        chunk_idx=chunk,
        final=final,
        total_steps=total_steps,
        config={**asdict(cfg), "checkpoint_dir": str(cfg.checkpoint_dir)},
        learner_state=learner,
    )
    dump(payload, cfg.checkpoint_dir / "checkpoint_latest.pkl")
    np.savez(cfg.checkpoint_dir / "policy_params_latest.npz", **flatten_params(learner.params))
    if final:
        dump(payload, cfg.checkpoint_dir / "checkpoint_final.pkl")
        np.savez(cfg.checkpoint_dir / "policy_params_final.npz", **flatten_params(learner.params))


def save_policy_snapshot(state: LoopState, cfg: Config, tag: str):
    params = take_host(state.learner_state.params)
    np.savez(cfg.checkpoint_dir / f"policy_params_{tag}.npz", **flatten_params(params))


def train(cfg: Config) -> None:
    out = cfg.checkpoint_dir
    out.mkdir(parents=True, exist_ok=True)
    metrics = out / "metrics.csv"
    metrics.unlink(missing_ok=True)
    steps_per_iter = cfg.num_envs * cfg.rollout_len
    num_iters = max(1, math.ceil(cfg.target_steps / steps_per_iter))
    cap = min(math.ceil(cfg.buffer_transitions / cfg.rollout_len), cfg.num_envs * num_iters)
    min_buffer = cfg.min_buffer_size or cfg.batch_size
    hist_len = cfg.frame_stack - 1 + cfg.rollout_len
    actual_steps = num_iters * steps_per_iter
    run_config = {**asdict(cfg), "checkpoint_dir": str(out), "actual_steps": actual_steps}
    run_config |= {
        "env": ENV_NAME,
        "replay_buffer_chunks": cap,
        "replay_buffer_transitions_effective": cap * cfg.rollout_len,
    }
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True))
    print(
        "Paper pixels config: "
        f"env={ENV_NAME} envs={cfg.num_envs} batch={cfg.batch_size} "
        f"rollout={cfg.rollout_len} frame_stack={cfg.frame_stack} "
        f"steps_per_iter={steps_per_iter} num_iters={num_iters} actual_steps={actual_steps}"
    )
    print(
        "Replay/optim config: "
        f"replay_fraction={cfg.replay_fraction:.3f} buffer_chunks={cap} "
        f"buffer_transitions={cap * cfg.rollout_len} lr={cfg.learning_rate} "
        f"weight_decay={cfg.weight_decay} max_abs_update={cfg.max_abs_update}"
    )
    print(
        "CNN config: "
        f"channels={cfg.channels} fc={cfg.fc_size} "
        f"model_lstm={cfg.model_lstm_size} model_head={cfg.model_head_size}"
    )

    env = CraftaxPixelsBatch(cfg.num_envs, cfg.frame_stack)
    agent = make_agent(env, cfg)
    update_params = load_update_rule()
    chex.assert_trees_all_equal_shapes_and_dtypes(
        agent.update_rule.init_params(jax.random.PRNGKey(0))[0], update_params
    )

    rng = jax.random.PRNGKey(cfg.seed)
    rng, reset_rng, learner_rng, actor_rng = jax.random.split(rng, 4)
    env_state, ts = env.reset(reset_rng)
    learner = agent.initial_learner_state(learner_rng)
    actor = agent.initial_actor_state(actor_rng)
    raw = to_u8(ts.observation["observation"][..., -3:])
    frame_tail = jnp.repeat(raw[:, None], cfg.frame_stack - 1, axis=1)
    replay = init_replay(cap, hist_len, env.raw_shape, cfg.rollout_len, env.action_spec.maximum + 1)

    def unroll(params, actor_state, timestep, env_state, rng):
        def one(carry, step_rng):
            env_state, timestep, actor_state = carry
            actor_ts, actor_state = agent.actor_step(params, step_rng, timestep, actor_state)
            env_state, timestep = env.step(env_state, actor_ts.actions)
            return (env_state, timestep, actor_state), actor_ts

        (env_state, timestep, actor_state), rollout = jax.lax.scan(
            one, (env_state, timestep, actor_state), jax.random.split(rng, cfg.rollout_len)
        )
        return (
            strip_rollout(types.ActorRollout.from_timestep(rollout)),
            actor_state,
            timestep,
            env_state,
        )

    def log_cb(v):
        names = ("iter", "steps", "avg_return", "loss", "grad_norm", "episodes")
        row = dict(
            zip(names, (int(v[0]), int(v[1]), float(v[2]), float(v[3]), float(v[4]), int(v[5])))
        )
        exists = metrics.exists()
        with metrics.open("a", newline="") as f:
            w = csv.DictWriter(f, names)
            if not exists:
                w.writeheader()
            w.writerow(row)
        print(
            "iter={iter} steps={steps} avg_return={avg_return:.3f} "
            "loss={loss:.4f} grad_norm={grad_norm:.4f} episodes={episodes}".format(**row)
        )

    def train_step(st: LoopState, _):
        rng, rollout_rng, replay_rng, fresh_rng, mix_rng, step_rng = jax.random.split(st.rng, 6)
        rollout, actor, timestep, env_state = unroll(
            st.learner_state.params, st.actor_state, st.timestep, st.env_state, rollout_rng
        )
        rollout_bt = swap_tb(rollout)
        replay, frame_tail = add_replay(st.replay, rollout_bt, st.frame_tail, cap)
        acc, returns = reward_scan(st.acc_rewards, rollout)
        z = jnp.array(0.0, jnp.float32)

        def update(learner):
            replay_batch = sample_replay(
                replay, replay_rng, cfg.batch_size, cfg.rollout_len, cfg.frame_stack
            )
            fresh_batch = sample_on_policy(rollout_bt, fresh_rng, cfg.batch_size)
            batch = mix_batches(
                replay_batch, fresh_batch, mix_rng, cfg.replay_fraction, cfg.batch_size
            )
            learner, _, m = agent.learner_step(
                step_rng, batch, learner, actor, update_params, False
            )
            return learner, (m.get("total_loss", z), m.get("global_gradient_norm", z))

        learner, (loss, grad_norm) = jax.lax.cond(
            replay.size >= min_buffer,
            update,
            lambda learner: (learner, (z, z)),
            st.learner_state,
        )
        total = st.total_steps + steps_per_iter
        idx = total // steps_per_iter
        ep = jnp.sum(1.0 - rollout.discounts)
        ret = jnp.sum(returns * (1.0 - rollout.discounts))
        vals = (idx, total, jnp.where(ep > 0, ret / ep, 0.0), loss, grad_norm, ep)
        jax.lax.cond(
            (cfg.log_every > 0) & (idx % cfg.log_every == 0),
            lambda _: jax.debug.callback(log_cb, vals),
            lambda _: None,
            None,
        )
        return LoopState(
            rng, env_state, timestep, learner, actor, replay, frame_tail, acc, total
        ), None

    state = LoopState(
        rng,
        env_state,
        ts,
        learner,
        actor,
        replay,
        frame_tail,
        jnp.zeros((cfg.num_envs,), jnp.float32),
        jnp.array(0, jnp.int32),
    )
    compiled: dict[int, Any] = {}

    def chunk_fn(n):
        if n not in compiled:
            compiled[n] = jax.jit(lambda s: jax.lax.scan(train_step, s, None, n)[0])
        return compiled[n]

    done = 0
    last_snapshot_idx = 0
    total_chunks = math.ceil(num_iters / cfg.chunk_iterations)
    for chunk in range(1, total_chunks + 1):
        n = min(cfg.chunk_iterations, num_iters - done)
        state = chunk_fn(n)(state)
        done += n
        jax.block_until_ready(state.total_steps)
        total_steps = int(jax.device_get(state.total_steps))
        if cfg.snapshot_every_steps > 0:
            snapshot_idx = total_steps // cfg.snapshot_every_steps
            if snapshot_idx > last_snapshot_idx:
                last_snapshot_idx = snapshot_idx
                tag = f"step_{snapshot_idx * cfg.snapshot_every_steps:09d}"
                save_policy_snapshot(state, cfg, tag)
                print(f"policy_snapshot {tag} steps={total_steps}")
        if chunk % cfg.checkpoint_every_chunks == 0:
            save(state, cfg, chunk, False)
            print(f"checkpoint chunk={chunk} steps={total_steps}")
    save(state, cfg, total_chunks, True)
    print(
        f"final_checkpoint pkl={out / 'checkpoint_final.pkl'} params={out / 'policy_params_final.npz'}"
    )


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
