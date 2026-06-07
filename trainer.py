import csv, json, math, pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import chex, dm_env, jax, jax.numpy as jnp, numpy as np
from craftax.craftax_env import make_craftax_env_from_name
from disco_rl import agent as agent_lib, types
from dm_env import specs

AXIS, ENV, ROLLOUT, LR = "devices", "Craftax-Symbolic-v1", 29, 3e-4
GLOBAL_ENVS, GLOBAL_BATCH, BUFFER_TRANSITIONS = 4, 256, 400_000


@dataclass(frozen=True)
class Config:
    checkpoint_dir: Path
    target_steps: int = 1_000_000_000
    seed: int = 0
    log_every: int = 1000
    chunk_iterations: int = 10_000
    checkpoint_every_chunks: int = 1


@chex.dataclass(mappable_dataclass=False)
class EnvState:
    state: Any
    rng: chex.PRNGKey


@chex.dataclass(mappable_dataclass=False)
class Buffer:
    data: types.ActorRollout
    idx: chex.Array
    size: chex.Array


@chex.dataclass(mappable_dataclass=False)
class LoopState:
    rng: chex.PRNGKey
    env_state: EnvState
    timestep: types.EnvironmentTimestep
    learner_state: agent_lib.LearnerState
    actor_state: Any
    buffer: Buffer
    acc_rewards: chex.Array
    total_steps: chex.Array


class CraftaxBatch:
    def __init__(self, batch: int):
        self.batch, self.env = batch, make_craftax_env_from_name(ENV, auto_reset=False)
        self.params = self.env.default_params
        a, o = self.env.action_space(self.params), self.env.observation_space(self.params)
        self.action_spec = specs.BoundedArray((), np.int32, 0, a.n - 1)
        self.obs_spec = {"observation": specs.Array(o.shape, o.dtype)}
        self.vstep, self.vreset = jax.vmap(self._step), jax.vmap(self._reset)

    def _step(self, s: EnvState, action):
        rng, step_rng, reset_rng = jax.random.split(s.rng, 3)
        obs, state, reward, done, _ = self.env.step(step_rng, s.state, action, self.params)
        reset_obs, reset_state = self.env.reset(reset_rng, self.params)
        state = jax.tree.map(lambda r, x: jax.lax.select(done, r, x), reset_state, state)
        obs = jax.tree.map(lambda r, x: jax.lax.select(done, r, x), reset_obs, obs)
        return EnvState(state, rng), types.EnvironmentTimestep(
            {"observation": jnp.asarray(obs, jnp.float32)},
            jax.lax.select(done, dm_env.StepType.LAST, dm_env.StepType.MID),
            jnp.asarray(reward, jnp.float32),
        )

    def _reset(self, rng):
        rng, reset_rng = jax.random.split(rng)
        obs, state = self.env.reset(reset_rng, self.params)
        return EnvState(state, rng), types.EnvironmentTimestep(
            {"observation": jnp.asarray(obs, jnp.float32)},
            jnp.asarray(dm_env.StepType.MID),
            jnp.asarray(0.0, jnp.float32),
        )

    def step(self, state, actions):
        return self.vstep(state, actions)

    def reset(self, rng):
        return self.vreset(jax.random.split(rng, self.batch))


def make_agent(env: CraftaxBatch) -> agent_lib.Agent:
    s = agent_lib.get_settings_disco()
    s.learning_rate = LR
    s.net_settings.name = "mlp"
    s.net_settings.net_args = dict(
        dense=(512, 512),
        model_arch_name="lstm",
        head_w_init_std=1e-2,
        model_kwargs=dict(head_mlp_hiddens=(128,), lstm_size=128),
    )
    return agent_lib.Agent(s, env.obs_spec, env.action_spec, batch_axis_name=AXIS)


def load_update_rule(path: Path | None = None):
    path = (
        path
        or Path(__file__).parent / "external/disco_rl/disco_rl/update_rules/weights/disco_103.npz"
    )
    with np.load(path) as f:
        flat = dict(f)
    return {k[:-2]: {"w": flat[k], "b": flat[k[:-2] + "/b"]} for k in flat if k.endswith("/w")}


def tb(x):
    return jax.tree.map(lambda y: jnp.swapaxes(y, 0, 1), x)


def strip(r):
    return types.ActorRollout(
        r.observations,
        r.actions,
        r.rewards,
        r.discounts,
        {"logits": r.agent_outs["logits"]},
        r.states,
        r.logits,
    )


def host0(x):
    return jax.tree.map(lambda y: np.asarray(jax.device_get(y[0])), x)


def init_buffer(example, cap):
    data = jax.tree.map(lambda y: jnp.zeros((cap,) + y.shape[1:], y.dtype), tb(example))
    return Buffer(data, jnp.array(0, jnp.int32), jnp.array(0, jnp.int32))


def add_buffer(buf, rollout, cap):
    idx = (buf.idx + jnp.arange(rollout.rewards.shape[0])) % cap
    data = jax.tree.map(lambda b, x: b.at[idx].set(x), buf.data, rollout)
    return Buffer(
        data,
        (buf.idx + rollout.rewards.shape[0]) % cap,
        jnp.minimum(cap, buf.size + rollout.rewards.shape[0]),
    )


def sample(buf, rng, batch):
    idx = jax.random.randint(rng, (batch,), 0, buf.size)
    return tb(jax.tree.map(lambda x: x[idx], buf.data))


def reward_scan(acc, rollout):
    def step(a, x):
        r, d = x
        a = a + r
        return a * d, a

    return jax.lax.scan(step, acc, (rollout.rewards, rollout.discounts))


def dump(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def save(state, cfg: Config, chunk: int, final: bool):
    learner, out = host0(state.learner_state), cfg.checkpoint_dir
    flat = {f"{m}/{k}": np.asarray(v) for m, p in learner.params.items() for k, v in p.items()}
    payload = dict(
        format="disocraft_fixed_pmap_v1",
        chunk_idx=chunk,
        final=final,
        total_steps=int(np.asarray(jax.device_get(state.total_steps[0]))),
        config={**asdict(cfg), "checkpoint_dir": str(cfg.checkpoint_dir)},
        learner_state=learner,
    )
    dump(payload, out / "checkpoint_latest.pkl")
    np.savez(out / "policy_params_latest.npz", **flat)
    if final:
        dump(payload, out / "checkpoint_final.pkl")
        np.savez(out / "policy_params_final.npz", **flat)


def train(cfg: Config) -> None:
    devices = jax.local_devices()[:2]
    if len(devices) != 2:
        raise RuntimeError(f"expected 2 local devices, got {len(devices)}")
    out, local_envs, local_batch = cfg.checkpoint_dir, GLOBAL_ENVS // 2, GLOBAL_BATCH // 2
    out.mkdir(parents=True, exist_ok=True)
    metrics = out / "metrics.csv"
    metrics.unlink(missing_ok=True)
    steps_per_iter, num_iterations = (
        GLOBAL_ENVS * ROLLOUT,
        round(cfg.target_steps / (GLOBAL_ENVS * ROLLOUT)),
    )
    cap = math.ceil((BUFFER_TRANSITIONS // 2) / ROLLOUT)
    actual_steps = num_iterations * steps_per_iter
    (out / "run_config.json").write_text(
        json.dumps(
            {
                **asdict(cfg),
                "checkpoint_dir": str(out),
                "env": ENV,
                "devices": 2,
                "global_envs": GLOBAL_ENVS,
                "global_batch": GLOBAL_BATCH,
                "rollout": ROLLOUT,
                "num_iterations": num_iterations,
                "actual_steps": actual_steps,
                "fifo_replay_ratio": GLOBAL_BATCH / GLOBAL_ENVS,
                "global_buffer_transitions": cap * ROLLOUT * 2,
            },
            indent=2,
        )
    )
    print(
        f"Pmap config: devices=2 local_envs={local_envs} global_envs={GLOBAL_ENVS} local_batch={local_batch} global_batch={GLOBAL_BATCH} rollout_len={ROLLOUT} steps_per_iter={steps_per_iter} num_iterations={num_iterations} actual_env_steps={actual_steps}"
    )
    print(
        f"Replay/checkpoint config: mode=fifo fifo_replay_ratio={GLOBAL_BATCH / GLOBAL_ENVS:.2f} fifo_replay_fraction={1 - GLOBAL_ENVS / GLOBAL_BATCH:.4f} global_buffer_transitions={cap * ROLLOUT * 2} checkpoint_dir={out} metrics_csv={metrics}"
    )
    env, agent, update_params = CraftaxBatch(local_envs), None, load_update_rule()
    agent = make_agent(env)
    chex.assert_trees_all_equal_shapes_and_dtypes(
        agent.update_rule.init_params(jax.random.PRNGKey(0))[0], update_params
    )

    def init_one(rng):
        rng, reset_rng = jax.random.split(rng)
        learner, actor = (
            agent.initial_learner_state(jax.random.PRNGKey(cfg.seed + 2027)),
            agent.initial_actor_state(jax.random.PRNGKey(cfg.seed + 2028)),
        )
        env_state, ts = env.reset(reset_rng)
        obs = jax.tree.map(lambda s: jnp.zeros((local_envs,) + s.shape, s.dtype), env.obs_spec)
        dummy = types.EnvironmentTimestep(
            obs, jnp.zeros((local_envs,), jnp.int32), jnp.zeros((local_envs,), jnp.float32)
        )
        actor_ts, _ = agent.actor_step(learner.params, jax.random.PRNGKey(0), dummy, actor)
        zeros = jax.tree.map(lambda x: jnp.zeros((ROLLOUT,) + x.shape, x.dtype), actor_ts)
        return LoopState(
            rng,
            env_state,
            ts,
            learner,
            actor,
            init_buffer(strip(types.ActorRollout.from_timestep(zeros)), cap),
            jnp.zeros((local_envs,), jnp.float32),
            jnp.array(0, jnp.int64),
        )

    def unroll(params, actor, ts, env_state, rng):
        def step(carry, r):
            env_state, ts, actor = carry
            actor_ts, actor = agent.actor_step(params, r, ts, actor)
            env_state, ts = env.step(env_state, actor_ts.actions)
            return (env_state, ts, actor), actor_ts

        (env_state, ts, actor), rollout = jax.lax.scan(
            step, (env_state, ts, actor), jax.random.split(rng, ROLLOUT)
        )
        return strip(types.ActorRollout.from_timestep(rollout)), actor, ts, env_state

    def log_cb(v):
        names = ("iter", "steps", "avg_return", "loss", "grad_norm", "episodes")
        row = dict(
            zip(names, (int(v[0]), int(v[1]), float(v[2]), float(v[3]), float(v[4]), float(v[5])))
        )
        exists = metrics.exists()
        with metrics.open("a", newline="") as f:
            w = csv.DictWriter(f, names)
            if not exists:
                w.writeheader()
            w.writerow(row)
        print(
            "iter={iter} steps={steps} avg_return={avg_return:.3f} loss={loss:.4f} grad_norm={grad_norm:.4f} episodes={episodes:.0f}".format(
                **row
            )
        )

    def train_step(st, _):
        rng, rollout_rng, update_rng = jax.random.split(st.rng, 3)
        rollout, actor, ts, env_state = unroll(
            st.learner_state.params, st.actor_state, st.timestep, st.env_state, rollout_rng
        )
        buf = add_buffer(st.buffer, tb(rollout), cap)
        acc, returns = reward_scan(st.acc_rewards, rollout)
        sample_rng, step_rng = jax.random.split(update_rng)
        z = jnp.array(0.0, jnp.float32)

        def update(learner):
            batch = sample(buf, sample_rng, local_batch)
            learner, _, m = agent.learner_step(
                step_rng, batch, learner, actor, update_params, False
            )
            return learner, (m.get("total_loss", z), m.get("global_gradient_norm", z))

        learner, (loss, grad_norm) = jax.lax.cond(
            buf.size >= local_batch, update, lambda learner: (learner, (z, z)), st.learner_state
        )
        total = st.total_steps + steps_per_iter
        idx = total // steps_per_iter
        ret_sum, eps = (
            jax.lax.psum(jnp.sum(returns * (1.0 - rollout.discounts)), AXIS),
            jax.lax.psum(jnp.sum(1.0 - rollout.discounts), AXIS),
        )
        vals = (
            idx,
            total,
            jnp.where(eps > 0, ret_sum / eps, 0.0),
            jax.lax.pmean(loss, AXIS),
            jax.lax.pmean(grad_norm, AXIS),
            eps,
        )
        jax.lax.cond(
            (idx % cfg.log_every == 0) & (jax.lax.axis_index(AXIS) == 0),
            lambda _: jax.debug.callback(log_cb, vals),
            lambda _: None,
            None,
        )
        return LoopState(rng, env_state, ts, learner, actor, buf, acc, total), None

    init = jax.pmap(init_one, axis_name=AXIS, devices=devices)
    pmaps = {}

    def chunk_fn(n):
        if n not in pmaps:
            pmaps[n] = jax.pmap(
                lambda s: jax.lax.scan(train_step, s, None, n)[0],
                axis_name=AXIS,
                devices=devices,
                donate_argnums=(0,),
            )
        return pmaps[n]

    state = init(jax.random.split(jax.random.PRNGKey(cfg.seed), 2))
    done, total_chunks = 0, math.ceil(num_iterations / cfg.chunk_iterations)
    for chunk in range(1, total_chunks + 1):
        n = min(cfg.chunk_iterations, num_iterations - done)
        state = chunk_fn(n)(state)
        done += n
        jax.block_until_ready(state.total_steps)
        if chunk % cfg.checkpoint_every_chunks == 0:
            save(state, cfg, chunk, False)
            print(f"checkpoint chunk={chunk}")
    save(state, cfg, total_chunks, True)
    print(
        f"final_checkpoint pkl={out / 'checkpoint_final.pkl'} params={out / 'policy_params_final.npz'}"
    )
