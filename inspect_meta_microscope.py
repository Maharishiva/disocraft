"""Render a DiscoRL meta-network microscope from one training-shaped rollout."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import chex
import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from craftax.craftax_classic.constants import Action
from disco_rl import types
from disco_rl import utils as disco_utils
from train_paper_pixels import (
    Config,
    CraftaxPixelsBatch,
    load_update_rule,
    make_agent,
    strip_rollout,
    swap_tb,
)


ACTION_NAMES = [a.name.lower().replace("_", " ") for a in Action]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=7000)
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--rollout_len", type=int, default=29)
    p.add_argument("--frame_stack", type=int, default=32)
    p.add_argument("--scale", type=int, default=7)
    p.add_argument("--fps", type=int, default=4)
    return p.parse_args()


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def entropy_np(probs: np.ndarray, axis: int = -1) -> np.ndarray:
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.sum(probs * np.log(probs), axis=axis)


def categorical_kl_np(target_logits: np.ndarray, online_logits: np.ndarray) -> np.ndarray:
    target = softmax_np(target_logits)
    online_logp = np.log(np.clip(softmax_np(online_logits), 1e-12, 1.0))
    target_logp = np.log(np.clip(target, 1e-12, 1.0))
    return np.sum(target * (target_logp - online_logp), axis=-1)


def take(x):
    return jax.tree.map(lambda y: np.asarray(jax.device_get(y)), x)


def load_checkpoint(path: Path):
    with path.open("rb") as f:
        payload = pickle.load(f)
    return payload, jax.tree.map(jnp.asarray, payload["learner_state"])


def make_cfg(args: argparse.Namespace, payload: dict) -> Config:
    saved = payload.get("config", {})
    channels = tuple(saved.get("channels", (256, 384, 384, 256)))
    return Config(
        checkpoint_dir=args.out_dir.resolve(),
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        rollout_len=args.rollout_len,
        frame_stack=args.frame_stack,
        replay_fraction=float(saved.get("replay_fraction", 0.99)),
        buffer_transitions=int(saved.get("buffer_transitions", 100_000)),
        channels=channels,
        fc_size=int(saved.get("fc_size", 768)),
        model_lstm_size=int(saved.get("model_lstm_size", 1024)),
        model_head_size=int(saved.get("model_head_size", 1024)),
    )


def sample_fresh_batch(agent, env, cfg: Config, learner_state, rng):
    rng, reset_rng, actor_rng, rollout_rng, sample_rng = jax.random.split(rng, 5)
    env_state, timestep = env.reset(reset_rng)
    actor = agent.initial_actor_state(actor_rng)

    def one(carry, step_rng):
        env_state, timestep, actor = carry
        actor_ts, actor = agent.actor_step(
            learner_state.params, step_rng, timestep, actor
        )
        env_state, timestep = env.step(env_state, actor_ts.actions)
        return (env_state, timestep, actor), actor_ts

    (_, _, actor_after), rollout_ts = jax.lax.scan(
        one,
        (env_state, timestep, actor),
        jax.random.split(rollout_rng, cfg.rollout_len),
    )
    rollout = strip_rollout(types.ActorRollout.from_timestep(rollout_ts))
    rollout_bt = swap_tb(rollout)
    idx = jax.random.randint(sample_rng, (cfg.batch_size,), 0, cfg.num_envs)
    batch = swap_tb(jax.tree.map(lambda x: x[idx], rollout_bt))
    source_returns = jnp.sum(rollout_bt.rewards, axis=1)
    return batch, actor_after, idx, source_returns


def inspect_batch(agent, cfg: Config, learner_state, batch, actor_after, rng):
    update_params = load_update_rule()
    agent_out, _ = agent.unroll_net(learner_state.params, actor_after, batch)
    eta_inputs = types.UpdateRuleInputs(
        observations=batch.observations,
        actions=batch.actions,
        rewards=batch.rewards[1:],
        is_terminal=batch.discounts[1:] == 0,
        behaviour_agent_out=batch.agent_outs,
        agent_out=agent_out,
        value_out=None,
    )
    meta_out, _ = agent.update_rule.unroll_meta_net(
        meta_params=update_params,
        params=learner_state.params,
        state=actor_after,
        meta_state=learner_state.meta_state,
        rollout=eta_inputs,
        hyper_params=agent.settings.hyper_params.to_dict(),
        unroll_policy_fn=agent._network.unroll,
        rng=rng,
        axis_name=None,
    )
    loss_meta, _ = agent.update_rule.agent_loss(
        eta_inputs,
        meta_out,
        agent.settings.hyper_params.to_dict(),
        backprop=False,
    )
    loss_value, _ = agent.update_rule.agent_loss_no_meta(
        eta_inputs,
        meta_out,
        agent.settings.hyper_params.to_dict(),
    )
    actions = batch.actions[:-1]
    z_a = disco_utils.batch_lookup(agent_out["z"][:-1], actions)
    return {
        "batch": take(batch),
        "agent_out": take(agent_out),
        "meta_out": take(meta_out),
        "loss_meta": take(loss_meta),
        "loss_value": take(loss_value),
        "z_a": take(z_a),
    }


def draw_bars(draw, xy, values, fill, outline, center_zero=False):
    x, y, w, h = xy
    values = np.asarray(values, dtype=np.float32)
    n = len(values)
    gap = 2
    bar_w = max(2, (w - gap * (n - 1)) // n)
    draw.rectangle([x, y, x + w, y + h], outline=outline)
    if center_zero:
        mid = y + h // 2
        draw.line([x, mid, x + w, mid], fill=outline)
        scale = h * 0.5 / max(np.max(np.abs(values)), 1e-6)
        for i, v in enumerate(values):
            bx = x + i * (bar_w + gap)
            by = int(mid - v * scale)
            draw.rectangle([bx, min(mid, by), bx + bar_w - 1, max(mid, by)], fill=fill)
    else:
        scale = h / max(np.max(values), 1e-6)
        for i, v in enumerate(values):
            bh = int(v * scale)
            bx = x + i * (bar_w + gap)
            draw.rectangle([bx, y + h - bh, bx + bar_w - 1, y + h], fill=fill)


def draw_hist(draw, xy, values, fill, outline, bins=40):
    counts, _ = np.histogram(np.asarray(values, dtype=np.float32), bins=bins)
    draw_bars(draw, xy, counts, fill, outline)


def draw_series(draw, xy, series, current, fill, outline):
    x, y, w, h = xy
    arr = np.asarray(series, dtype=np.float32)
    draw.rectangle([x, y, x + w, y + h], outline=outline)
    if arr.size < 2:
        return
    lo, hi = float(arr.min()), float(arr.max())
    if abs(hi - lo) < 1e-6:
        hi = lo + 1.0
    pts = []
    for i, v in enumerate(arr):
        px = x + int(i * w / max(arr.size - 1, 1))
        py = y + h - int((float(v) - lo) * h / (hi - lo))
        pts.append((px, py))
    draw.line(pts, fill=fill, width=2)
    cx = x + int(current * w / max(arr.size - 1, 1))
    draw.line([cx, y, cx, y + h], fill=(255, 215, 120), width=2)


def topk_text(probs: np.ndarray, k: int = 4) -> str:
    order = np.argsort(probs)[::-1][:k]
    return "  ".join(f"{ACTION_NAMES[i]} {probs[i]:.2f}" for i in order)


def make_frame(obs, data, t, chosen_b, scale):
    current_probs = data["current_probs"][:, chosen_b]
    target_probs = data["target_probs"][:, chosen_b]
    push = target_probs - current_probs
    action = int(data["actions"][t, chosen_b])
    reward = float(data["rewards"][t, chosen_b])
    done = bool(data["terminal"][t, chosen_b])
    cum_reward = float(np.sum(data["rewards"][: t + 1, chosen_b]))

    game = Image.fromarray(np.asarray(np.clip(obs * 255, 0, 255), np.uint8))
    game = game.resize((game.width * scale, game.height * scale), Image.Resampling.NEAREST)
    panel_w = 720
    img_h = max(game.height, 620)
    img = Image.new("RGB", (game.width + panel_w, img_h), (16, 18, 22))
    img.paste(game, (0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    x0 = game.width + 14
    white = (238, 241, 246)
    grey = (174, 181, 194)
    outline = (70, 76, 88)
    blue = (83, 158, 255)
    orange = (255, 169, 77)
    green = (91, 205, 137)
    red = (245, 96, 96)
    purple = (185, 135, 255)

    draw.text((x0, 10), "DiscoRL meta-network microscope", fill=white, font=font)
    draw.text(
        (x0, 28),
        (
            f"fresh training-shaped batch  step {t:02d}/{len(data['rewards']) - 1:02d}  "
            f"action {ACTION_NAMES[action]}  reward {reward:+.2f}  return {cum_reward:.2f}"
        ),
        fill=white,
        font=font,
    )
    draw.text(
        (x0, 46),
        f"terminal {done}  selected batch row {chosen_b}  source env {int(data['source_idx'][chosen_b])}",
        fill=grey,
        font=font,
    )

    y = 72
    draw.text((x0, y), "current policy probs", fill=grey, font=font)
    draw_bars(draw, (x0, y + 16, 300, 58), current_probs[t], blue, outline)
    draw.text((x0 + 318, y + 22), topk_text(current_probs[t]), fill=white, font=font)
    y += 92
    draw.text((x0, y), "meta target pi_hat probs", fill=grey, font=font)
    draw_bars(draw, (x0, y + 16, 300, 58), target_probs[t], orange, outline)
    draw.text((x0 + 318, y + 22), topk_text(target_probs[t]), fill=white, font=font)
    y += 92
    draw.text((x0, y), "policy push = target - current", fill=grey, font=font)
    draw_bars(draw, (x0, y + 16, 300, 58), push[t], green, outline, center_zero=True)
    strongest = int(np.argmax(np.abs(push[t])))
    draw.text(
        (x0 + 318, y + 22),
        f"strongest {ACTION_NAMES[strongest]} {push[t, strongest]:+.3f}",
        fill=white,
        font=font,
    )

    y += 96
    draw.text(
        (x0, y),
        (
            f"KL(pi_hat||policy) {data['pi_kl'][t, chosen_b]:.3f}   "
            f"H(policy) {data['policy_entropy'][t, chosen_b]:.2f}   "
            f"H(target) {data['target_entropy'][t, chosen_b]:.2f}"
        ),
        fill=white,
        font=font,
    )
    draw.text(
        (x0, y + 18),
        (
            f"adv {data['adv'][t, chosen_b]:+.2f}   "
            f"norm_adv {data['normalized_adv'][t, chosen_b]:+.2f}   "
            f"q_td {data['q_td'][t, chosen_b]:+.2f}"
        ),
        fill=white,
        font=font,
    )
    y += 48
    draw.text((x0, y), "time series: return / adv / KL", fill=grey, font=font)
    draw_series(draw, (x0, y + 16, 210, 48), data["cum_rewards"][:, chosen_b], t, green, outline)
    draw_series(draw, (x0 + 235, y + 16, 210, 48), data["adv"][:, chosen_b], t, orange, outline)
    draw_series(draw, (x0 + 470, y + 16, 210, 48), data["pi_kl"][:, chosen_b], t, purple, outline)

    y += 86
    draw.text(
        (x0, y),
        f"y target/current KL {data['y_kl'][t, chosen_b]:.3f}   z target/current KL {data['z_kl'][t, chosen_b]:.3f}",
        fill=white,
        font=font,
    )
    y += 24
    draw.text((x0, y), "y_hat logits histogram", fill=grey, font=font)
    draw_hist(draw, (x0, y + 16, 315, 46), data["y_hat"][t, chosen_b], red, outline)
    draw.text((x0 + 350, y), "z_hat logits histogram for action", fill=grey, font=font)
    draw_hist(draw, (x0 + 350, y + 16, 315, 46), data["z_hat"][t, chosen_b], purple, outline)
    return np.asarray(img)


def render_gif(out_dir: Path, trace: dict, args: argparse.Namespace) -> Path:
    rewards = trace["batch"].rewards[1:]
    returns = rewards.sum(axis=0)
    chosen_b = int(np.argmax(returns))
    obs = trace["batch"].observations["observation"][:-1, chosen_b, :, :, -3:]
    agent_out = trace["agent_out"]
    meta_out = trace["meta_out"]
    current_logits = agent_out["logits"][:-1]
    target_logits = meta_out["pi"]
    y_logits = agent_out["y"][:-1]
    z_logits = trace["z_a"]
    y_hat = meta_out["y"]
    z_hat = meta_out["z"]
    data = {
        "actions": trace["batch"].actions[:-1],
        "rewards": rewards,
        "terminal": trace["batch"].discounts[1:] == 0,
        "source_idx": trace["source_idx"],
        "current_probs": softmax_np(current_logits),
        "target_probs": softmax_np(target_logits),
        "policy_entropy": entropy_np(softmax_np(current_logits)),
        "target_entropy": entropy_np(softmax_np(target_logits)),
        "pi_kl": categorical_kl_np(target_logits, current_logits),
        "y_kl": categorical_kl_np(y_hat, y_logits),
        "z_kl": categorical_kl_np(z_hat, z_logits),
        "adv": meta_out["adv"],
        "normalized_adv": meta_out["normalized_adv"],
        "q_td": meta_out["q_td"],
        "cum_rewards": np.cumsum(rewards, axis=0),
        "y_hat": y_hat,
        "z_hat": z_hat,
    }
    frames = [
        make_frame(obs[t], data, t, chosen_b, args.scale)
        for t in range(data["rewards"].shape[0])
    ]
    gif_path = out_dir / "meta_microscope.gif"
    imageio.mimsave(gif_path, frames, duration=1 / args.fps, loop=0)
    return gif_path


def plot_timeseries(out_dir: Path, trace: dict) -> Path:
    rewards = trace["batch"].rewards[1:]
    chosen_b = int(np.argmax(rewards.sum(axis=0)))
    agent_out = trace["agent_out"]
    meta_out = trace["meta_out"]
    current_logits = agent_out["logits"][:-1, chosen_b]
    target_logits = meta_out["pi"][:, chosen_b]
    y_logits = agent_out["y"][:-1, chosen_b]
    z_logits = trace["z_a"][:, chosen_b]
    y_hat = meta_out["y"][:, chosen_b]
    z_hat = meta_out["z"][:, chosen_b]
    t = np.arange(target_logits.shape[0])
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), dpi=150, sharex=True)
    axes[0].plot(t, np.cumsum(rewards[:, chosen_b]), label="cum reward")
    axes[0].plot(t, meta_out["adv"][:, chosen_b], label="adv")
    axes[0].plot(t, meta_out["q_td"][:, chosen_b], label="q_td")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(t, categorical_kl_np(target_logits, current_logits), label="pi KL")
    axes[1].plot(t, categorical_kl_np(y_hat, y_logits), label="y KL")
    axes[1].plot(t, categorical_kl_np(z_hat, z_logits), label="z KL")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    axes[2].plot(t, entropy_np(softmax_np(current_logits)), label="policy entropy")
    axes[2].plot(t, entropy_np(softmax_np(target_logits)), label="target entropy")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    push = softmax_np(target_logits) - softmax_np(current_logits)
    push_abs = max(float(np.max(np.abs(push))), 1e-6)
    im = axes[3].imshow(
        push.T, aspect="auto", cmap="coolwarm", vmin=-push_abs, vmax=push_abs
    )
    axes[3].set_yticks(np.arange(len(ACTION_NAMES)))
    axes[3].set_yticklabels(ACTION_NAMES, fontsize=7)
    axes[3].set_xlabel("rollout step")
    axes[3].set_title("policy push: pi_hat - policy")
    fig.colorbar(im, ax=axes[3], fraction=0.025)
    fig.tight_layout()
    path = out_dir / "meta_timeseries.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_weight_norms(out_dir: Path) -> Path:
    params = load_update_rule()
    rows = []
    for module, vals in params.items():
        for name, arr in vals.items():
            arr = np.asarray(arr)
            rows.append((f"{module}/{name}", float(np.linalg.norm(arr)), arr.size))
    rows.sort(key=lambda x: x[1], reverse=True)
    top = rows[:35]
    labels = [x[0].replace("disco_update_rule/~/", "") for x in top]
    norms = [x[1] for x in top]
    fig, ax = plt.subplots(figsize=(9, 9), dpi=150)
    ax.barh(np.arange(len(top)), norms, color="#5b9bd5")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("L2 norm")
    ax.set_title("Disco-103 meta-network parameter norms, top 35")
    fig.tight_layout()
    path = out_dir / "meta_weight_norms.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload, learner_state = load_checkpoint(args.checkpoint.expanduser().resolve())
    cfg = make_cfg(args, payload)
    env = CraftaxPixelsBatch(cfg.num_envs, cfg.frame_stack)
    agent = make_agent(env, cfg)

    def run(seed):
        rng = jax.random.PRNGKey(seed)
        batch, actor_after, source_idx, source_returns = sample_fresh_batch(
            agent, env, cfg, learner_state, rng
        )
        trace = inspect_batch(agent, cfg, learner_state, batch, actor_after, rng)
        trace["source_idx"] = take(source_idx)
        trace["source_returns"] = take(source_returns)
        return trace

    trace = run(args.seed)
    gif_path = render_gif(out_dir, trace, args)
    timeseries_path = plot_timeseries(out_dir, trace)
    weights_path = plot_weight_norms(out_dir)
    np.savez_compressed(
        out_dir / "meta_microscope_trace.npz",
        actions=trace["batch"].actions,
        rewards=trace["batch"].rewards,
        discounts=trace["batch"].discounts,
        current_logits=trace["agent_out"]["logits"],
        target_logits=trace["meta_out"]["pi"],
        y=trace["agent_out"]["y"],
        y_hat=trace["meta_out"]["y"],
        z_a=trace["z_a"],
        z_hat=trace["meta_out"]["z"],
        adv=trace["meta_out"]["adv"],
        normalized_adv=trace["meta_out"]["normalized_adv"],
        q_td=trace["meta_out"]["q_td"],
        loss_meta=trace["loss_meta"],
        loss_value=trace["loss_value"],
        source_idx=trace["source_idx"],
        source_returns=trace["source_returns"],
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_total_steps": int(payload.get("total_steps", -1)),
        "caveat": "Fresh on-policy rollout shaped like training; replay buffer rows are unavailable because checkpoints do not save replay.",
        "num_envs": cfg.num_envs,
        "batch_size": cfg.batch_size,
        "rollout_len": cfg.rollout_len,
        "meta_steps": cfg.rollout_len - 1,
        "gif_path": str(gif_path),
        "timeseries_path": str(timeseries_path),
        "weight_norms_path": str(weights_path),
        "mean_pi_kl": float(np.mean(categorical_kl_np(trace["meta_out"]["pi"], trace["agent_out"]["logits"][:-1]))),
        "mean_policy_entropy": float(np.mean(entropy_np(softmax_np(trace["agent_out"]["logits"][:-1])))),
        "mean_target_entropy": float(np.mean(entropy_np(softmax_np(trace["meta_out"]["pi"])))),
        "mean_adv": float(np.mean(trace["meta_out"]["adv"])),
        "mean_q_td": float(np.mean(trace["meta_out"]["q_td"])),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
