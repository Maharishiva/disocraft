"""Render one Craftax Classic pixel episode with DiscoRL head overlays."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import distrax
from dm_env import specs
import dm_env
import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from craftax.craftax_classic.constants import Action
from craftax.craftax_env import make_craftax_env_from_name
from disco_rl import types
from train_paper_pixels import ENV_NAME, FRAME_STACK, Config, make_agent, stack_frames


ACTION_NAMES = [a.name.lower().replace("_", " ") for a in Action]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out_gif", type=Path, required=True)
    p.add_argument("--seed", type=int, default=21)
    p.add_argument("--action_mode", choices=("sample", "greedy"), default="sample")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--scale", type=int, default=7)
    p.add_argument("--max_steps", type=int, default=10000)
    return p.parse_args()


def load_params(path: Path):
    with path.open("rb") as f:
        payload = pickle.load(f)
    return payload["learner_state"].params


def entropy(probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-(probs * np.log(probs)).sum())


def draw_bars(draw, xy, values, color, outline, labels=None, max_value=None):
    x, y, w, h = xy
    values = np.asarray(values, dtype=np.float32)
    max_value = float(max_value if max_value is not None else max(values.max(), 1e-6))
    n = len(values)
    gap = 2
    bar_w = max(1, (w - gap * (n - 1)) // n)
    draw.rectangle([x, y, x + w, y + h], outline=outline)
    for i, v in enumerate(values):
        bh = int(h * float(v) / max_value)
        bx = x + i * (bar_w + gap)
        draw.rectangle([bx, y + h - bh, bx + bar_w - 1, y + h], fill=color)
    if labels:
        for i, lab in labels:
            bx = x + i * (bar_w + gap)
            draw.text((bx, y + h + 2), lab, fill=(210, 210, 210))


def draw_hist(draw, xy, values, color, outline):
    counts, _ = np.histogram(np.asarray(values, dtype=np.float32), bins=24)
    draw_bars(draw, xy, counts, color, outline, max_value=max(int(counts.max()), 1))


def draw_line(draw, xy, values, color, outline):
    x, y, w, h = xy
    draw.rectangle([x, y, x + w, y + h], outline=outline)
    if len(values) < 2:
        return
    arr = np.asarray(values, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if abs(hi - lo) < 1e-6:
        hi = lo + 1.0
    pts = []
    for i, v in enumerate(arr):
        px = x + int(i * w / max(len(arr) - 1, 1))
        py = y + h - int((float(v) - lo) * h / (hi - lo))
        pts.append((px, py))
    draw.line(pts, fill=color, width=2)


def topk_text(probs: np.ndarray, k: int = 5) -> str:
    order = np.argsort(probs)[::-1][:k]
    return "  ".join(f"{ACTION_NAMES[i]} {probs[i]:.2f}" for i in order)


def make_panel(
    obs,
    step,
    action,
    reward,
    cum_return,
    achievements,
    policy_probs,
    y_logits,
    z_logits,
    returns,
    scale,
):
    game = Image.fromarray(np.asarray(np.clip(obs * 255.0, 0, 255), np.uint8))
    game = game.resize((game.width * scale, game.height * scale), Image.Resampling.NEAREST)
    panel_w = 470
    img = Image.new("RGB", (game.width + panel_w, game.height), (16, 18, 22))
    img.paste(game, (0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    x0 = game.width + 14
    grey = (185, 190, 198)
    white = (240, 242, 246)
    blue = (88, 166, 255)
    green = (87, 200, 130)
    orange = (255, 176, 80)
    purple = (188, 140, 255)
    outline = (65, 70, 82)

    z_probs = np.asarray(jax.nn.softmax(jnp.asarray(z_logits)), dtype=np.float32)
    y_probs = np.asarray(jax.nn.softmax(jnp.asarray(y_logits)), dtype=np.float32)
    draw.text(
        (x0, 12),
        f"step {step:04d}   action {ACTION_NAMES[action]}",
        fill=white,
        font=font,
    )
    draw.text(
        (x0, 30),
        f"reward {reward:+.2f}   return {cum_return:.2f}   achievements {achievements}/22",
        fill=white,
        font=font,
    )
    draw.text(
        (x0, 48),
        f"H(policy) {entropy(policy_probs):.2f}   H(y) {entropy(y_probs):.2f}   H(z[a]) {entropy(z_probs):.2f}",
        fill=grey,
        font=font,
    )
    draw.text((x0, 72), "cumulative return", fill=grey, font=font)
    draw_line(draw, (x0, 88, panel_w - 30, 52), returns, green, outline)

    y = 158
    draw.text((x0, y), "policy probs over actions", fill=grey, font=font)
    top = np.argsort(policy_probs)[::-1][:3]
    labels = [(int(i), str(int(i))) for i in top]
    draw_bars(draw, (x0, y + 16, panel_w - 30, 66), policy_probs, blue, outline, labels, 1.0)
    draw.text((x0, y + 88), topk_text(policy_probs), fill=white, font=font)

    y = 270
    draw.text((x0, y), "y logits histogram", fill=grey, font=font)
    draw_hist(draw, (x0, y + 16, panel_w - 30, 52), y_logits, orange, outline)
    draw.text(
        (x0, y + 74),
        f"y mean {np.mean(y_logits):+.2f}  std {np.std(y_logits):.2f}",
        fill=white,
        font=font,
    )

    y = 360
    draw.text((x0, y), "z logits histogram for chosen action", fill=grey, font=font)
    draw_hist(draw, (x0, y + 16, panel_w - 30, 52), z_logits, purple, outline)
    draw.text(
        (x0, y + 74),
        f"z mean {np.mean(z_logits):+.2f}  std {np.std(z_logits):.2f}",
        fill=white,
        font=font,
    )
    return np.asarray(img)


def main() -> None:
    args = parse_args()
    params = load_params(args.checkpoint.expanduser().resolve())
    env = make_craftax_env_from_name(ENV_NAME, auto_reset=False)
    env_params = env.default_params
    action_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    fake_env = type(
        "RenderEnvSpec",
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
    agent = make_agent(fake_env, Config(checkpoint_dir=args.out_gif.parent))
    actor_state = agent.initial_actor_state(jax.random.PRNGKey(args.seed + 1000))

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(jax.random.split(reset_rng, 1)[0], env_params)
    frames = jnp.repeat(obs[None, None, ...], FRAME_STACK, axis=1)
    done = False
    cum_return = 0.0
    returns = [0.0]
    gif_frames = []

    step_fn = jax.jit(lambda key, state, action: env.step(key, state, action, env_params))
    for step in range(args.max_steps):
        rng, action_rng, step_rng = jax.random.split(rng, 3)
        ts = types.EnvironmentTimestep(
            observation={"observation": stack_frames(frames).astype(jnp.float32)},
            step_type=jnp.asarray([dm_env.StepType.MID], jnp.int32),
            reward=jnp.zeros((1,), jnp.float32),
        )
        should_reset = ts.step_type == dm_env.StepType.LAST
        outs, actor_state = agent._network.one_step(
            params, actor_state, ts.observation, should_reset
        )
        logits = outs["logits"][0]
        policy_probs = np.asarray(jax.nn.softmax(logits))
        if args.action_mode == "sample":
            action = int(distrax.Softmax(logits=logits).sample(seed=action_rng))
        else:
            action = int(jnp.argmax(logits))
        next_obs, next_state, reward, done, _ = step_fn(
            step_rng, state, jnp.asarray(action, jnp.int32)
        )
        reward_f = float(jax.device_get(reward))
        cum_return += reward_f
        returns.append(cum_return)
        achievements = int(np.asarray(jax.device_get(next_state.achievements)).sum())
        gif_frames.append(
            make_panel(
                np.asarray(jax.device_get(obs)),
                step,
                action,
                reward_f,
                cum_return,
                achievements,
                policy_probs,
                np.asarray(jax.device_get(outs["y"][0])),
                np.asarray(jax.device_get(outs["z"][0, action])),
                returns,
                args.scale,
            )
        )
        obs, state = next_obs, next_state
        frames = jnp.concatenate([frames[:, 1:], obs[None, None, ...]], axis=1)
        if bool(jax.device_get(done)):
            break

    args.out_gif.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out_gif, gif_frames, duration=1 / args.fps, loop=0)
    print(
        f"wrote {args.out_gif} frames={len(gif_frames)} "
        f"return={cum_return:.3f} achievements={achievements}/22 mode={args.action_mode} seed={args.seed}"
    )


if __name__ == "__main__":
    main()
