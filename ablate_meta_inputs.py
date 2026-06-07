"""Counterfactual ablations for DiscoRL meta-network inputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from disco_rl import types
from disco_rl.value_fns import value_utils
from inspect_meta_microscope import (
    ACTION_NAMES,
    categorical_kl_np,
    entropy_np,
    load_checkpoint,
    make_cfg,
    sample_fresh_batch,
    softmax_np,
    take,
)
from train_paper_pixels import Config, CraftaxPixelsBatch, load_update_rule, make_agent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=7600)
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--rollout_len", type=int, default=29)
    p.add_argument("--frame_stack", type=int, default=32)
    p.add_argument("--scale", type=int, default=7)
    p.add_argument("--fps", type=int, default=4)
    return p.parse_args()


def replace_eta(eta: types.UpdateRuleInputs, **kwargs) -> types.UpdateRuleInputs:
    data = dict(
        observations=eta.observations,
        actions=eta.actions,
        rewards=eta.rewards,
        is_terminal=eta.is_terminal,
        behaviour_agent_out=eta.behaviour_agent_out,
        agent_out=eta.agent_out,
        value_out=eta.value_out,
        extra_from_rule=eta.extra_from_rule,
    )
    data.update(kwargs)
    return types.UpdateRuleInputs(**data)


def replace_agent_out(agent_out: dict, **kwargs) -> dict:
    out = dict(agent_out)
    out.update(kwargs)
    return out


def zero_tree(x):
    return jax.tree.map(jnp.zeros_like, x)


def compute_base(agent, learner_state, batch, actor_after):
    agent_out, _ = agent.unroll_net(learner_state.params, actor_after, batch)
    eta = types.UpdateRuleInputs(
        observations=batch.observations,
        actions=batch.actions,
        rewards=batch.rewards[1:],
        is_terminal=batch.discounts[1:] == 0,
        behaviour_agent_out=batch.agent_outs,
        agent_out=agent_out,
        value_out=None,
    )
    target_out, _ = agent._network.unroll(
        learner_state.meta_state["target_params"],
        actor_after,
        eta.observations,
        eta.should_reset_mask_fwd,
    )
    hyper = agent.settings.hyper_params.to_dict()
    value_outs, _, _ = value_utils.get_value_outs(
        value_net_out=None,
        target_value_net_out=None,
        q_net_out=agent_out["q"],
        target_q_net_out=target_out["q"],
        rollout=eta,
        pi_logits=agent_out["logits"],
        discount=agent.update_rule._value_discount,
        lambda_=hyper["value_fn_td_lambda"],
        nonlinear_transform=True,
        categorical_value=True,
        max_abs_value=agent.update_rule._max_abs_value,
        drop_last=False,
        adv_ema_state=learner_state.meta_state["adv_ema_state"],
        adv_ema_fn=agent.update_rule._adv_ema,
        td_ema_state=learner_state.meta_state["td_ema_state"],
        td_ema_fn=agent.update_rule._td_ema,
        axis_name=None,
    )
    extra = dict(
        v_scalar=value_outs.value,
        adv=value_outs.adv,
        normalized_adv=value_outs.normalized_adv,
        q=value_outs.target_q_value,
        qv_adv=value_outs.qv_adv,
        normalized_qv_adv=value_outs.normalized_qv_adv,
        target_out=target_out,
    )
    return eta, extra, value_outs


def apply_meta(agent, update_params, meta_state, eta, extra):
    rollout = replace_eta(eta, extra_from_rule=extra)
    meta_out, _ = agent.update_rule._eta_apply(
        update_params, meta_state["rnn_state"], None, rollout, axis_name=None
    )
    return meta_out


def make_cases(eta, extra, interesting_t: int, interesting_b: int):
    target_zero = zero_tree(extra["target_out"])
    target_current = eta.agent_out
    reward_plus = eta.rewards.at[interesting_t, interesting_b].add(1.0)
    reward_minus = eta.rewards.at[interesting_t, interesting_b].add(-1.0)
    terminal_here = eta.is_terminal.at[interesting_t, interesting_b].set(True)
    return {
        "baseline": (eta, extra),
        "no_reward_input": (replace_eta(eta, rewards=jnp.zeros_like(eta.rewards)), extra),
        "reward_plus_frame": (replace_eta(eta, rewards=reward_plus), extra),
        "reward_minus_frame": (replace_eta(eta, rewards=reward_minus), extra),
        "terminal_at_frame": (replace_eta(eta, is_terminal=terminal_here), extra),
        "no_advantage": (
            eta,
            extra
            | {
                "adv": jnp.zeros_like(extra["adv"]),
                "normalized_adv": jnp.zeros_like(extra["normalized_adv"]),
            },
        ),
        "no_value_q": (
            eta,
            extra
            | {
                "v_scalar": jnp.zeros_like(extra["v_scalar"]),
                "q": jnp.zeros_like(extra["q"]),
                "qv_adv": jnp.zeros_like(extra["qv_adv"]),
                "normalized_qv_adv": jnp.zeros_like(extra["normalized_qv_adv"]),
            },
        ),
        "target_equals_current": (eta, extra | {"target_out": target_current}),
        "no_target_features": (eta, extra | {"target_out": target_zero}),
        "behaviour_equals_current": (
            replace_eta(eta, behaviour_agent_out=eta.agent_out),
            extra,
        ),
        "uniform_current_policy": (
            replace_eta(
                eta,
                agent_out=replace_agent_out(
                    eta.agent_out, logits=jnp.zeros_like(eta.agent_out["logits"])
                ),
            ),
            extra,
        ),
        "zero_y_input": (
            replace_eta(
                eta,
                agent_out=replace_agent_out(
                    eta.agent_out, y=jnp.zeros_like(eta.agent_out["y"])
                ),
            ),
            extra,
        ),
        "zero_z_input": (
            replace_eta(
                eta,
                agent_out=replace_agent_out(
                    eta.agent_out, z=jnp.zeros_like(eta.agent_out["z"])
                ),
            ),
            extra,
        ),
    }


def summarize_cases(case_logits: dict[str, np.ndarray], base_logits: np.ndarray, t: int, b: int):
    base_probs = softmax_np(base_logits)
    rows = []
    for name, logits in case_logits.items():
        probs = softmax_np(logits)
        delta = probs - base_probs
        selected_delta = delta[t, b]
        top = int(np.argmax(np.abs(selected_delta)))
        rows.append(
            {
                "case": name,
                "mean_l1_prob_delta": float(np.mean(np.sum(np.abs(delta), axis=-1))),
                "mean_kl_from_baseline": float(np.mean(categorical_kl_np(base_logits, logits))),
                "selected_l1_prob_delta": float(np.sum(np.abs(selected_delta))),
                "selected_top_changed_action": ACTION_NAMES[top],
                "selected_top_delta": float(selected_delta[top]),
                "selected_target_entropy": float(entropy_np(probs[t, b])),
            }
        )
    rows.sort(key=lambda r: r["mean_l1_prob_delta"], reverse=True)
    return rows


def draw_heatmap(draw, x, y, matrix, row_names, col_names, cell_w=36, cell_h=24):
    max_abs = max(float(np.max(np.abs(matrix))), 1e-6)
    font = ImageFont.load_default()
    for j, name in enumerate(col_names):
        draw.text((x + 160 + j * cell_w, y - 14), str(j), fill=(180, 186, 196), font=font)
    for i, name in enumerate(row_names):
        draw.text((x, y + i * cell_h + 6), name[:22], fill=(230, 233, 238), font=font)
        for j, val in enumerate(matrix[i]):
            mag = min(abs(float(val)) / max_abs, 1.0)
            if val >= 0:
                color = (int(45 + 40 * mag), int(75 + 130 * mag), int(110 + 120 * mag))
            else:
                color = (int(100 + 145 * mag), int(55 + 45 * mag), int(60 + 45 * mag))
            x0 = x + 160 + j * cell_w
            y0 = y + i * cell_h
            draw.rectangle([x0, y0, x0 + cell_w - 2, y0 + cell_h - 2], fill=color)
            if abs(float(val)) > 0.015:
                draw.text((x0 + 3, y0 + 6), f"{val:+.2f}", fill=(245, 245, 245), font=font)


def draw_bars(draw, xy, values, color, outline):
    x, y, w, h = xy
    values = np.asarray(values, dtype=np.float32)
    n = len(values)
    gap = 2
    bar_w = max(2, (w - gap * (n - 1)) // n)
    draw.rectangle([x, y, x + w, y + h], outline=outline)
    for i, v in enumerate(values):
        bh = int(h * float(v) / max(float(np.max(values)), 1e-6))
        bx = x + i * (bar_w + gap)
        draw.rectangle([bx, y + h - bh, bx + bar_w - 1, y + h], fill=color)


def topk_text(probs, k=4):
    order = np.argsort(probs)[::-1][:k]
    return "  ".join(f"{ACTION_NAMES[i]} {probs[i]:.2f}" for i in order)


def render_gif(out_dir, trace, case_logits, rows, interesting, args):
    t_star, b_star = interesting
    batch = trace["batch"]
    obs = batch.observations["observation"][:-1, b_star, :, :, -3:]
    current_probs = softmax_np(trace["agent_out"]["logits"][:-1])[:, b_star]
    base_probs = softmax_np(case_logits["baseline"])[:, b_star]
    case_names = [r["case"] for r in rows if r["case"] != "baseline"][:12]
    frames = []
    for t in range(base_probs.shape[0]):
        game = Image.fromarray(np.asarray(np.clip(obs[t] * 255, 0, 255), np.uint8))
        game = game.resize((game.width * args.scale, game.height * args.scale), Image.Resampling.NEAREST)
        panel_w = 890
        h = max(game.height, 650)
        img = Image.new("RGB", (game.width + panel_w, h), (16, 18, 22))
        img.paste(game, (0, 0))
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        x0 = game.width + 14
        white = (238, 241, 246)
        grey = (176, 183, 194)
        outline = (67, 73, 86)
        blue = (82, 155, 250)
        orange = (255, 171, 78)
        draw.text((x0, 10), "DiscoRL causal ablation: effect on pi_hat", fill=white, font=font)
        draw.text(
            (x0, 28),
            f"row {b_star}, step {t:02d}; interesting frame is step {t_star:02d}; action {ACTION_NAMES[int(batch.actions[:-1][t, b_star])]}",
            fill=white,
            font=font,
        )
        draw.text(
            (x0, 46),
            f"reward {float(batch.rewards[1:][t, b_star]):+.2f}; baseline target entropy {entropy_np(base_probs[t]):.3f}",
            fill=grey,
            font=font,
        )
        draw.text((x0, 74), "current policy", fill=grey, font=font)
        draw_bars(draw, (x0, 92, 330, 52), current_probs[t], blue, outline)
        draw.text((x0 + 350, 104), topk_text(current_probs[t]), fill=white, font=font)
        draw.text((x0, 158), "baseline meta target pi_hat", fill=grey, font=font)
        draw_bars(draw, (x0, 176, 330, 52), base_probs[t], orange, outline)
        draw.text((x0 + 350, 188), topk_text(base_probs[t]), fill=white, font=font)

        matrix = np.stack(
            [softmax_np(case_logits[name])[:, b_star][t] - base_probs[t] for name in case_names],
            axis=0,
        )
        draw.text(
            (x0, 250),
            "ablation effect: target_probs(case) - target_probs(baseline)",
            fill=grey,
            font=font,
        )
        draw_heatmap(draw, x0, 280, matrix, case_names, ACTION_NAMES)
        draw.text(
            (x0 + 160, 280 + len(case_names) * 24 + 8),
            "action ids: " + " ".join(f"{i}:{name[:5]}" for i, name in enumerate(ACTION_NAMES)),
            fill=grey,
            font=font,
        )
        if t == t_star:
            draw.rectangle([game.width + 4, 4, img.width - 6, img.height - 6], outline=(255, 214, 102), width=3)
        frames.append(np.asarray(img))
    path = out_dir / "meta_ablation.gif"
    imageio.mimsave(path, frames, duration=1 / args.fps, loop=0)
    return path


def render_static(out_dir, trace, case_logits, rows, interesting):
    t, b = interesting
    base_probs = softmax_np(case_logits["baseline"])
    names = [r["case"] for r in rows if r["case"] != "baseline"]
    matrix = np.stack([softmax_np(case_logits[name])[t, b] - base_probs[t, b] for name in names])
    max_abs = max(float(np.max(np.abs(matrix))), 1e-6)
    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-max_abs, vmax=max_abs, aspect="auto")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_xticks(np.arange(len(ACTION_NAMES)))
    ax.set_xticklabels(ACTION_NAMES, rotation=45, ha="right")
    ax.set_title(f"Meta target probability change at interesting frame t={t}, batch row={b}")
    fig.colorbar(im, ax=ax, label="target_prob(case) - target_prob(baseline)")
    fig.tight_layout()
    path = out_dir / "selected_frame_ablation_heatmap.png"
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
    update_params = load_update_rule()
    rng = jax.random.PRNGKey(args.seed)
    batch, actor_after, source_idx, source_returns = sample_fresh_batch(
        agent, env, cfg, learner_state, rng
    )
    eta, extra, _ = compute_base(agent, learner_state, batch, actor_after)
    base_meta = apply_meta(agent, update_params, learner_state.meta_state, eta, extra)
    base_probs = softmax_np(np.asarray(jax.device_get(base_meta["pi"])))
    current_probs = softmax_np(np.asarray(jax.device_get(eta.agent_out["logits"][:-1])))
    push_l1 = np.sum(np.abs(base_probs - current_probs), axis=-1)
    interesting = tuple(int(x) for x in np.unravel_index(np.argmax(push_l1), push_l1.shape))
    cases = make_cases(eta, extra, interesting[0], interesting[1])
    case_logits = {}
    for name, (case_eta, case_extra) in cases.items():
        meta = base_meta if name == "baseline" else apply_meta(
            agent, update_params, learner_state.meta_state, case_eta, case_extra
        )
        case_logits[name] = np.asarray(jax.device_get(meta["pi"]))

    trace = {
        "batch": take(batch),
        "agent_out": take(eta.agent_out),
        "source_idx": take(source_idx),
        "source_returns": take(source_returns),
    }
    rows = summarize_cases(case_logits, case_logits["baseline"], interesting[0], interesting[1])
    gif_path = render_gif(out_dir, trace, case_logits, rows, interesting, args)
    heatmap_path = render_static(out_dir, trace, case_logits, rows, interesting)
    csv_path = out_dir / "ablation_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        out_dir / "ablation_trace.npz",
        current_logits=trace["agent_out"]["logits"][:-1],
        **{f"pi_hat_{name}": logits for name, logits in case_logits.items()},
        actions=trace["batch"].actions[:-1],
        rewards=trace["batch"].rewards[1:],
        source_idx=trace["source_idx"],
        source_returns=trace["source_returns"],
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_total_steps": int(payload.get("total_steps", -1)),
        "interesting_t": interesting[0],
        "interesting_batch_row": interesting[1],
        "interesting_source_env": int(trace["source_idx"][interesting[1]]),
        "interesting_push_l1": float(push_l1[interesting]),
        "gif_path": str(gif_path),
        "heatmap_path": str(heatmap_path),
        "csv_path": str(csv_path),
        "rows": rows,
        "caveat": "Ablations are on a fresh on-policy rollout. Checkpoints do not contain replay, so this is training-shaped but not replay-sampled.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
