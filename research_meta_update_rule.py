"""Research-style perturbation and surrogate analysis for DiscoRL pi_hat."""

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
from train_paper_pixels import CraftaxPixelsBatch, load_update_rule, make_agent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=8100)
    p.add_argument("--num_rollouts", type=int, default=6)
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--rollout_len", type=int, default=29)
    p.add_argument("--frame_stack", type=int, default=32)
    p.add_argument("--scale", type=int, default=7)
    p.add_argument("--fps", type=int, default=4)
    return p.parse_args()


def center_logits(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x, axis=-1, keepdims=True)


def sign_log_np(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


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


def replace_out(out: dict, **kwargs) -> dict:
    data = dict(out)
    data.update(kwargs)
    return data


def blend_tree(current, target, alpha: float):
    return jax.tree.map(lambda c, t: c + alpha * (t - c), current, target)


def batch_shuffle_tree(x, shift: int = 1):
    return jax.tree.map(lambda y: jnp.roll(y, shift=shift, axis=1), x)


def temp_logits(logits, temp: float):
    mean = jnp.mean(logits, axis=-1, keepdims=True)
    return mean + (logits - mean) / temp


def compute_eta_extra(agent, learner_state, batch, actor_state, target_out=None, eta=None):
    if eta is None:
        agent_out, _ = agent.unroll_net(learner_state.params, actor_state, batch)
        eta = types.UpdateRuleInputs(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards[1:],
            is_terminal=batch.discounts[1:] == 0,
            behaviour_agent_out=batch.agent_outs,
            agent_out=agent_out,
            value_out=None,
        )
    if target_out is None:
        target_out, _ = agent._network.unroll(
            learner_state.meta_state["target_params"],
            actor_state,
            eta.observations,
            eta.should_reset_mask_fwd,
        )
    hyper = agent.settings.hyper_params.to_dict()
    value_outs, _, _ = value_utils.get_value_outs(
        value_net_out=None,
        target_value_net_out=None,
        q_net_out=eta.agent_out["q"],
        target_q_net_out=target_out["q"],
        rollout=eta,
        pi_logits=eta.agent_out["logits"],
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


def make_perturbations(agent, learner_state, batch, actor_state, eta, extra, target_out):
    current = eta.agent_out
    cases = {}

    def add_case(name, case_eta, case_target=None, case_extra_updates=None):
        case_eta, case_extra, _ = compute_eta_extra(
            agent,
            learner_state,
            batch,
            actor_state,
            target_out=case_target if case_target is not None else target_out,
            eta=case_eta,
        )
        if case_extra_updates:
            case_extra = case_extra | case_extra_updates(case_extra)
        cases[name] = (case_eta, case_extra)

    add_case("baseline", eta, target_out)
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.25):
        add_case(f"target_lag_alpha_{alpha:.2f}", eta, blend_tree(current, target_out, alpha))
    add_case(
        "target_policy_current_only",
        eta,
        replace_out(target_out, logits=current["logits"]),
    )
    add_case("target_y_current_only", eta, replace_out(target_out, y=current["y"]))
    add_case("target_z_current_only", eta, replace_out(target_out, z=current["z"]))
    add_case("target_q_current_only", eta, replace_out(target_out, q=current["q"]))
    add_case(
        "target_policy_soft_temp2",
        eta,
        replace_out(target_out, logits=temp_logits(target_out["logits"], 2.0)),
    )
    add_case(
        "target_policy_sharp_temp0.5",
        eta,
        replace_out(target_out, logits=temp_logits(target_out["logits"], 0.5)),
    )
    add_case("target_shuffle_batch", eta, batch_shuffle_tree(target_out, shift=1))
    add_case(
        "target_policy_shuffle_batch",
        eta,
        replace_out(target_out, logits=jnp.roll(target_out["logits"], shift=1, axis=1)),
    )
    add_case(
        "target_z_shuffle_batch",
        eta,
        replace_out(target_out, z=jnp.roll(target_out["z"], shift=1, axis=1)),
    )
    add_case(
        "behaviour_policy_current",
        replace_eta(eta, behaviour_agent_out=eta.agent_out),
        target_out,
    )
    add_case(
        "uniform_current_policy",
        replace_eta(
            eta,
            agent_out=replace_out(eta.agent_out, logits=jnp.zeros_like(eta.agent_out["logits"])),
        ),
        target_out,
    )
    for scale in (0.0, 0.5, 2.0):
        add_case(
            f"scale_adv_{scale:.1f}",
            eta,
            target_out,
            lambda e, scale=scale: {
                "adv": e["adv"] * scale,
                "normalized_adv": e["normalized_adv"] * scale,
            },
        )
        add_case(
            f"scale_qv_{scale:.1f}",
            eta,
            target_out,
            lambda e, scale=scale: {
                "qv_adv": e["qv_adv"] * scale,
                "normalized_qv_adv": e["normalized_qv_adv"] * scale,
            },
        )
    return cases


def aligned(x: np.ndarray, t: int) -> np.ndarray:
    return x[:t] if x.shape[0] >= t else x


def action_lookup(x: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return np.take_along_axis(x, actions[..., None], axis=2)[..., 0]


def build_surrogate_dataset(records: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    xs = []
    ys = []
    names: list[str] = []
    groups: list[str] = []
    for rec in records:
        eta = rec["eta"]
        extra = rec["extra"]
        pi_hat = rec["pi_hat"]
        t_len, b_size, n_actions = pi_hat.shape
        cur_logits = eta.agent_out["logits"][:t_len]
        beh_logits = eta.behaviour_agent_out["logits"][:t_len]
        target = extra["target_out"]
        tgt_logits = target["logits"][:t_len]
        cur_p = softmax_np(cur_logits)
        beh_p = softmax_np(beh_logits)
        tgt_p = softmax_np(tgt_logits)
        target_y = target["y"][:t_len]
        current_y = eta.agent_out["y"][:t_len]
        target_z = target["z"][:t_len]
        current_z = eta.agent_out["z"][:t_len]
        y_kl = categorical_kl_np(target_y, current_y)
        z_kl = categorical_kl_np(target_z, current_z)
        target_q = aligned(extra["q"], t_len)
        qv = aligned(extra["qv_adv"], t_len)
        nqv = aligned(extra["normalized_qv_adv"], t_len)
        adv = aligned(extra["adv"], t_len)
        nadv = aligned(extra["normalized_adv"], t_len)
        rewards = eta.rewards[:t_len]
        terminals = eta.is_terminal[:t_len].astype(np.float32)
        actions = eta.actions[:t_len]
        out = center_logits(pi_hat) - center_logits(cur_logits)
        cur_entropy = entropy_np(cur_p)
        tgt_entropy = entropy_np(tgt_p)
        beh_entropy = entropy_np(beh_p)
        maxp = np.max(cur_p, axis=-1)
        argmax = np.argmax(cur_p, axis=-1)
        onehot_actions = np.eye(n_actions, dtype=np.float32)[actions.astype(np.int32)]
        onehot_argmax = np.eye(n_actions, dtype=np.float32)[argmax.astype(np.int32)]
        p_taken = action_lookup(cur_p, actions)
        tgt_taken = action_lookup(tgt_p, actions)
        q_taken = action_lookup(target_q, actions)
        time_frac = np.linspace(0.0, 1.0, t_len, dtype=np.float32)[:, None]
        z_current_entropy = entropy_np(softmax_np(current_z))
        z_target_entropy = entropy_np(softmax_np(target_z))

        feat_arrays: list[np.ndarray] = []

        def expand(value):
            arr = np.asarray(value, dtype=np.float64)
            if arr.shape == ():
                return np.full((t_len, b_size, n_actions), float(arr), dtype=np.float64)
            if arr.shape == (t_len,):
                arr = arr[:, None]
            if arr.shape == (t_len, 1):
                arr = np.broadcast_to(arr[:, :, None], (t_len, b_size, n_actions))
            elif arr.shape == (t_len, b_size):
                arr = np.broadcast_to(arr[:, :, None], (t_len, b_size, n_actions))
            elif arr.shape != (t_len, b_size, n_actions):
                raise ValueError(f"unexpected feature shape {arr.shape}")
            return arr

        def add(name, group, value):
            if not xs:
                names.append(name)
                groups.append(group)
            feat_arrays.append(expand(value).reshape(-1))

        add("bias", "bias", 1.0)
        add("p_current", "policy", cur_p)
        add("logit_current_centered", "policy", center_logits(cur_logits))
        add("is_current_argmax", "policy", onehot_argmax)
        add("p_current_max", "policy", maxp)
        add("entropy_current", "policy", cur_entropy)
        add("p_target", "target_policy", tgt_p)
        add("target_minus_current_p", "target_policy", tgt_p - cur_p)
        add("logit_target_centered", "target_policy", center_logits(tgt_logits))
        add("entropy_target", "target_policy", tgt_entropy)
        add("p_behaviour", "behaviour", beh_p)
        add("behaviour_minus_current_p", "behaviour", beh_p - cur_p)
        add("entropy_behaviour", "behaviour", beh_entropy)
        add("is_taken_action", "action", onehot_actions)
        add("taken_current_p", "action", p_taken)
        add("taken_target_p", "action", tgt_taken)
        add("reward", "reward_terminal", rewards)
        add("reward_signlog", "reward_terminal", sign_log_np(rewards))
        add("terminal", "reward_terminal", terminals)
        add("time_frac", "time", time_frac)
        add("adv", "advantage", adv)
        add("norm_adv", "advantage", nadv)
        add("q_target_a", "q_value", target_q)
        add("q_taken", "q_value", q_taken)
        add("q_minus_taken", "q_value", target_q - q_taken[:, :, None])
        add("qv_adv_a", "q_value", qv)
        add("norm_qv_adv_a", "q_value", nqv)
        add("y_target_current_kl", "yz_aux", y_kl)
        add("z_target_current_kl_a", "yz_aux", z_kl)
        add("z_current_entropy_a", "yz_aux", z_current_entropy)
        add("z_target_entropy_a", "yz_aux", z_target_entropy)

        def future_shift(value, k: int):
            arr = np.asarray(value)
            out_f = np.zeros_like(arr)
            if k < t_len:
                out_f[:-k] = arr[k:]
            return out_f

        def future_sum(value, horizon: int):
            arr = np.asarray(value)
            out_f = np.zeros_like(arr)
            for k in range(1, horizon + 1):
                out_f += future_shift(arr, k)
            return out_f

        for horizon in (1, 2, 4, 8, 16):
            add(f"future_reward_t+{horizon}", "future_reward_terminal", future_shift(rewards, horizon))
            add(
                f"future_reward_sum_next{horizon}",
                "future_reward_terminal",
                future_sum(rewards, horizon),
            )
            add(
                f"future_terminal_any_next{horizon}",
                "future_reward_terminal",
                np.clip(future_sum(terminals, horizon), 0.0, 1.0),
            )
            add(f"future_adv_t+{horizon}", "future_advantage", future_shift(adv, horizon))
            add(
                f"future_norm_adv_t+{horizon}",
                "future_advantage",
                future_shift(nadv, horizon),
            )
            add(
                f"future_entropy_current_t+{horizon}",
                "future_policy",
                future_shift(cur_entropy, horizon),
            )
            add(
                f"future_target_minus_current_p_t+{horizon}",
                "future_policy",
                future_shift(tgt_p - cur_p, horizon),
            )
            add(
                f"future_logit_current_centered_t+{horizon}",
                "future_policy",
                future_shift(center_logits(cur_logits), horizon),
            )
            add(
                f"future_qv_adv_a_t+{horizon}",
                "future_q_value",
                future_shift(qv, horizon),
            )
            add(
                f"future_norm_qv_adv_a_t+{horizon}",
                "future_q_value",
                future_shift(nqv, horizon),
            )
            add(
                f"future_q_minus_taken_t+{horizon}",
                "future_q_value",
                future_shift(target_q - q_taken[:, :, None], horizon),
            )
            add(
                f"future_z_target_current_kl_a_t+{horizon}",
                "future_yz_aux",
                future_shift(z_kl, horizon),
            )
            add(
                f"future_z_current_entropy_a_t+{horizon}",
                "future_yz_aux",
                future_shift(z_current_entropy, horizon),
            )
            add(
                f"future_z_target_entropy_a_t+{horizon}",
                "future_yz_aux",
                future_shift(z_target_entropy, horizon),
            )
        xs.append(np.stack(feat_arrays, axis=1))
        ys.append(out.reshape(-1))
    return np.concatenate(xs), np.concatenate(ys), names, groups


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1e-2):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    Xs = (X - mean) / std
    Xs[:, 0] = 1.0
    n_features = Xs.shape[1]
    reg = np.eye(n_features) * alpha
    reg[0, 0] = 0.0
    coef = np.linalg.solve(Xs.T @ Xs + reg, Xs.T @ y)
    pred = Xs @ coef
    return coef, pred, mean, std


def r2(y, pred):
    denom = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - pred) ** 2) / max(denom, 1e-12))


def group_drop_r2(X, y, groups, full_r2):
    out = []
    group_names = sorted(set(groups) - {"bias"})
    for group in group_names:
        keep = np.asarray([g != group for g in groups], bool)
        coef, pred, _, _ = ridge_fit(X[:, keep], y)
        score = r2(y, pred)
        out.append({"group": group, "r2_without_group": score, "r2_drop": full_r2 - score})
    return sorted(out, key=lambda r: r["r2_drop"], reverse=True)


def perturbation_rows(case_logits_by_rollout, baseline_by_rollout):
    rows = []
    for name in sorted(case_logits_by_rollout):
        base = np.concatenate(baseline_by_rollout, axis=1)
        case = np.concatenate(case_logits_by_rollout[name], axis=1)
        bp = softmax_np(base)
        cp = softmax_np(case)
        delta = cp - bp
        rows.append(
            {
                "case": name,
                "mean_l1_prob_delta": float(np.mean(np.sum(np.abs(delta), axis=-1))),
                "p95_l1_prob_delta": float(np.percentile(np.sum(np.abs(delta), axis=-1), 95)),
                "mean_kl_from_baseline": float(np.mean(categorical_kl_np(base, case))),
                "mean_target_entropy": float(np.mean(entropy_np(cp))),
            }
        )
    return sorted(rows, key=lambda r: r["mean_l1_prob_delta"], reverse=True)


def plot_bars(out_dir, rows, key, title, filename):
    top = rows[:20]
    labels = [r.get("case", r.get("group")) for r in top]
    vals = [r[key] for r in top]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
    ax.barh(np.arange(len(top)), vals, color="#4c78a8")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_coeffs(out_dir, coef, names, groups):
    idx = np.argsort(np.abs(coef))[::-1]
    idx = [i for i in idx if names[i] != "bias"][:25]
    fig, ax = plt.subplots(figsize=(10, 7), dpi=160)
    labels = [f"{names[i]} [{groups[i]}]" for i in idx]
    vals = [coef[i] for i in idx]
    colors = ["#4c78a8" if v >= 0 else "#e45756" for v in vals]
    ax.barh(np.arange(len(idx)), vals, color=colors)
    ax.set_yticks(np.arange(len(idx)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Ridge surrogate standardized coefficients for centered-logit push")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = out_dir / "surrogate_top_coefficients.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_scatter(out_dir, y, pred):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), min(len(y), 40_000), replace=False)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    ax.scatter(y[idx], pred[idx], s=2, alpha=0.15)
    lim = max(float(np.max(np.abs(y[idx]))), float(np.max(np.abs(pred[idx]))), 1e-6)
    ax.plot([-lim, lim], [-lim, lim], color="black", lw=1)
    ax.set_xlabel("actual pi_hat centered-logit push")
    ax.set_ylabel("surrogate prediction")
    ax.set_title("Linear surrogate fit")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = out_dir / "surrogate_pred_vs_actual.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def render_selected_gif(out_dir, selected, args):
    trace, case_logits, rows, selected_tb = selected
    t_star, b_star = selected_tb
    batch = trace["batch"]
    obs = batch.observations["observation"][:-1, b_star, :, :, -3:]
    base = softmax_np(case_logits["baseline"])[:, b_star]
    current = softmax_np(trace["eta"].agent_out["logits"][:-1])[:, b_star]
    case_names = [r["case"] for r in rows if r["case"] != "baseline"][:12]
    frames = []
    font = ImageFont.load_default()
    for t in range(base.shape[0]):
        game = Image.fromarray(np.asarray(np.clip(obs[t] * 255, 0, 255), np.uint8))
        game = game.resize((game.width * args.scale, game.height * args.scale), Image.Resampling.NEAREST)
        w, h = game.width + 910, max(game.height, 650)
        img = Image.new("RGB", (w, h), (16, 18, 22))
        img.paste(game, (0, 0))
        draw = ImageDraw.Draw(img)
        x0 = game.width + 14
        draw.text((x0, 10), "Realistic target/value perturbations", fill=(240, 242, 246), font=font)
        draw.text(
            (x0, 28),
            f"step {t:02d}; selected frame {t_star:02d}; action {ACTION_NAMES[int(batch.actions[:-1][t, b_star])]}",
            fill=(240, 242, 246),
            font=font,
        )
        draw.text(
            (x0, 48),
            f"current top {ACTION_NAMES[int(np.argmax(current[t]))]}  baseline target top {ACTION_NAMES[int(np.argmax(base[t]))]}",
            fill=(178, 184, 195),
            font=font,
        )
        matrix = np.stack([softmax_np(case_logits[name])[:, b_star][t] - base[t] for name in case_names])
        max_abs = max(float(np.max(np.abs(matrix))), 1e-6)
        for i, name in enumerate(case_names):
            draw.text((x0, 84 + i * 28), name[:28], fill=(230, 233, 238), font=font)
            for a, val in enumerate(matrix[i]):
                mag = min(abs(float(val)) / max_abs, 1.0)
                color = (
                    (int(55 + 40 * mag), int(80 + 125 * mag), int(115 + 115 * mag))
                    if val >= 0
                    else (int(110 + 135 * mag), int(55 + 40 * mag), int(65 + 35 * mag))
                )
                xx = x0 + 210 + a * 38
                yy = 80 + i * 28
                draw.rectangle([xx, yy, xx + 35, yy + 25], fill=color)
                if abs(float(val)) > 0.015:
                    draw.text((xx + 2, yy + 7), f"{val:+.2f}", fill=(245, 245, 245), font=font)
        draw.text(
            (x0 + 210, 84 + len(case_names) * 28 + 8),
            " ".join(f"{i}:{name[:5]}" for i, name in enumerate(ACTION_NAMES)),
            fill=(178, 184, 195),
            font=font,
        )
        if t == t_star:
            draw.rectangle([game.width + 4, 4, w - 6, h - 6], outline=(255, 215, 110), width=3)
        frames.append(np.asarray(img))
    path = out_dir / "realistic_perturbation_selected_rollout.gif"
    imageio.mimsave(path, frames, duration=1 / args.fps, loop=0)
    return path


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload, learner_state = load_checkpoint(args.checkpoint.expanduser().resolve())
    cfg = make_cfg(args, payload)
    env = CraftaxPixelsBatch(cfg.num_envs, cfg.frame_stack)
    agent = make_agent(env, cfg)
    update_params = load_update_rule()

    records = []
    case_logits_by_rollout: dict[str, list[np.ndarray]] = {}
    selected = None
    selected_push = -1.0
    for i in range(args.num_rollouts):
        rng = jax.random.PRNGKey(args.seed + i)
        batch, actor_after, _, _ = sample_fresh_batch(agent, env, cfg, learner_state, rng)
        eta, extra, _ = compute_eta_extra(agent, learner_state, batch, actor_after)
        target_out = extra["target_out"]
        cases = make_perturbations(agent, learner_state, batch, actor_after, eta, extra, target_out)
        rollout_case_logits = {}
        for name, (case_eta, case_extra) in cases.items():
            pi = apply_meta(agent, update_params, learner_state.meta_state, case_eta, case_extra)["pi"]
            pi = np.asarray(jax.device_get(pi))
            rollout_case_logits[name] = pi
            case_logits_by_rollout.setdefault(name, []).append(pi)
        base_probs = softmax_np(rollout_case_logits["baseline"])
        current_probs = softmax_np(np.asarray(jax.device_get(eta.agent_out["logits"][:-1])))
        push = np.sum(np.abs(base_probs - current_probs), axis=-1)
        local_max = float(np.max(push))
        eta_np = take(eta)
        extra_np = take(extra)
        records.append({"eta": eta_np, "extra": extra_np, "pi_hat": rollout_case_logits["baseline"]})
        if local_max > selected_push:
            selected_push = local_max
            selected = (
                {"batch": take(batch), "eta": eta_np, "extra": extra_np},
                rollout_case_logits,
                perturbation_rows({k: [v] for k, v in rollout_case_logits.items()}, [rollout_case_logits["baseline"]]),
                tuple(int(x) for x in np.unravel_index(np.argmax(push), push.shape)),
            )
        print(f"rollout={i + 1}/{args.num_rollouts} max_push={local_max:.3f}", flush=True)

    perturb_rows = perturbation_rows(
        case_logits_by_rollout, case_logits_by_rollout["baseline"]
    )
    perturb_csv = out_dir / "perturbation_summary.csv"
    write_csv(perturb_csv, perturb_rows)
    perturb_plot = plot_bars(
        out_dir,
        perturb_rows,
        "mean_l1_prob_delta",
        "Realistic perturbation effect on pi_hat",
        "perturbation_effects.png",
    )

    X, y, names, groups = build_surrogate_dataset(records)
    rng = np.random.default_rng(0)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]
    coef, pred_train, mean, std = ridge_fit(X[train_idx], y[train_idx], alpha=1e-2)
    Xtest = (X[test_idx] - mean) / std
    Xtest[:, 0] = 1.0
    pred_test = Xtest @ coef
    train_r2 = r2(y[train_idx], pred_train)
    test_r2 = r2(y[test_idx], pred_test)
    group_rows = group_drop_r2(X[train_idx], y[train_idx], groups, train_r2)
    group_csv = out_dir / "surrogate_group_drop.csv"
    write_csv(group_csv, group_rows)
    coeff_rows = [
        {
            "feature": names[i],
            "group": groups[i],
            "standardized_coef": float(coef[i]),
            "abs_standardized_coef": float(abs(coef[i])),
        }
        for i in np.argsort(np.abs(coef))[::-1]
    ]
    coeff_csv = out_dir / "surrogate_coefficients.csv"
    write_csv(coeff_csv, coeff_rows)
    coeff_plot = plot_coeffs(out_dir, coef, names, groups)
    group_plot = plot_bars(
        out_dir,
        group_rows,
        "r2_drop",
        "Surrogate R2 drop when feature group is removed",
        "surrogate_group_importance.png",
    )
    scatter_plot = plot_scatter(out_dir, y[test_idx], pred_test)
    gif_path = render_selected_gif(out_dir, selected, args)

    equation_md = out_dir / "equation_notes.md"
    top = [r for r in coeff_rows if r["feature"] != "bias"][:12]
    equation_md.write_text(
        "# Surrogate update notes\n\n"
        "The fitted target is centered(pi_hat_logits) - centered(current_logits).\n"
        "A PPO-like direct advantage equation is not enough here: the best sparse story is a mixture of policy inertia/lag, target-policy reference, and Q/z action features.\n\n"
        "Approximate standardized equation, keeping only large terms:\n\n"
        "```text\n"
        "delta_logit_a ~= "
        + " + ".join(
            f"{r['standardized_coef']:+.3f}*z({r['feature']})" for r in top[:8]
        )
        + "\n```\n\n"
        f"Train R2: {train_r2:.3f}; test R2: {test_r2:.3f}.\n"
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_total_steps": int(payload.get("total_steps", -1)),
        "num_rollouts": args.num_rollouts,
        "samples_action_level": int(len(y)),
        "surrogate_train_r2": train_r2,
        "surrogate_test_r2": test_r2,
        "selected_push_l1": selected_push,
        "perturbation_csv": str(perturb_csv),
        "coeff_csv": str(coeff_csv),
        "group_csv": str(group_csv),
        "perturbation_plot": str(perturb_plot),
        "coeff_plot": str(coeff_plot),
        "group_plot": str(group_plot),
        "scatter_plot": str(scatter_plot),
        "gif_path": str(gif_path),
        "equation_notes": str(equation_md),
        "top_perturbations": perturb_rows[:8],
        "top_coefficients": top,
        "group_importance": group_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
