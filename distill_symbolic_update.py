"""Distill DiscoRL pi targets into sparse symbolic update candidates.

This intentionally excludes the learned auxiliary y/z heads.  The goal is to
approximate the meta-network's policy target using only observable RL terms and
small nonlinear/product bases that could plausibly become a hand-written rule.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from inspect_meta_microscope import load_checkpoint, make_cfg, sample_fresh_batch, softmax_np, take
from research_meta_update_rule import (
    action_lookup,
    aligned,
    apply_meta,
    center_logits,
    compute_eta_extra,
    entropy_np,
    r2,
    sign_log_np,
)
from train_paper_pixels import CraftaxPixelsBatch, load_update_rule, make_agent, unflatten_params


EPS = 1e-8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=9100)
    p.add_argument("--num_rollouts", type=int, default=6)
    p.add_argument("--num_envs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--rollout_len", type=int, default=29)
    p.add_argument("--frame_stack", type=int, default=32)
    p.add_argument("--max_interactions", type=int, default=512)
    p.add_argument("--behaviour_policy_params", type=Path)
    p.add_argument("--update_policy_params", type=Path)
    p.add_argument("--target_policy_params", type=Path)
    return p.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_policy_params(path: Path):
    with np.load(path.expanduser().resolve()) as f:
        return jax.tree.map(jnp.asarray, unflatten_params(dict(f)))


def with_params(learner_state, params, target_params=None):
    meta_state = learner_state.meta_state
    if target_params is not None:
        meta_state = dict(meta_state)
        meta_state["target_params"] = target_params
    return dataclasses.replace(learner_state, params=params, meta_state=meta_state)


def safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, EPS, 1.0))


def soft_rank_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(np.argsort(-x, axis=-1), axis=-1).astype(np.float32)
    rank01 = order / max(x.shape[-1] - 1, 1)
    centered = x - x.mean(axis=-1, keepdims=True)
    return rank01, centered


def future_shift(value: np.ndarray, k: int) -> np.ndarray:
    arr = np.asarray(value)
    out = np.zeros_like(arr)
    if k < arr.shape[0]:
        out[:-k] = arr[k:]
    return out


def future_sum(value: np.ndarray, horizon: int, discount: float = 1.0) -> np.ndarray:
    arr = np.asarray(value)
    out = np.zeros_like(arr)
    scale = 1.0
    for k in range(1, horizon + 1):
        out += scale * future_shift(arr, k)
        scale *= discount
    return out


def expand(value: np.ndarray | float, t_len: int, b_size: int, n_actions: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == ():
        return np.full((t_len, b_size, n_actions), float(arr), dtype=np.float32)
    if arr.shape == (t_len,):
        arr = arr[:, None]
    if arr.shape == (t_len, 1):
        return np.broadcast_to(arr[:, :, None], (t_len, b_size, n_actions))
    if arr.shape == (t_len, b_size):
        return np.broadcast_to(arr[:, :, None], (t_len, b_size, n_actions))
    if arr.shape == (t_len, b_size, n_actions):
        return arr
    raise ValueError(f"unexpected feature shape {arr.shape}")


def add_feature(
    arrays: list[np.ndarray],
    names: list[str],
    groups: list[str],
    roles: list[str],
    name: str,
    group: str,
    role: str,
    value: np.ndarray | float,
    shape: tuple[int, int, int],
    record_idx: int,
) -> None:
    if record_idx == 0:
        names.append(name)
        groups.append(group)
        roles.append(role)
    arrays.append(expand(value, *shape).reshape(-1))


def build_base_dataset(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    xs, ys, cur_logits_all = [], [], []
    names: list[str] = []
    groups: list[str] = []
    roles: list[str] = []

    for rec_i, rec in enumerate(records):
        eta = rec["eta"]
        extra = rec["extra"]
        pi_hat = rec["pi_hat"]
        t_len, b_size, n_actions = pi_hat.shape
        shape = (t_len, b_size, n_actions)

        cur_logits = eta.agent_out["logits"][:t_len]
        beh_logits = eta.behaviour_agent_out["logits"][:t_len]
        lag_logits = extra["target_out"]["logits"][:t_len]
        cur_p, beh_p, lag_p = softmax_np(cur_logits), softmax_np(beh_logits), softmax_np(lag_logits)
        cur_logp, beh_logp, lag_logp = safe_log(cur_p), safe_log(beh_p), safe_log(lag_p)
        cur_entropy, beh_entropy, lag_entropy = entropy_np(cur_p), entropy_np(beh_p), entropy_np(lag_p)
        maxp = cur_p.max(axis=-1)
        argmax = np.argmax(cur_p, axis=-1)
        actions = eta.actions[:t_len].astype(np.int32)
        onehot_actions = np.eye(n_actions, dtype=np.float32)[actions]
        onehot_argmax = np.eye(n_actions, dtype=np.float32)[argmax]

        rewards = eta.rewards[:t_len]
        terminals = eta.is_terminal[:t_len].astype(np.float32)
        adv = aligned(extra["adv"], t_len)
        nadv = aligned(extra["normalized_adv"], t_len)
        q = aligned(extra["q"], t_len)
        qv = aligned(extra["qv_adv"], t_len)
        nqv = aligned(extra["normalized_qv_adv"], t_len)
        q_taken = action_lookup(q, actions)

        cur_rank, cur_centered_p = soft_rank_features(cur_p)
        lag_rank, lag_centered_p = soft_rank_features(lag_p)
        q_rank, q_centered = soft_rank_features(q)
        qv_centered = qv - qv.mean(axis=-1, keepdims=True)
        ratio = np.clip(cur_p / np.clip(beh_p, EPS, None), 0.0, 10.0)
        log_ratio = np.clip(cur_logp - beh_logp, -5.0, 5.0)
        lag_push_logit = center_logits(lag_logits) - center_logits(cur_logits)
        lag_push_prob = lag_p - cur_p
        entropy_grad = -cur_p * (cur_logp + cur_entropy[:, :, None])
        pg_adv = adv[:, :, None] * (onehot_actions - cur_p)
        norm_pg_adv = nadv[:, :, None] * (onehot_actions - cur_p)
        qv_pg = qv * (onehot_actions - cur_p)
        norm_qv_pg = nqv * (onehot_actions - cur_p)
        time_frac = np.linspace(0.0, 1.0, t_len, dtype=np.float32)[:, None]
        out = center_logits(pi_hat) - center_logits(cur_logits)

        arrays: list[np.ndarray] = []

        def add(name: str, group: str, role: str, value: np.ndarray | float) -> None:
            add_feature(arrays, names, groups, roles, name, group, role, value, shape, rec_i)

        add("bias", "bias", "bias", 1.0)
        add("time_frac", "time", "gate", time_frac)
        add("reward", "reward_terminal", "gate", rewards)
        add("reward_signlog", "reward_terminal", "gate", sign_log_np(rewards))
        add("terminal", "reward_terminal", "gate", terminals)
        add("adv", "advantage", "gate", adv)
        add("norm_adv", "advantage", "gate", nadv)
        add("entropy_current", "policy_state", "gate", cur_entropy)
        add("entropy_behaviour", "policy_state", "gate", beh_entropy)
        add("entropy_lag_target", "lag_policy_state", "gate", lag_entropy)
        add("p_current_max", "policy_state", "gate", maxp)
        add("taken_current_p", "action_context", "gate", action_lookup(cur_p, actions))
        add("taken_behaviour_p", "action_context", "gate", action_lookup(beh_p, actions))
        add("taken_lag_target_p", "action_context", "gate", action_lookup(lag_p, actions))
        add("taken_q", "value_state", "gate", q_taken)

        add("p_current_a", "policy_action", "action", cur_p)
        add("logp_current_a", "policy_action", "action", cur_logp)
        add("logit_current_centered_a", "policy_action", "action", center_logits(cur_logits))
        add("p_current_centered_a", "policy_action", "action", cur_centered_p)
        add("rank_current_a", "policy_action", "action", cur_rank)
        add("p_behaviour_a", "behaviour_action", "action", beh_p)
        add("log_ratio_current_behaviour_a", "behaviour_action", "action", log_ratio)
        add("ratio_current_behaviour_a", "behaviour_action", "action", ratio)
        add("p_lag_target_a", "lag_policy_action", "action", lag_p)
        add("logp_lag_target_a", "lag_policy_action", "action", lag_logp)
        add("lag_minus_current_p_a", "lag_policy_action", "action", lag_push_prob)
        add("lag_minus_current_logit_a", "lag_policy_action", "action", lag_push_logit)
        add("lag_target_centered_p_a", "lag_policy_action", "action", lag_centered_p)
        add("rank_lag_target_a", "lag_policy_action", "action", lag_rank)
        add("is_taken_action_a", "action", "action", onehot_actions)
        add("is_current_argmax_a", "action", "action", onehot_argmax)
        add("q_target_a", "q_value", "action", q)
        add("q_centered_a", "q_value", "action", q_centered)
        add("q_minus_taken_a", "q_value", "action", q - q_taken[:, :, None])
        add("rank_q_a", "q_value", "action", q_rank)
        add("qv_adv_a", "q_value", "action", qv)
        add("norm_qv_adv_a", "q_value", "action", nqv)
        add("qv_centered_a", "q_value", "action", qv_centered)
        add("entropy_grad_a", "entropy_rule", "rule", entropy_grad)
        add("pg_adv_a", "pg_rule", "rule", pg_adv)
        add("norm_pg_adv_a", "pg_rule", "rule", norm_pg_adv)
        add("qv_pg_a", "pg_rule", "rule", qv_pg)
        add("norm_qv_pg_a", "pg_rule", "rule", norm_qv_pg)

        for horizon in (1, 2, 4, 8, 16):
            add(f"future_reward_sum{horizon}", "future_reward_terminal", "gate", future_sum(rewards, horizon))
            add(
                f"future_reward_disc{horizon}",
                "future_reward_terminal",
                "gate",
                future_sum(rewards, horizon, discount=0.97),
            )
            add(
                f"future_terminal_any{horizon}",
                "future_reward_terminal",
                "gate",
                np.clip(future_sum(terminals, horizon), 0.0, 1.0),
            )
            add(f"future_adv_t+{horizon}", "future_advantage", "gate", future_shift(adv, horizon))
            add(f"future_norm_adv_t+{horizon}", "future_advantage", "gate", future_shift(nadv, horizon))
            add(
                f"future_entropy_current_t+{horizon}",
                "future_policy_state",
                "gate",
                future_shift(cur_entropy, horizon),
            )
            add(
                f"future_lag_minus_current_p_t+{horizon}",
                "future_lag_policy_action",
                "action",
                future_shift(lag_push_prob, horizon),
            )
            add(
                f"future_lag_minus_current_logit_t+{horizon}",
                "future_lag_policy_action",
                "action",
                future_shift(lag_push_logit, horizon),
            )
            add(f"future_qv_adv_t+{horizon}", "future_q_value", "action", future_shift(qv, horizon))
            add(
                f"future_norm_qv_adv_t+{horizon}",
                "future_q_value",
                "action",
                future_shift(nqv, horizon),
            )
            add(f"future_pg_adv_t+{horizon}", "future_pg_rule", "rule", future_shift(pg_adv, horizon))
            add(
                f"future_norm_pg_adv_t+{horizon}",
                "future_pg_rule",
                "rule",
                future_shift(norm_pg_adv, horizon),
            )
            add(
                f"future_entropy_grad_t+{horizon}",
                "future_entropy_rule",
                "rule",
                future_shift(entropy_grad, horizon),
            )

        xs.append(np.stack(arrays, axis=1))
        ys.append(out.reshape(-1))
        cur_logits_all.append(center_logits(cur_logits).reshape(-1))

    return (
        np.concatenate(xs).astype(np.float32),
        np.concatenate(ys).astype(np.float32),
        np.concatenate(cur_logits_all).astype(np.float32),
        names,
        groups,
        roles,
    )


def standardize(train_x: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    return (x - mean) / std, mean, std


def ridge_subset(x: np.ndarray, y: np.ndarray, cols: np.ndarray, alpha: float) -> np.ndarray:
    xt = x[:, cols].astype(np.float64, copy=False)
    yt = y.astype(np.float64, copy=False)
    reg = np.eye(len(cols), dtype=np.float64) * alpha
    if cols[0] == 0:
        reg[0, 0] = 0.0
    return np.linalg.solve(xt.T @ xt + reg, xt.T @ yt)


def predict_subset(x: np.ndarray, cols: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return x[:, cols] @ coef.astype(x.dtype, copy=False)


def make_interactions(
    x: np.ndarray,
    names: list[str],
    groups: list[str],
    roles: list[str],
    train_idx: np.ndarray,
    max_interactions: int,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    gate_idx = [i for i, r in enumerate(roles) if r == "gate" and names[i] != "bias"]
    action_idx = [i for i, r in enumerate(roles) if r in {"action", "rule"} and names[i] != "bias"]

    # Prefer gates/actions that carry actual signal before forming products.
    var = x[train_idx].var(axis=0)
    gate_idx = sorted(gate_idx, key=lambda i: var[i], reverse=True)[:32]
    action_idx = sorted(action_idx, key=lambda i: var[i], reverse=True)[:48]

    chunks = [x]
    new_names, new_groups, new_roles = list(names), list(groups), list(roles)
    made = 0
    for gi in gate_idx:
        for ai in action_idx:
            if made >= max_interactions:
                break
            chunks.append((x[:, gi] * x[:, ai])[:, None].astype(np.float32))
            new_names.append(f"{names[gi]} * {names[ai]}")
            new_groups.append(f"product:{groups[gi]}:{groups[ai]}")
            new_roles.append("product")
            made += 1
        if made >= max_interactions:
            break
    return np.concatenate(chunks, axis=1), new_names, new_groups, new_roles


def greedy_omp_path(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    names: list[str],
) -> tuple[list[dict], dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    checkpoints = {8, 16, 24, 32, 48, 64, 96, 128, 192}
    active = [0]
    candidates = np.array([i for i, n in enumerate(names) if n != "bias"], dtype=np.int32)
    saved: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rows: list[dict] = []

    coef = ridge_subset(x_train, y_train, np.array(active, dtype=np.int32), alpha=1e-6)
    pred_train = predict_subset(x_train, np.array(active, dtype=np.int32), coef)
    residual = y_train - pred_train
    for step in range(1, max(checkpoints) + 1):
        corr = np.abs(x_train[:, candidates].T @ residual)
        best_pos = int(np.argmax(corr))
        best_col = int(candidates[best_pos])
        active.append(best_col)
        candidates = np.delete(candidates, best_pos)
        cols = np.array(active, dtype=np.int32)
        coef = ridge_subset(x_train, y_train, cols, alpha=1e-3)
        pred_train = predict_subset(x_train, cols, coef)
        residual = y_train - pred_train
        if step in checkpoints:
            pred_test = predict_subset(x_test, cols, coef)
            rows.append(
                {
                    "model": f"omp_{step}_symbolic_basis",
                    "terms": int(len(cols)),
                    "train_r2": r2(y_train, pred_train),
                    "test_r2": r2(y_test, pred_test),
                }
            )
            saved[step] = (cols.copy(), coef.copy(), pred_test.copy())
    return rows, saved


def fit_sparse_symbolic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    names: list[str],
) -> tuple[list[dict], dict, np.ndarray, np.ndarray, np.ndarray]:
    all_cols = np.arange(x_train.shape[1])
    full_coef = ridge_subset(x_train, y_train, all_cols, alpha=1e-2)
    full_pred_test = predict_subset(x_test, all_cols, full_coef)
    rows = [
        {
            "model": "ridge_all_basis",
            "terms": int(len(all_cols)),
            "train_r2": r2(y_train, predict_subset(x_train, all_cols, full_coef)),
            "test_r2": r2(y_test, full_pred_test),
        }
    ]

    abs_order = np.argsort(np.abs(full_coef))[::-1]
    abs_order = [i for i in abs_order if names[i] != "bias"]
    saved: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    sparse_rows: list[dict] = []
    for k in (8, 16, 24, 32, 48, 64, 96, 128, 192):
        cols = np.array([0] + abs_order[:k], dtype=np.int32)
        coef = ridge_subset(x_train, y_train, cols, alpha=1e-3)
        pred_train = predict_subset(x_train, cols, coef)
        pred_test = predict_subset(x_test, cols, coef)
        row = {
            "model": f"top_{k}_sparse_basis",
            "terms": int(len(cols)),
            "train_r2": r2(y_train, pred_train),
            "test_r2": r2(y_test, pred_test),
        }
        rows.append(row)
        sparse_rows.append(row)
        saved[row["model"]] = (cols, coef, pred_test)

    omp_rows, omp_saved = greedy_omp_path(x_train, y_train, x_test, y_test, names)
    rows.extend(omp_rows)
    sparse_rows.extend(omp_rows)
    for r in omp_rows:
        step = int(r["model"].split("_")[1])
        saved[r["model"]] = omp_saved[step]

    best_sparse_score = max(r["test_r2"] for r in sparse_rows)
    readable = [r for r in sparse_rows if r["test_r2"] >= 0.98 * best_sparse_score]
    chosen = min(readable, key=lambda r: r["terms"]).copy()
    cols, coef, pred_test = saved[chosen["model"]]
    chosen["model"] = "chosen_" + chosen["model"]
    rows.append(chosen)
    return rows, chosen, cols, coef, pred_test


def plot_top_terms(out_dir: Path, term_rows: list[dict]) -> Path:
    top = term_rows[:30]
    fig, ax = plt.subplots(figsize=(11, 8), dpi=160)
    vals = [r["standardized_coef"] for r in top]
    labels = [r["feature"] for r in top]
    colors = ["#4c78a8" if v >= 0 else "#e45756" for v in vals]
    ax.barh(np.arange(len(top)), vals, color=colors)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_title("Sparse symbolic update terms, no y/z features")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = out_dir / "symbolic_top_terms.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_scores(out_dir: Path, rows: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    labels = [r["model"].replace("_", " ") for r in rows]
    vals = [r["test_r2"] for r in rows]
    ax.plot(np.arange(len(rows)), vals, marker="o", color="#4c78a8")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("test R2")
    ax.set_title("Symbolic basis model size sweep")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = out_dir / "symbolic_model_scores.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_pred(out_dir: Path, y: np.ndarray, pred: np.ndarray) -> Path:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), min(len(y), 40_000), replace=False)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    ax.scatter(y[idx], pred[idx], s=2, alpha=0.15)
    lim = max(float(np.max(np.abs(y[idx]))), float(np.max(np.abs(pred[idx]))), 1e-6)
    ax.plot([-lim, lim], [-lim, lim], color="black", lw=1)
    ax.set_xlabel("actual centered pi_hat - current")
    ax.set_ylabel("symbolic surrogate")
    ax.set_title("No-y/z symbolic surrogate")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = out_dir / "symbolic_pred_vs_actual.png"
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
    base_learner_state = learner_state
    behaviour_label = str(args.checkpoint)
    update_label = str(args.checkpoint)
    target_label = "checkpoint_meta_state_target_params"

    if args.update_policy_params:
        update_params_policy = load_policy_params(args.update_policy_params)
        target_params = (
            load_policy_params(args.target_policy_params)
            if args.target_policy_params
            else update_params_policy
        )
        learner_state = with_params(learner_state, update_params_policy, target_params)
        update_label = str(args.update_policy_params)
        target_label = str(args.target_policy_params or args.update_policy_params)

    behaviour_learner_state = learner_state
    if args.behaviour_policy_params:
        behaviour_learner_state = with_params(
            base_learner_state,
            load_policy_params(args.behaviour_policy_params),
            None,
        )
        behaviour_label = str(args.behaviour_policy_params)

    records = []
    for i in range(args.num_rollouts):
        rng = jax.random.PRNGKey(args.seed + i)
        batch, behaviour_actor_after, _, _ = sample_fresh_batch(
            agent, env, cfg, behaviour_learner_state, rng
        )
        if args.behaviour_policy_params:
            _, actor_after, _, _ = sample_fresh_batch(
                agent, env, cfg, learner_state, jax.random.fold_in(rng, 12345)
            )
        else:
            actor_after = behaviour_actor_after
        eta, extra, _ = compute_eta_extra(agent, learner_state, batch, actor_after)
        pi = apply_meta(agent, update_params, learner_state.meta_state, eta, extra)["pi"]
        pi_np = np.asarray(jax.device_get(pi))
        records.append({"eta": take(eta), "extra": take(extra), "pi_hat": pi_np})
        cur = np.asarray(jax.device_get(eta.agent_out["logits"][:-1]))
        push = np.mean(np.sum(np.abs(softmax_np(pi_np) - softmax_np(cur)), axis=-1))
        print(f"rollout={i + 1}/{args.num_rollouts} mean_target_push_l1={push:.3f}", flush=True)

    x_base, y, cur_logits_centered, names, groups, roles = build_base_dataset(records)
    rng = np.random.default_rng(0)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]
    x_base, mean, std = standardize(x_base[train_idx], x_base)
    x_base[:, 0] = 1.0
    x, names, groups, roles = make_interactions(
        x_base, names, groups, roles, train_idx, args.max_interactions
    )
    x, basis_mean, basis_std = standardize(x[train_idx], x)
    x[:, 0] = 1.0

    rows, chosen, cols, coef, pred_test = fit_sparse_symbolic(
        x[train_idx], y[train_idx], x[test_idx], y[test_idx], names
    )
    term_rows = [
        {
            "rank": rank,
            "feature": names[col],
            "group": groups[col],
            "role": roles[col],
            "standardized_coef": float(c),
            "abs_standardized_coef": float(abs(c)),
        }
        for rank, (col, c) in enumerate(
            sorted(zip(cols, coef), key=lambda z: abs(z[1]), reverse=True), start=1
        )
        if names[col] != "bias"
    ]

    score_csv = out_dir / "symbolic_model_scores.csv"
    terms_csv = out_dir / "symbolic_terms.csv"
    write_csv(score_csv, rows)
    write_csv(terms_csv, term_rows)
    terms_plot = plot_top_terms(out_dir, term_rows)
    score_plot = plot_scores(out_dir, rows)
    pred_plot = plot_pred(out_dir, y[test_idx], pred_test)

    pred_pi_centered = cur_logits_centered[test_idx] + pred_test
    actual_pi_centered = cur_logits_centered[test_idx] + y[test_idx]
    # Binary-action-level rows are flattened; this probability diagnostic is only
    # a centered-logit scalar proxy, so keep the true model-selection metric R2.
    equation = " + ".join(
        f"{r['standardized_coef']:+.3f}*z({r['feature']})" for r in term_rows[:16]
    )
    equation_md = out_dir / "symbolic_equation.md"
    equation_md.write_text(
        "# No-y/z symbolic surrogate\n\n"
        "Target: `center(pi_hat_logits) - center(current_policy_logits)`.\n\n"
        "Forbidden inputs: raw `agent_out/y`, `agent_out/z`, `target_out/y`, `target_out/z`, "
        "and any feature derived from them.\n\n"
        "Allowed inputs: current policy, behaviour policy, lag/target policy logits, rewards, "
        "terminals, value/Q/advantage terms, sampled action, entropy terms, and future-window "
        "shifts/sums of those same observable terms.\n\n"
        "Sparse nonlinear basis: hand-built PPO/Q/entropy/lag terms plus gate/action products.\n\n"
        "Approximate standardized equation:\n\n"
        "```text\n"
        f"delta_logit_a ~= {equation}\n"
        "```\n\n"
        f"Chosen model: {chosen['model']}; train R2={chosen['train_r2']:.3f}; "
        f"test R2={chosen['test_r2']:.3f}.\n"
    )
    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_total_steps": int(payload.get("total_steps", -1)),
        "behaviour_policy": behaviour_label,
        "update_policy": update_label,
        "target_policy": target_label,
        "off_policy": bool(args.behaviour_policy_params),
        "note": (
            "If update_policy_params is a policy-only npz, optimizer state, meta RNN state, "
            "and advantage/TD EMA state come from the checkpoint scaffold. If target_policy_params "
            "is omitted, target params are set equal to update params."
        ),
        "num_rollouts": args.num_rollouts,
        "samples_action_level": int(len(y)),
        "base_features": int(len([n for n in names if " * " not in n])),
        "total_basis_features": int(len(names)),
        "target": "center(pi_hat_logits) - center(current_policy_logits)",
        "forbidden_inputs": ["agent_out/y", "agent_out/z", "target_out/y", "target_out/z"],
        "chosen": chosen,
        "model_scores": rows,
        "top_terms": term_rows[:30],
        "score_csv": str(score_csv),
        "terms_csv": str(terms_csv),
        "terms_plot": str(terms_plot),
        "score_plot": str(score_plot),
        "pred_plot": str(pred_plot),
        "equation": str(equation_md),
        "pred_pi_centered_std": float(np.std(pred_pi_centered)),
        "actual_pi_centered_std": float(np.std(actual_pi_centered)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
