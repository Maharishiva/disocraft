"""Evaluate pixel policy checkpoints and plot return over training steps."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import jax

from eval_paper_pixels import make_eval, summarize
from train_paper_pixels import Config, unflatten_params


STEP_RE = re.compile(r"step_(\d+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, action="append", default=[])
    p.add_argument("--latest_checkpoint", type=Path, default=None)
    p.add_argument("--num_batches", type=int, default=1)
    p.add_argument("--num_envs", type=int, default=512)
    p.add_argument("--seed", type=int, default=2000)
    p.add_argument("--action_mode", choices=("greedy", "sample"), default="sample")
    p.add_argument("--channels", type=str, default="256,384,384,256")
    p.add_argument("--fc_size", type=int, default=768)
    p.add_argument("--model_lstm_size", type=int, default=1024)
    p.add_argument("--model_head_size", type=int, default=1024)
    return p.parse_args()


def load_params(path: Path):
    if path.suffix == ".npz":
        with np.load(path) as f:
            params = unflatten_params({k: f[k] for k in f.files})
        match = STEP_RE.search(path.name)
        steps = int(match.group(1)) if match else -1
        return params, {"total_steps": steps}
    with path.open("rb") as f:
        payload = pickle.load(f)
    return payload["learner_state"].params, payload


def checkpoints(args: argparse.Namespace) -> list[Path]:
    paths = list(args.checkpoint) or sorted(args.run_dir.glob("policy_params_step_*.npz"))
    if args.latest_checkpoint is not None:
        paths.append(args.latest_checkpoint)
    return paths


def row_from_arrays(path: Path, payload: dict, returns, lengths, achievements, scores, done):
    return {
        "checkpoint": path.name,
        "path": str(path),
        "steps": int(payload.get("total_steps", -1)),
        "episodes": int(returns.size),
        "done_fraction": float(np.mean(done)),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "median_return": float(np.median(returns)),
        "p05_return": float(np.percentile(returns, 5)),
        "p95_return": float(np.percentile(returns, 95)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "mean_length": float(np.mean(lengths)),
        "mean_achievements": float(np.mean(achievements)),
        "mean_score": float(np.mean(scores)),
    }


def step_ema(x: np.ndarray, y: np.ndarray, half_life_steps: float) -> np.ndarray:
    out = np.empty_like(y, dtype=np.float64)
    avg = float(y[0])
    last_x = float(x[0])
    for i, (xi, yi) in enumerate(zip(x, y)):
        alpha = 1.0 - 0.5 ** (max(float(xi - last_x), 0.0) / half_life_steps)
        avg = avg * (1.0 - alpha) + float(yi) * alpha
        out[i] = avg
        last_x = float(xi)
    return out


def plot(run_dir: Path, out_dir: Path, rows: list[dict]) -> Path:
    eval_steps = np.asarray([r["steps"] for r in rows], dtype=np.float64)
    eval_returns = np.asarray([r["mean_return"] for r in rows], dtype=np.float64)
    order = np.argsort(eval_steps)
    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=160)
    metrics = run_dir / "metrics.csv"
    if metrics.exists():
        data = np.genfromtxt(metrics, delimiter=",", names=True)
        if data.size:
            steps = np.atleast_1d(data["steps"]).astype(np.float64)
            avg = np.atleast_1d(data["avg_return"]).astype(np.float64)
            episodes = np.atleast_1d(data["episodes"]).astype(np.float64)
            valid = episodes > 0
            ax.scatter(
                steps[valid] / 1e6,
                avg[valid],
                s=12,
                c="#9aa4b2",
                alpha=0.35,
                label="logged rollout avg_return",
            )
            if valid.sum() > 2:
                ax.plot(
                    steps[valid] / 1e6,
                    step_ema(steps[valid], avg[valid], half_life_steps=1_000_000),
                    color="#687386",
                    lw=2.0,
                    alpha=0.9,
                    label="logged EMA, 1M-step half-life",
                )
    ax.plot(
        eval_steps[order] / 1e6,
        eval_returns[order],
        marker="o",
        lw=2.8,
        ms=6,
        color="#ff7f0e",
        label="proper full-episode eval",
    )
    for x, y in zip(eval_steps[order], eval_returns[order]):
        ax.text(x / 1e6, y + 0.15, f"{x / 1e6:.0f}M", fontsize=8, ha="center")
    ax.set_title("DiscoRL Craftax Classic Pixels: return vs training steps")
    ax.set_xlabel("training steps, millions")
    ax.set_ylabel("episode return")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    out = out_dir / "eval_return_over_steps.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_batch, max_steps = make_eval(args)
    rows = []
    csv_path = out_dir / "eval_sweep.csv"
    paths = checkpoints(args)
    start = time.time()
    for ckpt_idx, path in enumerate(paths):
        params, payload = load_params(path)
        all_returns, all_lengths, all_ach, all_scores, all_done = [], [], [], [], []
        for batch in range(args.num_batches):
            rng = jax.random.PRNGKey(args.seed + ckpt_idx * 10_000 + batch)
            returns, lengths, achievements, scores, done = jax.device_get(
                eval_batch(params, rng)
            )
            all_returns.append(returns)
            all_lengths.append(lengths)
            all_ach.append(achievements)
            all_scores.append(scores)
            all_done.append(done)
        row = row_from_arrays(
            path,
            payload,
            np.concatenate(all_returns),
            np.concatenate(all_lengths),
            np.concatenate(all_ach),
            np.concatenate(all_scores),
            np.concatenate(all_done),
        )
        rows.append(row)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plot_path = plot(args.run_dir, out_dir, rows)
        print(
            f"{path.name}: steps={row['steps']} mean_return={row['mean_return']:.3f} "
            f"max={row['max_return']:.3f} mean_len={row['mean_length']:.1f} "
            f"done={row['done_fraction']:.3f} elapsed={time.time() - start:.1f}s",
            flush=True,
        )
    summary = {
        "action_mode": args.action_mode,
        "num_envs": args.num_envs,
        "num_batches": args.num_batches,
        "episodes_per_checkpoint": args.num_envs * args.num_batches,
        "max_steps": max_steps,
        "rows": rows,
        "csv_path": str(csv_path),
        "plot_path": str(plot_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
