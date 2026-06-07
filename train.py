"""Tiny launcher for the fixed 2xA100 Disco103 Craftax run."""

from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
from trainer import Config, train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", type=Path, default=None)
    p.add_argument("--target_steps", type=int, default=1_000_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=1000)
    p.add_argument("--chunk_iterations", type=int, default=10_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.checkpoint_dir or Path(
        "runs", f"train_1b_fifo_256x4_{datetime.now():%Y%m%d_%H%M%S}"
    )
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    status = run_dir / "status.txt"
    cfg = Config(
        checkpoint_dir=run_dir,
        target_steps=args.target_steps,
        seed=args.seed,
        log_every=args.log_every,
        chunk_iterations=args.chunk_iterations,
    )
    status.write_text(
        "\n".join(
            [
                "status=running",
                f"started_at={datetime.now().isoformat()}",
                f"run_dir={run_dir}",
                f"target_steps={cfg.target_steps}",
                "global_envs=4",
                "global_batch=256",
                "replay_mode=fifo",
                "fifo_replay_ratio=64",
            ]
        )
        + "\n"
    )
    try:
        train(cfg)
    except BaseException as exc:
        with status.open("a") as f:
            f.write(f"train_status=1\nerror={type(exc).__name__}: {exc}\n")
            f.write(f"finished_at={datetime.now().isoformat()}\n")
        raise
    with status.open("a") as f:
        f.write("train_status=0\n")
        f.write(f"finished_at={datetime.now().isoformat()}\n")


if __name__ == "__main__":
    main()
