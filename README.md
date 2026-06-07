# Disocraft

Train DiscoRL's Disco103 update rule on Craftax (symbolic, default).

## Local setup

Install `uv` first:

```bash
brew install uv
```

This repo expects Python 3.11+ because `disco_rl` requires it. Initialize
submodules, then let `uv` create and sync `.venv`:

```bash
git submodule update --init --recursive
uv sync
```

Apply the DiscoRL patch for JAX tracer compatibility:

```bash
uv run python scripts/patch_disco_rl.py
```

Format repo-owned Python files with Ruff:

```bash
uv run ruff format train.py trainer.py train_full.py train_pmap.py train_pmap_full.py scripts
```

## Training

```bash
uv run python train.py --checkpoint_dir runs/train_1b_fifo_256x4
```

The default entrypoint is intentionally narrow: 2 local devices, symbolic
Craftax, `global_envs=4`, `global_batch=256`, FIFO replay, 1B target env steps,
and latest/final checkpoints.

```bash
uv run python train.py --target_steps 1000000 --checkpoint_dir runs/smoke
```

The Disco103 weights are loaded from `external/disco_rl` by default. Use
`train_full.py` or `train_pmap_full.py` for the older configurable runners.

Notes:
- `trainer.py` is the fixed pmap training core.
- `train.py` is only the small CLI/status wrapper.
