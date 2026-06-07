# Disocraft

Reproduce DiscoRL's **Disco103** learned update rule on **Craftax Classic
(pixels)**. The canonical run reaches roughly **~12 episode return** (Crafter-style
achievement reward) over tens of millions of env steps.

The single source of truth for training is `train_paper_pixels.py`; the
`eval_pixel_*` / `eval_paper_pixels.py` scripts evaluate its checkpoints.

> Exploratory work (symbolic-update distillation, the symbolic+pmap runner,
> meta-network analysis, etc.) lives on the `research/full-tree` branch, not here.

## Setup

Install `uv` first:

```bash
brew install uv
```

Python 3.11+ is required (`disco_rl` needs it). Initialize submodules
(`external/disco_rl`, `external/Craftax`), then let `uv` create and sync `.venv`:

```bash
git submodule update --init --recursive
uv sync
```

Apply the DiscoRL patch for JAX tracer compatibility:

```bash
uv run python scripts/patch_disco_rl.py
```

## Training

```bash
uv run python train_paper_pixels.py --checkpoint_dir runs/paper_pixels
```

Defaults are the paper-shaped Craftax Classic pixel setup: `Craftax-Classic-Pixels-v1`,
`num_envs=8`, `batch_size=24`, 32-frame stacking, a `256,384,384,256` conv torso
with a 768 FC and 1024 LSTM/head, and the Disco103 weights loaded from
`external/disco_rl`. `--target_steps` defaults to 1M for a quick run; use a larger
budget for the full result:

```bash
uv run python train_paper_pixels.py --checkpoint_dir runs/paper_pixels --target_steps 60000000
```

Checkpoints are written to the run dir (`checkpoint_latest.pkl` during training,
`checkpoint_final.pkl` at the end, plus `policy_params_*.npz` and `metrics.csv`).
Set `--snapshot_every_steps` to keep periodic snapshots for the sweep eval.

## Evaluation

Full-episode return:

```bash
uv run python eval_paper_pixels.py \
  --checkpoint runs/paper_pixels/checkpoint_final.pkl \
  --out_dir eval/paper_pixels
```

Per-achievement success rates:

```bash
uv run python eval_pixel_achievement_rates.py \
  --checkpoint runs/paper_pixels/checkpoint_final.pkl \
  --out_dir eval/achievements
```

Return vs. training steps across snapshots:

```bash
uv run python eval_pixel_sweep.py \
  --run_dir runs/paper_pixels \
  --out_dir eval/sweep
```

Render one episode to a GIF:

```bash
uv run python render_pixel_episode.py \
  --checkpoint runs/paper_pixels/checkpoint_final.pkl \
  --out_gif episode.gif
```

The eval scripts default to the same model architecture as training. If you
override `--channels / --fc_size / --model_lstm_size / --model_head_size` at train
time, pass the matching values at eval time.

## Formatting

```bash
uv run ruff format train_paper_pixels.py eval_paper_pixels.py \
  eval_pixel_achievement_rates.py eval_pixel_sweep.py render_pixel_episode.py scripts
```
