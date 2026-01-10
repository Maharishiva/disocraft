# Disocraft

Train DiscoRL's Disco103 update rule on Craftax (symbolic, default).

## Local setup

This repo expects Python 3.11+ because `disco_rl` requires it.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

Install JAX first (CPU or GPU, choose one):

```bash
# CPU only
pip install -U "jax[cpu]"
```

Clone external dependencies if you do not already have them:

```bash
git clone https://github.com/google-deepmind/disco_rl.git external/disco_rl
git clone https://github.com/MichaelTMatthews/Craftax.git external/Craftax
```

Then install the local clones + extra deps:

```bash
pip install -r requirements.txt
```

Apply the DiscoRL patch for JAX tracer compatibility:

```bash
python scripts/patch_disco_rl.py
```

## Training

```bash
python train.py --num_iterations 1000 --num_envs 1
```

Common overrides:

```bash
python train.py \
  --env_name Craftax-Symbolic-v1 \
  --num_envs 1 \
  --rollout_len 32 \
  --batch_size 64 \
  --learning_rate 3e-4
```

The Disco103 weights are loaded from `external/disco_rl` by default. Use
`--weights_path` to point to a different copy if needed.
