"""Re-run toy meta-training with identical seeds, logging J every iter."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import optax
from toy_discorl import init_eta, meta_loss

key = jax.random.PRNGKey(0)
key, k_eta = jax.random.split(key)
eta = init_eta(k_eta)
opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-3))
opt_state = opt.init(eta)


@jax.jit
def meta_step(eta, opt_state, key):
    loss, g = jax.value_and_grad(meta_loss)(eta, key)
    updates, opt_state = opt.update(g, opt_state)
    return optax.apply_updates(eta, updates), opt_state, -loss


js = []
for i in range(601):
    key, sub = jax.random.split(key)
    eta, opt_state, j = meta_step(eta, opt_state, sub)
    js.append(float(j))
    if i % 100 == 0:
        print(i, round(js[-1], 3), flush=True)

json.dump(js, open(Path(__file__).parent / 'curve.json', 'w'))
print('saved', len(js), 'points; final', round(js[-1], 3))
