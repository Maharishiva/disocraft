# toy_discorl.py -- a ~200-line re-implementation of DiscoRL meta-training.
#
# Environment: 6-state chain. Agent starts at s=0, reward 1 on first arrival
# at the goal s=5 (absorbing), horizon 10. A uniform policy reaches the goal
# ~10% of the time, so there is real credit assignment to discover.
#
# Agent (theta): tabular policy logits pi[S,A] and prediction logits y[S,K]
# with NO predefined semantics -- the meta-network decides what y means.
#
# Meta-network (eta): a small RNN that scans the trajectory BACKWARDS over
# (reward, action, pi(.|s), softmax(y_t), softmax(y_{t+1})) and emits targets
# (pi_hat, y_hat) per step. The agent minimizes KL(targets || predictions).
#
# Meta-objective: exact (differentiable) expected return of the final policy
# after K_INNER updates, computed by dynamic programming. The real system
# replaces this with an advantage actor-critic estimate of grad_theta J.
#
# Part 2 demonstrates MixFlow-MG: the same meta-gradient computed with a
# custom_vjp inner step that uses forward-over-reverse Hessian-vector
# products instead of reverse-over-reverse backprop.

import functools
import jax
import jax.numpy as jnp
import optax

S, A, K_PRED, H_META = 6, 2, 4, 16   # states, actions, |y|, meta hidden size
T, B, K_INNER = 10, 16, 8            # horizon, rollouts/update, inner updates
ALPHA = 5.0                          # inner SGD learning rate
GOAL = S - 1


# ----------------------------------------------------------------------------
# Environment: chain MDP with absorbing goal.
# ----------------------------------------------------------------------------
def env_step(s, a):
    s2 = jnp.where(s == GOAL, GOAL, jnp.clip(s + 2 * a - 1, 0, GOAL))
    r = jnp.where((s != GOAL) & (s2 == GOAL), 1.0, 0.0)
    return s2, r


def rollout(theta, key):
    """Sample one trajectory. Data only -- no gradient flows through sampling."""
    pi_logits = jax.lax.stop_gradient(theta['pi'])

    def step(s, key_t):
        a = jax.random.categorical(key_t, pi_logits[s])
        s2, r = env_step(s, a)
        return s2, (s, a, r)

    keys = jax.random.split(key, T)
    s_last, (ss, aa, rr) = jax.lax.scan(step, jnp.int32(0), keys)
    states = jnp.concatenate([ss, s_last[None]])          # [T+1]
    return dict(s=states, a=aa, r=rr)


def exact_return(pi_logits):
    """Differentiable J(pi): exact probability of reaching the goal within T."""
    pi = jax.nn.softmax(pi_logits)
    idx = jnp.arange(S)

    def step(carry, _):
        p, j = carry
        mass = p.at[GOAL].set(0.0)                        # non-absorbed mass
        p2 = jnp.zeros(S).at[jnp.clip(idx + 1, 0, GOAL)].add(pi[:, 1] * mass)
        p2 = p2.at[jnp.clip(idx - 1, 0, GOAL)].add(pi[:, 0] * mass)
        r = p2[GOAL]                                      # newly absorbed mass
        p2 = p2.at[GOAL].add(p[GOAL])
        return (p2, j + r), None

    p0 = jnp.zeros(S).at[0].set(1.0)
    (_, j), _ = jax.lax.scan(step, (p0, 0.0), None, length=T)
    return j


# ----------------------------------------------------------------------------
# Meta-network: backwards RNN over the trajectory -> targets (pi_hat, y_hat).
# ----------------------------------------------------------------------------
def init_eta(key):
    f = 1 + A + A + 2 * K_PRED                            # per-step features
    k1, k2, k3, k4 = jax.random.split(key, 4)
    g = lambda k, shape, s: s * jax.random.normal(k, shape)
    return dict(Wx=g(k1, (f, H_META), 0.3), Wh=g(k2, (H_META, H_META), 0.3),
                b=jnp.zeros(H_META),
                Wpi=g(k3, (H_META, A), 0.1), Wy=g(k4, (H_META, K_PRED), 0.1))


def meta_targets(eta, theta, traj):
    s, a, r = traj['s'], traj['a'], traj['r']
    pi_p = jax.nn.softmax(theta['pi'][s[:-1]])            # [T, A]
    y_p = jax.nn.softmax(theta['y'][s])                   # [T+1, K]
    # Inputs are treated as data w.r.t. theta (cf. stop_grad input transforms).
    pi_p, y_p = jax.lax.stop_gradient((pi_p, y_p))
    feats = jnp.concatenate(                              # [T, F]
        [r[:, None], jax.nn.one_hot(a, A), pi_p, y_p[:-1], y_p[1:]], axis=-1)

    def cell(h, x):                                       # h carries the FUTURE
        h = jnp.tanh(x @ eta['Wx'] + h @ eta['Wh'] + eta['b'])
        return h, h

    _, hs = jax.lax.scan(cell, jnp.zeros(H_META), feats, reverse=True)
    return hs @ eta['Wpi'], hs @ eta['Wy']                # target logits


# ----------------------------------------------------------------------------
# Agent loss: KL(targets || predictions), gradients flow into eta via targets.
# ----------------------------------------------------------------------------
def kl(p_logits, q_logits):
    p = jax.nn.softmax(p_logits)
    return jnp.sum(p * (jax.nn.log_softmax(p_logits)
                        - jax.nn.log_softmax(q_logits)), axis=-1)


def agent_loss(theta, eta, traj):
    pi_hat, y_hat = meta_targets(eta, theta, traj)
    s = traj['s'][:-1]
    return jnp.mean(kl(pi_hat, theta['pi'][s]) + kl(y_hat, theta['y'][s]))


def batched_loss(theta, eta, trajs):
    return jnp.mean(jax.vmap(agent_loss, in_axes=(None, None, 0))(
        theta, eta, trajs))


def init_theta():
    return dict(pi=jnp.zeros((S, A)), y=jnp.zeros((S, K_PRED)))


# ----------------------------------------------------------------------------
# Meta-objective: unroll K_INNER agent updates, return -J(theta_K).
# jax.grad of this is the meta-gradient (reverse-over-reverse by default).
# ----------------------------------------------------------------------------
def unroll_inner(eta, key, step_fn):
    theta = init_theta()
    for _ in range(K_INNER):
        key, sub = jax.random.split(key)
        trajs = jax.vmap(rollout, in_axes=(None, 0))(
            theta, jax.random.split(sub, B))
        theta = step_fn(theta, eta, trajs)
    return theta


def sgd_step(theta, eta, trajs):
    g = jax.grad(batched_loss)(theta, eta, trajs)
    return jax.tree.map(lambda p, gi: p - ALPHA * gi, theta, g)


def meta_loss(eta, key):
    theta = unroll_inner(eta, key, sgd_step)
    return -exact_return(theta['pi'])


# ----------------------------------------------------------------------------
# Part 2: MixFlow-MG style inner step. Identical numbers, different autodiff.
# The backward pass computes (H v, M v) with ONE forward-over-reverse JVP of
# the inner gradient function instead of backprop through the backward graph.
# ----------------------------------------------------------------------------
def make_mixflow_step(trajs):
    def inner_grad(theta, eta):                           # g = grad_theta L
        return jax.grad(batched_loss, argnums=(0, 1))(theta, eta, trajs)

    def primal(theta, eta):
        g_theta = jax.grad(batched_loss)(theta, eta, trajs)
        return jax.tree.map(lambda p, gi: p - ALPHA * gi, theta, g_theta)

    @jax.custom_vjp
    def step(theta, eta):
        return primal(theta, eta)

    def fwd(theta, eta):
        return primal(theta, eta), (theta, eta)

    def bwd(res, v):
        theta, eta = res
        # Schwarz symmetry: v.(dg/dtheta) = H v, v.(dg/deta) = (d grad_eta L/dtheta) v
        _, (Hv, Mv) = jax.jvp(lambda th: inner_grad(th, eta), (theta,), (v,))
        v_theta = jax.tree.map(lambda a, b: a - ALPHA * b, v, Hv)
        v_eta = jax.tree.map(lambda b: -ALPHA * b, Mv)
        return v_theta, v_eta

    step.defvjp(fwd, bwd)
    return step


def meta_loss_mixflow(eta, key):
    theta = init_theta()
    for _ in range(K_INNER):
        key, sub = jax.random.split(key)
        trajs = jax.vmap(rollout, in_axes=(None, 0))(
            theta, jax.random.split(sub, B))
        theta = make_mixflow_step(trajs)(theta, eta)
    return -exact_return(theta['pi'])


# ----------------------------------------------------------------------------
# Baseline: REINFORCE inner rule with the best lr from a sweep.
# ----------------------------------------------------------------------------
def reinforce_step(theta, lr, trajs):
    def loss(th):
        def one(traj):
            logp = jax.nn.log_softmax(th['pi'][traj['s'][:-1]])
            lp_a = jnp.take_along_axis(logp, traj['a'][:, None], 1)[:, 0]
            ret = jnp.cumsum(traj['r'][::-1])[::-1]
            return -jnp.mean(lp_a * ret)
        return jnp.mean(jax.vmap(one)(trajs))
    g = jax.grad(loss)(theta)
    return jax.tree.map(lambda p, gi: p - lr * gi, theta, g)


def eval_reinforce(lr, key):
    theta = unroll_inner(None, key, lambda th, _, tr: reinforce_step(th, lr, tr))
    return exact_return(theta['pi'])


# ----------------------------------------------------------------------------
# Meta-training loop.
# ----------------------------------------------------------------------------
def main():
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

    print(f"J(theta_0) uniform policy: {exact_return(init_theta()['pi']):.3f}")
    for i in range(601):
        key, sub = jax.random.split(key)
        eta, opt_state, j_final = meta_step(eta, opt_state, sub)
        if i % 50 == 0:
            print(f"meta-iter {i:4d}   J(theta_{K_INNER}) = {j_final:.3f}")

    # REINFORCE baseline (sweep lr, same budget: K_INNER updates of B rollouts)
    key, sub = jax.random.split(key)
    best = max((float(jnp.mean(jax.vmap(lambda k: eval_reinforce(lr, k))(
        jax.random.split(sub, 16)))), lr)
        for lr in [0.1, 0.3, 1.0, 3.0, 10.0])
    print(f"REINFORCE baseline, best lr={best[1]}: J = {best[0]:.3f}")

    # What did y learn? Probe the trained rule on a fresh agent.
    key, sub = jax.random.split(key)
    theta = unroll_inner(eta, sub, sgd_step)
    print("state, softmax(y[s]) after training a fresh agent with the rule:")
    for s in range(S):
        print(f"  s={s}  y={jnp.round(jax.nn.softmax(theta['y'][s]), 2)}")

    # MixFlow-MG check: same meta-gradient, different differentiation mode.
    key, sub = jax.random.split(key)
    g_naive = jax.grad(meta_loss)(eta, sub)
    g_mix = jax.grad(meta_loss_mixflow)(eta, sub)
    err = jax.tree.map(
        lambda a, b: float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(a)) + 1e-12)),
        g_naive, g_mix)
    print("max relative error naive vs mixflow meta-gradient, per eta leaf:")
    for name, e in err.items():
        print(f"  {name}: {e:.2e}")


if __name__ == '__main__':
    main()
