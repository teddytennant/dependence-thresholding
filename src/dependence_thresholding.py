"""Dependence thresholding for nonlinear latent factor structure.

Implements the parent-sharing graph, independent maximal clique extraction,
and structure-constrained map from Kim and Zhou, arXiv:2609.32926.
Estimation is Adam on mean squared error, not the paper's stochastic EM.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax

HIDDEN_UNITS = 16
ADAM_LR = 1e-2

__all__ = [
    "spearman_abs",
    "independent_maximal_cliques",
    "support_from_imcs",
    "structures_for_thresholds",
    "dependence_thresholding",
    "init_constrained_params",
    "predict_from_latents",
    "train_constrained",
    "default_thresholds",
]


class _Clique(frozenset):
    """Clique members plus the unique members found by closed-neighborhood search."""

    def __new__(cls, members, unique_members):
        obj = super().__new__(cls, members)
        obj.unique_members = frozenset(int(u) for u in unique_members)
        return obj

    def __repr__(self):
        members = ",".join(str(i) for i in sorted(self))
        uniques = ",".join(str(i) for i in sorted(self.unique_members))
        return f"Clique({{{members}}}, unique={{{uniques}}})"


def default_thresholds(m=50):
    """Fifty evenly spaced thresholds on [0, 1], as in Section 6.1."""
    return np.linspace(0.0, 1.0, int(m))


def _average_ranks(col):
    """1-based ranks with the average rank assigned to ties."""
    col = np.asarray(col, dtype=float)
    n = col.shape[0]
    order = np.argsort(col, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    sorted_col = col[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_col[j] == sorted_col[i]:
            j += 1
        # Ranks i+1, ..., j, averaged.
        avg = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg
        i = j
    return ranks


def spearman_abs(X):
    """Absolute Spearman rank correlation, shape (p, p). Diagonal is 1.

    Ranks use average ranks for ties. The correlation is Pearson correlation
    of those ranks. A constant column has undefined off-diagonal correlation;
    those entries are 0, and the diagonal is still 1.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must have shape (n, p)")
    n, p = X.shape
    if n == 0 or p == 0:
        out = np.eye(p, dtype=float)
        return out
    ranks = np.column_stack([_average_ranks(X[:, j]) for j in range(p)])
    centered = ranks - ranks.mean(axis=0, keepdims=True)
    ss = np.sum(centered * centered, axis=0)
    numer = centered.T @ centered
    denom = np.sqrt(np.outer(ss, ss))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.divide(numer, denom, out=np.zeros((p, p), dtype=float), where=denom > 0)
    np.fill_diagonal(corr, 1.0)
    return np.abs(corr)


def _adjacency_bool(adjacency):
    raw = np.asarray(adjacency)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    adj = np.logical_or(raw != 0, np.transpose(raw) != 0)
    adj = np.array(adj, dtype=bool, copy=True)
    np.fill_diagonal(adj, False)
    return adj


def _is_clique(adj, members):
    """True if every distinct pair in members is an edge. No self-loop required."""
    k = int(members.shape[0])
    if k <= 1:
        return True
    sub = adj[np.ix_(members, members)]
    off = ~np.eye(k, dtype=bool)
    return bool(np.all(sub[off]))


def independent_maximal_cliques(adjacency):
    """Independent maximal cliques via closed neighborhoods (Section 5.1).

    For each vertex i, C = N[i] = neighbors(i) union {i}. If C is a clique,
    it is an independent maximal clique and i is a unique member. Identical
    cliques are deduplicated. An isolated vertex is a clique of size 1.
    Self-loops are ignored.

    Returns a list of frozensets sorted by smallest unique-member index.
    Each item also carries a unique_members attribute.
    """
    adj = _adjacency_bool(adjacency)
    p = adj.shape[0]
    found = {}
    for i in range(p):
        neighbors = np.flatnonzero(adj[i])
        members = np.empty(neighbors.size + 1, dtype=int)
        members[0] = i
        if neighbors.size:
            members[1:] = neighbors
        members.sort()
        if not _is_clique(adj, members):
            continue
        key = frozenset(int(v) for v in members)
        found.setdefault(key, set()).add(i)
    records = []
    for members, uniques in found.items():
        records.append((min(uniques), tuple(sorted(members)), members, uniques))
    records.sort(key=lambda rec: (rec[0], rec[1]))
    return [_Clique(rec[2], rec[3]) for rec in records]


def support_from_imcs(cliques, p):
    """Binary support A of shape (p, d), A[i, j] = 1 iff i is in clique j.

    Columns are ordered by smallest unique-member index when that information
    is attached to the cliques (as from independent_maximal_cliques). Otherwise
    columns are ordered by the smallest member index, then by the member tuple.
    """
    p = int(p)
    cliques = list(cliques)

    def sort_key(clique):
        members = tuple(sorted(int(i) for i in clique))
        uniques = getattr(clique, "unique_members", None)
        if uniques:
            return (min(int(u) for u in uniques), members)
        if not members:
            return (-1, members)
        return (min(members), members)

    ordered = sorted(cliques, key=sort_key)
    d = len(ordered)
    support = np.zeros((p, d), dtype=np.int32)
    for j, clique in enumerate(ordered):
        for i in clique:
            i = int(i)
            if i < 0 or i >= p:
                raise ValueError(f"clique index {i} is outside 0..{p - 1}")
            support[i, j] = 1
    return support


def _threshold_adjacency(rho, tau):
    p = rho.shape[0]
    adj = np.zeros((p, p), dtype=np.int32)
    if p <= 1:
        return adj
    ii, jj = np.triu_indices(p, k=1)
    keep = rho[ii, jj] >= tau
    src = ii[keep]
    dst = jj[keep]
    adj[src, dst] = 1
    adj[dst, src] = 1
    return adj


def structures_for_thresholds(X, thresholds):
    """Parent-sharing supports for each threshold. No neural net training.

    Edge (i, j), i < j, is present iff |rho_ij| >= tau. Diagonal self-loops
    are not written. Each support is built from the independent maximal cliques.
    """
    rho = spearman_abs(X)
    p = rho.shape[0]
    thresholds = np.asarray(list(thresholds), dtype=float)
    supports = []
    for tau in thresholds:
        adj = _threshold_adjacency(rho, float(tau))
        cliques = independent_maximal_cliques(adj)
        supports.append(support_from_imcs(cliques, p))
    return supports


def _as_support(support):
    support = np.asarray(support)
    if support.ndim != 2:
        raise ValueError("support must have shape (p, d)")
    return support


def _parent_tuples(support):
    support = _as_support(support)
    parents = []
    for i in range(support.shape[0]):
        idx = tuple(int(j) for j in np.flatnonzero(support[i] != 0))
        parents.append(idx)
    return parents


def _apply_mlp(params, z):
    hidden = jnp.tanh(z @ params["w1"] + params["b1"])
    return hidden @ params["w2"] + params["b2"]


def _predict(mlps, Z, parents):
    n = Z.shape[0]
    cols = []
    for mlp, par in zip(mlps, parents):
        if len(par) == 0:
            cols.append(jnp.broadcast_to(mlp["bias"], (n,)))
        else:
            cols.append(_apply_mlp(mlp, Z[:, par]))
    return jnp.stack(cols, axis=1)


def _init_mlp(key, n_in):
    k1, k2 = jax.random.split(key)
    w1 = jax.random.normal(k1, (n_in, HIDDEN_UNITS)) * np.sqrt(2.0 / (n_in + HIDDEN_UNITS))
    b1 = jnp.zeros((HIDDEN_UNITS,))
    w2 = jax.random.normal(k2, (HIDDEN_UNITS,)) * np.sqrt(2.0 / (HIDDEN_UNITS + 1.0))
    b2 = jnp.zeros(())
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def _init_bias(_key):
    return {"bias": jnp.zeros(())}


def _init_params(key, n, support):
    support = _as_support(support)
    p, d = support.shape
    key, kz = jax.random.split(key)
    Z = jax.random.normal(kz, (int(n), int(d)))
    parents = _parent_tuples(support)
    mlps = []
    for par in parents:
        key, ki = jax.random.split(key)
        if len(par) == 0:
            mlps.append(_init_bias(ki))
        else:
            mlps.append(_init_mlp(ki, len(par)))
    return {"Z": Z, "mlps": tuple(mlps)}


def init_constrained_params(support, n, seed=0):
    """Initialize free latents Z (n, d) and one hard-masked map per observed variable."""
    return _init_params(jax.random.PRNGKey(int(seed)), int(n), support)


def predict_from_latents(params, Z, support):
    """Map latent rows to observed predictions. Non-parent columns are never read."""
    Z = jnp.asarray(Z)
    return _predict(params["mlps"], Z, _parent_tuples(support))


def _mse(pred, target):
    return jnp.mean(jnp.square(pred - target))


def _compile_train(parents, steps, lr):
    opt = optax.adam(lr)

    def loss_fn(params, X):
        pred = _predict(params["mlps"], params["Z"], parents)
        return _mse(pred, X)

    @jax.jit
    def run(params, X):
        opt_state = opt.init(params)

        def step(carry, _):
            params, opt_state = carry
            grads = jax.grad(loss_fn)(params, X)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), None

        (params, _), _ = jax.lax.scan(step, (params, opt_state), None, length=steps)
        return params

    return run, jax.jit(loss_fn)


def _compile_latent_fit(parents, steps, lr):
    opt = optax.adam(lr)

    def loss_fn(Z, mlps, X):
        pred = _predict(mlps, Z, parents)
        return _mse(pred, X)

    @jax.jit
    def run(Z, mlps, X):
        opt_state = opt.init(Z)

        def step(carry, _):
            Z, opt_state = carry
            grads = jax.grad(loss_fn)(Z, mlps, X)
            updates, opt_state = opt.update(grads, opt_state, Z)
            Z = optax.apply_updates(Z, updates)
            return (Z, opt_state), None

        (Z, _), _ = jax.lax.scan(step, (Z, opt_state), None, length=steps)
        return Z

    return run, jax.jit(loss_fn)


def _as_float_matrix(X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must have shape (n, p)")
    return X


def train_constrained(X, support, steps=200, seed=0, lr=ADAM_LR):
    """Adam on MSE for free latents and hard-masked per-variable maps.

    train_mse0 is the loss before any update. train_mse is the loss after
    `steps` updates. A variable with no parents is a learned bias.
    """
    X_np = _as_float_matrix(X)
    support = _as_support(support)
    if support.shape[0] != X_np.shape[1]:
        raise ValueError("support rows must match the number of observed variables")
    steps = int(steps)
    if steps < 0:
        raise ValueError("steps must be non-negative")
    X = jnp.asarray(X_np)
    params = _init_params(jax.random.PRNGKey(int(seed)), X_np.shape[0], support)
    parents = _parent_tuples(support)
    if steps == 0:
        pred = _predict(params["mlps"], params["Z"], parents)
        loss0 = float(_mse(pred, X))
        return {"params": params, "train_mse0": loss0, "train_mse": loss0, "support": support}
    run, loss_fn = _compile_train(parents, steps, float(lr))
    loss0 = float(loss_fn(params, X))
    params = run(params, X)
    loss1 = float(loss_fn(params, X))
    return {
        "params": params,
        "train_mse0": loss0,
        "train_mse": loss1,
        "support": np.array(support, dtype=np.int32, copy=True),
    }


def _validation_mse(mlps, X_val, support, steps, seed, lr):
    """Hold the maps fixed and fit validation latents, then return MSE.

    If no variable has a latent parent, the loss is the bias-only MSE.
    """
    X_val = jnp.asarray(np.asarray(X_val, dtype=np.float32))
    parents = _parent_tuples(support)
    d = _as_support(support).shape[1]
    n = int(X_val.shape[0])
    if d == 0 or all(len(par) == 0 for par in parents) or steps <= 0:
        pred = _predict(mlps, jnp.zeros((n, d)), parents)
        return float(_mse(pred, X_val))
    Z = jax.random.normal(jax.random.PRNGKey(int(seed)), (n, d))
    run, loss_fn = _compile_latent_fit(parents, int(steps), float(lr))
    Z = run(Z, mlps, X_val)
    return float(loss_fn(Z, mlps, X_val))


def _split_train_val(X):
    """Last 20% of rows, rounded, held out as validation. No shuffle."""
    n = int(X.shape[0])
    if n <= 1:
        return X, X[:0]
    n_val = int(round(0.2 * n))
    n_val = min(max(n_val, 1), n - 1)
    return X[:-n_val], X[-n_val:]


def dependence_thresholding(X, thresholds=None, steps=200, seed=0):
    """Algorithm 1 with Spearman dependence and validation MSE selection.

    For each threshold, build the parent-sharing graph, extract independent
    maximal cliques, and fit a structure-constrained map. The map with the
    smallest validation MSE is returned. Validation rows are the last 20%.
    Latents on those rows are refit with the trained maps held fixed.

    Returns a dict with support, n_factors, threshold, and val_loss.
    Identical supports are fit once; the reported threshold is the smallest
    grid value that produced the selected support.
    """
    X_np = _as_float_matrix(X)
    if thresholds is None:
        thresholds = default_thresholds(50)
    thresholds = np.asarray(list(thresholds), dtype=float)
    if thresholds.size == 0:
        raise ValueError("thresholds must be non-empty")
    supports = structures_for_thresholds(X_np, thresholds)
    X_train, X_val = _split_train_val(X_np)
    if X_val.shape[0] == 0:
        X_train, X_val = X_np, X_np

    best = None
    cache = {}
    for index, (tau, support) in enumerate(zip(thresholds, supports)):
        key = (support.shape, support.tobytes())
        if key not in cache:
            fit = train_constrained(X_train, support, steps=steps, seed=seed, lr=ADAM_LR)
            val_loss = _validation_mse(
                fit["params"]["mlps"],
                X_val,
                support,
                steps=steps,
                seed=seed,
                lr=ADAM_LR,
            )
            cache[key] = {
                "support": np.array(support, dtype=np.int32, copy=True),
                "val_loss": float(val_loss),
                "params": fit["params"],
                "train_mse0": fit["train_mse0"],
                "train_mse": fit["train_mse"],
            }
        record = cache[key]
        if best is None or record["val_loss"] < best["val_loss"]:
            best = {
                "support": record["support"],
                "n_factors": int(record["support"].shape[1]),
                "threshold": float(tau),
                "val_loss": float(record["val_loss"]),
                "params": record["params"],
                "train_mse0": float(record["train_mse0"]),
                "train_mse": float(record["train_mse"]),
                "threshold_index": int(index),
            }
    return {
        "support": best["support"],
        "n_factors": best["n_factors"],
        "threshold": best["threshold"],
        "val_loss": best["val_loss"],
        "params": best["params"],
        "train_mse0": best["train_mse0"],
        "train_mse": best["train_mse"],
    }
