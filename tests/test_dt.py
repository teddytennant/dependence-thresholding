"""Tests for dependence thresholding. CPU only, fixed seeds."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np

from dependence_thresholding import (
    HIDDEN_UNITS,
    dependence_thresholding,
    independent_maximal_cliques,
    init_constrained_params,
    predict_from_latents,
    spearman_abs,
    structures_for_thresholds,
    support_from_imcs,
    train_constrained,
)


def _edges(pairs, p):
    adj = np.zeros((p, p), dtype=np.int32)
    for i, j in pairs:
        adj[i, j] = 1
        adj[j, i] = 1
    return adj


def _same_support(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    cols_a = [tuple(np.asarray(a[:, j]).tolist()) for j in range(a.shape[1])]
    cols_b = [tuple(np.asarray(b[:, j]).tolist()) for j in range(b.shape[1])]
    return sorted(cols_a) == sorted(cols_b)


def _cluster(n, n_children, seed, noise=0.05):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 2))
    p = 2 * n_children
    X = np.zeros((n, p))
    for j in range(n_children):
        X[:, j] = np.tanh(3.0 * Z[:, 0]) + noise * rng.normal(size=n)
        X[:, n_children + j] = Z[:, 1] ** 3 + noise * rng.normal(size=n)
    return X


def test_two_disjoint_triangles():
    adj = _edges([(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)], 6)
    imcs = independent_maximal_cliques(adj)
    assert set(imcs) == {frozenset({0, 1, 2}), frozenset({3, 4, 5})}
    assert len(imcs) == 2
    support = support_from_imcs(imcs, 6)
    expected = np.array(
        [
            [1, 0],
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 1],
            [0, 1],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(support, expected)


def test_unique_member_rule():
    # Path 0-1-2. Closed neighborhood of the middle vertex is not a clique.
    path = _edges([(0, 1), (1, 2)], 3)
    path_imcs = independent_maximal_cliques(path)
    assert set(path_imcs) == {frozenset({0, 1}), frozenset({1, 2})}
    assert frozenset({0, 1, 2}) not in set(path_imcs)
    path_support = support_from_imcs(path_imcs, 3)
    np.testing.assert_array_equal(
        path_support,
        np.array([[1, 0], [1, 1], [0, 1]], dtype=np.int32),
    )

    # Hand graph. Edges 1-2, 0-3, 0-4. No edge 3-4, so N[0] = {0,3,4}
    # is not a clique and 0 is not a unique member.
    # IMCs: {1,2} unique members {1,2}; {0,3} unique member {3}; {0,4} unique member {4}.
    hand = _edges([(1, 2), (0, 3), (0, 4)], 5)
    hand_imcs = independent_maximal_cliques(hand)
    assert set(hand_imcs) == {
        frozenset({1, 2}),
        frozenset({0, 3}),
        frozenset({0, 4}),
    }
    # Column order follows smallest unique-member index: 1, then 3, then 4.
    # Sorting by minimum member would put the cliques that contain 0 first.
    hand_support = support_from_imcs(hand_imcs, 5)
    np.testing.assert_array_equal(
        hand_support,
        np.array(
            [
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.int32,
        ),
    )

    # Isolated vertices are size-1 IMCs, including when the diagonal is set.
    isolated = np.zeros((3, 3), dtype=np.int32)
    assert set(independent_maximal_cliques(isolated)) == {
        frozenset({0}),
        frozenset({1}),
        frozenset({2}),
    }
    np.testing.assert_array_equal(support_from_imcs(independent_maximal_cliques(isolated), 3), np.eye(3))
    loops = np.eye(3, dtype=np.int32)
    assert set(independent_maximal_cliques(loops)) == {
        frozenset({0}),
        frozenset({1}),
        frozenset({2}),
    }
    single = np.zeros((1, 1), dtype=np.int32)
    assert independent_maximal_cliques(single) == [frozenset({0})]


def test_spearman_monotone_and_noise():
    x = np.linspace(-1.5, 1.5, 50)
    rng = np.random.default_rng(0)
    noise1 = rng.normal(size=50)
    noise2 = rng.normal(size=50)
    X = np.column_stack([x, x**3, noise1, noise2])
    R = spearman_abs(X)
    assert R.shape == (4, 4)
    assert np.allclose(R, R.T)
    assert np.allclose(np.diag(R), 1.0)
    assert np.isclose(R[0, 1], 1.0)
    assert R[0, 2] < 1.0 and not np.isclose(R[0, 2], 1.0)
    assert R[1, 3] < 1.0 and not np.isclose(R[1, 3], 1.0)
    assert R[2, 3] < 1.0 and not np.isclose(R[2, 3], 1.0)

    # Average ranks: col0 ranks [1, 2.5, 2.5, 4], col1 ranks [1, 2, 3.5, 3.5].
    tied = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [2.0, 3.0],
            [4.0, 3.0],
        ]
    )
    Rt = spearman_abs(tied)
    assert np.isclose(Rt[0, 1], 3.75 / 4.5)
    assert np.isclose(Rt[0, 0], 1.0)


def test_cluster_structure_recovery():
    X = _cluster(n=800, n_children=4, seed=0, noise=0.05)
    p = X.shape[1]
    R = spearman_abs(X)
    within = []
    cross = []
    for i in range(p):
        for j in range(i + 1, p):
            same = (i < 4 and j < 4) or (i >= 4 and j >= 4)
            if same:
                within.append(R[i, j])
            else:
                cross.append(R[i, j])
    within_min = float(min(within))
    cross_max = float(max(cross))
    assert within_min > cross_max, f"no gap: within_min={within_min}, cross_max={cross_max}"
    tau = 0.5 * (within_min + cross_max)
    assert cross_max < tau < within_min

    supports = structures_for_thresholds(X, [tau])
    assert len(supports) == 1
    hat = supports[0]
    assert hat.shape == (8, 2)
    truth = np.zeros((8, 2), dtype=np.int32)
    truth[:4, 0] = 1
    truth[4:, 1] = 1
    assert _same_support(hat, truth)


def test_constrained_mask_and_selection():
    X = _cluster(n=64, n_children=3, seed=0, noise=0.05)
    result = dependence_thresholding(X, steps=30, seed=0)
    support = np.asarray(result["support"])
    p = X.shape[1]
    assert support.shape == (p, result["n_factors"])
    assert result["n_factors"] == support.shape[1]
    assert np.isfinite(result["val_loss"])
    assert "threshold" in result

    params = result["params"]
    assert len(params["mlps"]) == p
    saw_nonparent = False
    for i in range(p):
        parents = np.flatnonzero(support[i] != 0)
        mlp = params["mlps"][i]
        if parents.size == 0:
            assert "bias" in mlp
            assert "w1" not in mlp
        else:
            assert tuple(mlp["w1"].shape) == (parents.size, HIDDEN_UNITS)
            assert tuple(mlp["w2"].shape) == (HIDDEN_UNITS,)
            assert mlp["w1"].shape[0] == parents.size
        nonparent = np.flatnonzero(support[i] == 0)
        if result["n_factors"] > 0 and nonparent.size > 0:
            saw_nonparent = True

    # A fully dense single clique has no non-parent. Selection on this
    # two-cluster sample should not return that support.
    assert saw_nonparent, f"selected support has no non-parent:\n{support}"

    Z = params["Z"]
    pred = predict_from_latents(params, Z, support)
    for i in range(p):
        nonparent = np.flatnonzero(support[i] == 0)
        if nonparent.size == 0:
            continue
        noise = jax.random.normal(jax.random.PRNGKey(i + 7), (Z.shape[0], nonparent.size))
        Z_pert = Z.at[:, nonparent].set(noise)
        pred_pert = predict_from_latents(params, Z_pert, support)
        np.testing.assert_array_equal(np.asarray(pred[:, i]), np.asarray(pred_pert[:, i]))

        def _obs_i(z_row, index=i):
            return predict_from_latents(params, z_row[None, :], support)[0, index]

        jac = jax.jacrev(_obs_i)(Z[0])
        assert np.allclose(np.asarray(jac)[nonparent], 0.0)

    # Fresh init, same hard mask, so the check is not an accident of training.
    fresh = init_constrained_params(support, n=8, seed=1)
    Zf = jax.random.normal(jax.random.PRNGKey(2), (8, support.shape[1]))
    base = predict_from_latents(fresh, Zf, support)
    for i in range(p):
        nonparent = np.flatnonzero(support[i] == 0)
        if nonparent.size == 0:
            continue
        Z2 = jnp.array(Zf)
        Z2 = Z2.at[:, nonparent].add(3.5)
        other = predict_from_latents(fresh, Z2, support)
        np.testing.assert_array_equal(np.asarray(base[:, i]), np.asarray(other[:, i]))


def test_adam_reduces_train_mse():
    X = _cluster(n=64, n_children=3, seed=0, noise=0.05)
    support = np.zeros((X.shape[1], 2), dtype=np.int32)
    support[:3, 0] = 1
    support[3:, 1] = 1
    out = train_constrained(X, support, steps=30, seed=0)
    assert np.isfinite(out["train_mse0"])
    assert np.isfinite(out["train_mse"])
    assert out["train_mse"] < out["train_mse0"], (out["train_mse0"], out["train_mse"])
