import numpy as np
import pytest

from panel import select as sel


def _ols_r2(Z, panel):
    S = Z[:, list(panel)]
    B, *_ = np.linalg.lstsq(S, Z, rcond=None)
    return 1.0 - ((Z - S @ B) ** 2).sum() / (Z ** 2).sum()


def _lowrank(seed=0, n=60, p=15, rank=3, noise=0.3):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, rank)) @ rng.normal(size=(rank, p)) + noise * rng.normal(size=(n, p))
    return Z - Z.mean(axis=0)


def test_greedy_matches_bruteforce_greedy_and_reports_ols_r2():
    Z = _lowrank()
    sel_idx, r2 = sel.greedy_forward(Z, 4)
    chosen = []
    for step in range(4):
        cands = [j for j in range(Z.shape[1]) if j not in chosen]
        best = max(cands, key=lambda j: _ols_r2(Z, chosen + [j]))
        chosen.append(best)
        assert r2[step] == pytest.approx(_ols_r2(Z, chosen), rel=1e-9)
    assert list(sel_idx) == chosen
    assert np.all(np.diff(r2) > 0)


def test_greedy_first_pick_is_best_single_gene():
    Z = _lowrank(seed=4)
    first = sel.greedy_forward(Z, 1)[0][0]
    best = max(range(Z.shape[1]), key=lambda j: _ols_r2(Z, [j]))
    assert first == best


def test_greedy_skips_duplicate_gene():
    Z = _lowrank(seed=2)
    Z = np.column_stack([Z, Z[:, 0]])          # exact copy of gene 0
    picks = sel.greedy_forward(Z, 6)[0]
    assert not ({0, Z.shape[1] - 1} <= set(picks.tolist()))


def test_leverage_picks_the_genes_that_carry_the_variance():
    rng = np.random.default_rng(1)
    Z = 0.01 * rng.normal(size=(100, 10))
    Z[:, 3] += 5 * rng.normal(size=100)
    Z[:, 7] += 5 * rng.normal(size=100)
    assert set(sel.leverage_panel(Z, 2).tolist()) == {3, 7}


def test_cluster_medoids_one_per_correlated_block():
    rng = np.random.default_rng(2)
    n, blocks = 200, [range(0, 4), range(4, 9), range(9, 12)]
    Z = np.zeros((n, 12))
    for b in blocks:
        f = rng.normal(size=n)
        for j in b:
            Z[:, j] = f + 0.2 * rng.normal(size=n)
    out = sel.cluster_medoid_panels(Z, [3], "average")
    medoids, n_clusters = out[3]
    assert n_clusters == 3
    assert sorted(next(i for i, b in enumerate(blocks) if m in b) for m in medoids) == [0, 1, 2]
    # medoid property: smallest summed 1-|r| distance inside its block
    D = 1 - np.abs(np.corrcoef(Z, rowvar=False))
    for m in medoids:
        b = list(next(b for b in blocks if m in b))
        assert m == b[int(np.argmin(D[np.ix_(b, b)].sum(axis=1)))]


def test_pc_targets_are_uncorrelated_scores():
    Z = _lowrank(seed=5)
    T = sel.pc_targets(Z, 3)
    G = T.T @ T
    np.testing.assert_allclose(G - np.diag(np.diag(G)), 0, atol=1e-8)
    assert np.all(np.diff(np.diag(G)) <= 1e-9)


def _sparse_driver(seed=3, n=300, p=20, noise=0.05):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, p))
    T = Z[:, [0, 1, 2]] @ rng.normal(size=(3, 4)) * 3 + noise * rng.normal(size=(n, 4))
    return Z - Z.mean(0), T - T.mean(0)


def test_mtl_selects_the_true_drivers_and_hits_k():
    Z, T = _sparse_driver()
    out = sel.mtl_panels(Z, T, [2, 3], n_alphas=60, alpha_min_ratio=1e-3, max_iter=5000, tol=1e-6)
    assert set(out[3][0].tolist()) == {0, 1, 2}
    assert set(out[2][0].tolist()) < {0, 1, 2}
    assert len(out[2][0]) == 2 and out[2][1]["nnz_at_alpha"] >= 2


def test_mtl_raises_when_path_cannot_reach_k():
    Z, T = _sparse_driver()  # only 3 genes drive T; a short path never admits 10
    with pytest.raises(RuntimeError):
        sel.mtl_panels(Z, T, [10], n_alphas=10, alpha_min_ratio=0.5, max_iter=5000, tol=1e-6)


def test_pls_vip_sums_to_p_and_ranks_drivers_first():
    Z, T = _sparse_driver()
    vip = sel.pls_vip(Z, T, n_components=3)
    assert (vip ** 2).sum() == pytest.approx(Z.shape[1], rel=1e-8)
    assert set(sel.top_k(vip, 3).tolist()) == {0, 1, 2}
