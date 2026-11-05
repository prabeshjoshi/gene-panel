from itertools import combinations

import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression

from panel import rerun as rr
from panel import select as sel


def _Z(seed=0, n=120, p=30, rank=4, noise=0.4):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, rank)) @ rng.normal(size=(rank, p)) + noise * rng.normal(size=(n, p))
    return Z - Z.mean(axis=0)


CAND, TARG = np.arange(0, 30, 2), np.arange(1, 30, 2)   # disjoint blocks


# --- brief-required tests ------------------------------------------------------------------

def test_pivoted_qr_returns_k_distinct_candidate_indices():
    Z = _Z()
    for k in (1, 5, 15):
        p = rr.pivoted_qr_order(Z, CAND)[:k]
        assert len(p) == k and len(set(p.tolist())) == k
        assert set(p.tolist()) <= set(CAND.tolist())


def test_arm_c_selectors_never_pick_a_target_gene():
    Z = _Z(seed=1)
    tset = set(TARG.tolist())
    g, _ = rr.greedy_targets(Z, CAND, TARG, 6)
    assert tset.isdisjoint(g.tolist())
    assert tset.isdisjoint(rr.pivoted_qr_order(Z, CAND)[:6].tolist())
    panels, _ = rr.pls2_vip_iterative(Z, CAND, TARG, [3, 6], 0.10, 5, 3, seed=1)
    for p in panels.values():
        assert tset.isdisjoint(p.tolist())


def test_iterative_vip_returns_exactly_k_genes_for_every_k():
    Z = _Z(seed=2)
    panels, rounds = rr.pls2_vip_iterative(Z, CAND, TARG, [10, 5, 2], 0.10, 6, 3, seed=2)
    assert sorted(panels) == [2, 5, 10]
    for k, p in panels.items():
        assert len(p) == k and len(set(p.tolist())) == k
    assert set(panels[2].tolist()) <= set(panels[5].tolist()) <= set(panels[10].tolist())
    assert rounds[0]["remaining"] == len(CAND)


# --- correctness of the new pieces --------------------------------------------------------

def _rss(Z, panel, targ):
    S, T = Z[:, list(panel)], Z[:, list(targ)]
    B, *_ = np.linalg.lstsq(S, T, rcond=None)
    return float(((T - S @ B) ** 2).sum())


def test_greedy_targets_equals_pilot_greedy_when_blocks_coincide():
    Z = _Z(seed=3)
    allg = np.arange(Z.shape[1])
    assert list(rr.greedy_targets(Z, allg, allg, 5)[0]) == list(sel.greedy_forward(Z, 5)[0])


def test_greedy_targets_matches_bruteforce_greedy_on_disjoint_blocks():
    Z = _Z(seed=4)
    got, path = rr.greedy_targets(Z, CAND, TARG, 4)
    chosen = []
    for step in range(4):
        best = min((j for j in CAND if j not in chosen), key=lambda j: _rss(Z, chosen + [j], TARG))
        chosen.append(int(best))
        assert 1 - _rss(Z, chosen, TARG) / (Z[:, TARG] ** 2).sum() == pytest.approx(path[step], rel=1e-9)
    assert list(got) == chosen


def test_fast_pls2_matches_sklearn_predictions_and_vip():
    Z = _Z(seed=5)
    X, Y = Z[:90, CAND], Z[:90, TARG]
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    fit = rr.pls2_fit(Xc, Yc, 4)
    ref = PLSRegression(n_components=4, scale=False, max_iter=20000, tol=1e-14).fit(Xc, Yc)
    np.testing.assert_allclose(Xc @ fit["R"] @ fit["Q"].T, ref.predict(Xc) - Yc.mean(0) * 0,
                               rtol=1e-6, atol=1e-8)
    # reference VIP from the tightly converged sklearn fit (sel.pls_vip uses sklearn's tol=1e-6)
    ref_fit = {"W": ref.x_weights_, "T": ref.x_scores_, "Q": ref.y_loadings_}
    np.testing.assert_allclose(rr.vip_scores(fit), rr.vip_scores(ref_fit), rtol=1e-6)
    assert (rr.vip_scores(fit) ** 2).sum() == pytest.approx(len(CAND), rel=1e-9)


def test_pls2_cv_lv_is_deterministic_and_bounded():
    Z = _Z(seed=6)
    a1, t1 = rr.pls2_cv_lv(Z, CAND[:5], TARG, 20, 4, seed=3)
    a2, t2 = rr.pls2_cv_lv(Z, CAND[:5], TARG, 20, 4, seed=3)
    assert a1 == a2 and t1 == t2 and set(t1) == {1, 2, 3, 4, 5}


def test_pool_objective_with_target_block_equals_direct_ols():
    Z = _Z(seed=7)
    pool = CAND[:8]
    obj = sel.PoolObjective.fit(Z, pool, TARG)
    for S in [(0,), (1, 4), (0, 3, 7)]:
        assert obj(np.array(S)) == pytest.approx(_rss(Z, pool[list(S)], TARG), rel=1e-10)


def test_score_on_targets_disjoint_has_no_q2_and_matches_manual():
    Z = _Z(seed=8)
    Ztr, Zte = Z[:90], Z[90:]
    out = rr.score_on_targets(Ztr, Zte, CAND[:3], TARG, 10.0, disjoint=True)
    B = np.linalg.solve(Ztr[:, CAND[:3]].T @ Ztr[:, CAND[:3]] + 10 * np.eye(3), Ztr[:, CAND[:3]].T @ Ztr[:, TARG])
    Yh = Zte[:, CAND[:3]] @ B
    assert out["r2"] == pytest.approx(1 - ((Zte[:, TARG] - Yh) ** 2).sum() / (Zte[:, TARG] ** 2).sum())
    assert np.isnan(out["q2_nonpanel"])


def test_swap_local_search_reaches_a_true_1swap_optimum_and_never_worsens():
    Z = _Z(seed=9, p=24, noise=1.0)
    cand, targ = np.arange(0, 24, 2), np.arange(1, 24, 2)
    start = list(rr.greedy_targets(Z, cand, targ, 3)[0])
    panel, info = rr.swap_local_search(Z, cand, targ, start, max_iter=50)
    assert np.all(np.diff(info["rss_path"]) <= 1e-9)
    assert info["rss_path"][-1] == pytest.approx(_rss(Z, panel, targ), rel=1e-9)
    best_swap = min(_rss(Z, [g if g != i else j for g in panel], targ)
                    for i in panel for j in cand if j not in panel)
    assert best_swap >= _rss(Z, panel, targ) - 1e-8


def test_swap_local_search_finds_the_bruteforce_best_first_swap():
    Z = _Z(seed=10, p=20, noise=1.0)
    cand, targ = np.arange(0, 20, 2), np.arange(1, 20, 2)
    start = [0, 2, 4]
    _, info = rr.swap_local_search(Z, cand, targ, start, max_iter=1)
    best = min(_rss(Z, [g if g != i else j for g in start], targ)
               for i in start for j in cand if j not in start)
    assert info["rss_path"][-1] == pytest.approx(min(best, _rss(Z, start, targ)), rel=1e-9)
