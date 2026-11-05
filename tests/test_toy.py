from itertools import combinations

import numpy as np
import pytest

from panel import select as sel


def _Z(seed=0, n=50, p=14, rank=3, noise=0.4):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, rank)) @ rng.normal(size=(rank, p)) + noise * rng.normal(size=(n, p))
    return Z - Z.mean(axis=0)


def _ols_rss(Z, cols):
    S = Z[:, list(cols)]
    B, *_ = np.linalg.lstsq(S, Z, rcond=None)
    return float(((Z - S @ B) ** 2).sum())


def test_objective_equals_direct_ols_residual():
    Z = _Z()
    pool = np.array([1, 3, 4, 8, 9, 12])
    obj = sel.PoolObjective.fit(Z, pool)
    for S in [(0,), (1, 4), (0, 2, 5), (1, 2, 3, 5)]:
        assert obj(np.array(S)) == pytest.approx(_ols_rss(Z, pool[list(S)]), rel=1e-10)


def test_exhaustive_matches_bruteforce_and_batching_is_invariant():
    Z = _Z(seed=1)
    pool = np.arange(2, 12)
    obj = sel.PoolObjective.fit(Z, pool)
    for k in (2, 3):
        best, fbest, fall = sel.exhaustive_optimum(obj, len(pool), k, batch_size=7)
        _, _, fall_big = sel.exhaustive_optimum(obj, len(pool), k, batch_size=10_000)
        np.testing.assert_allclose(fall, fall_big, rtol=1e-12)
        brute = min(combinations(range(len(pool)), k), key=lambda S: _ols_rss(Z, pool[list(S)]))
        assert tuple(sorted(best.tolist())) == brute
        assert fbest == pytest.approx(_ols_rss(Z, pool[list(brute)]), rel=1e-10)


def test_greedy_on_objective_matches_fast_greedy_on_full_pool():
    Z = _Z(seed=2)
    pool = np.arange(Z.shape[1])
    obj = sel.PoolObjective.fit(Z, pool)
    assert list(sel.greedy_on_objective(obj, len(pool), 4)) == list(sel.greedy_forward(Z, 4)[0])


def test_greedy_never_beats_exhaustive():
    Z = _Z(seed=3, noise=1.0)
    pool = np.arange(Z.shape[1])
    obj = sel.PoolObjective.fit(Z, pool)
    for k in (2, 3, 4):
        g = sel.greedy_on_objective(obj, len(pool), k)
        _, fbest, _ = sel.exhaustive_optimum(obj, len(pool), k, batch_size=500)
        assert obj(g) >= fbest - 1e-9
