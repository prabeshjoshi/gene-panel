import numpy as np
import pytest
from sklearn.linear_model import Ridge

from panel import evaluate as ev


def _data(seed=0, n=80, g=12, rank=3, noise=0.1):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, rank)) @ rng.normal(size=(rank, g)) + noise * rng.normal(size=(n, g))
    return Z - Z[:60].mean(axis=0)


def test_ridge_matches_sklearn_without_intercept():
    Z = _data()
    Ztr, Zte, panel = Z[:60], Z[60:], [0, 4, 7]
    ours = ev.ridge_predict(Ztr, Zte, panel, alpha=2.5)
    sk = Ridge(alpha=2.5, fit_intercept=False).fit(Ztr[:, panel], Ztr).predict(Zte[:, panel])
    np.testing.assert_allclose(ours, sk, rtol=1e-8, atol=1e-10)


def test_score_panel_q2_excludes_panel_genes():
    Z = _data()
    Ztr, Zte, panel = Z[:60], Z[60:], [1, 2]
    out = ev.score_panel(Ztr, Zte, panel, alpha=1e-8)
    Zhat = ev.ridge_predict(Ztr, Zte, panel, 1e-8)
    nonpanel = [j for j in range(Z.shape[1]) if j not in panel]
    assert out["q2_nonpanel"] == pytest.approx(ev.r2_subset(Zte, Zhat, nonpanel))
    assert out["r2_all"] == pytest.approx(ev.r2_all(Zte, Zhat))
    # panel genes predict themselves almost exactly, so R2_all > Q2_nonpanel
    assert out["r2_all"] > out["q2_nonpanel"]


def test_cv_alpha_is_deterministic_and_returns_grid_argmax():
    Z = _data(seed=3)
    panels = [np.array([0, 1, 2]), np.array([3, 5, 8])]
    alphas = [0.01, 1.0, 100.0, 1e4]
    a1, t1 = ev.cv_ridge_alpha(Z, panels, alphas, n_folds=5, seed=1)
    a2, t2 = ev.cv_ridge_alpha(Z, panels, alphas, n_folds=5, seed=1)
    assert a1 == a2 and t1 == t2
    assert a1 in alphas
    assert t1[a1] == max(t1.values())
    # huge shrinkage destroys a good low-rank fit
    assert t1[1e4] < t1[a1]
