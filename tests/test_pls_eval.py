import numpy as np
from sklearn.cross_decomposition import PLSRegression

from panel import evaluate as ev


def _data(seed=0, n=120, p=15, rank=3, noise=0.3):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, rank)) @ rng.normal(size=(rank, p)) + noise * rng.normal(size=(n, p))
    return Z - Z[:90].mean(axis=0)


def test_truncated_pls_matches_a_fresh_fit_and_sklearn_predict():
    Z = _data()
    Ztr, Zte, panel = Z[:90], Z[90:], [0, 2, 5, 7, 11]
    R, Q = ev._pls_fit(Ztr[:, panel], Ztr, 5, 5000)
    for a in (1, 3, 5):
        ref = PLSRegression(n_components=a, scale=False, max_iter=5000).fit(Ztr[:, panel], Ztr)
        ours = Zte[:, panel] @ R[:, :a] @ Q[:, :a].T
        np.testing.assert_allclose(ours, ref.predict(Zte[:, panel]), rtol=1e-6, atol=1e-8)


def test_pls_with_all_lvs_equals_ols():
    Z = _data(seed=1)
    Ztr, Zte, panel = Z[:90], Z[90:], [1, 4, 6]
    pls = ev.score_panel_pls(Ztr, Zte, panel, n_lv=3, max_iter=5000)
    ols = ev.score_panel(Ztr, Zte, panel, alpha=1e-12)
    assert abs(pls["r2_all"] - ols["r2_all"]) < 1e-8


def test_cv_n_lv_is_deterministic_and_bounded_by_panel_size():
    Z = _data(seed=2)
    panel = [0, 3, 8, 9]
    a1, t1 = ev.pls_cv_n_lv(Z, panel, max_lv=10, n_folds=5, seed=1, max_iter=5000)
    a2, t2 = ev.pls_cv_n_lv(Z, panel, max_lv=10, n_folds=5, seed=1, max_iter=5000)
    assert a1 == a2 and t1 == t2
    assert set(t1) == {1, 2, 3, 4}
    assert t1[a1] == max(t1.values())
