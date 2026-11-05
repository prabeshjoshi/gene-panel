import numpy as np

from panel import select as sel


def test_top_variance_picks_most_variable_columns():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 6))
    X[:, 4] *= 10
    X[:, 1] *= 5
    top = sel.top_variance_genes(X, 2)
    assert list(top) == [4, 1]


def test_top_variance_caps_at_number_of_genes():
    X = np.arange(12, dtype=float).reshape(4, 3)
    assert len(sel.top_variance_genes(X, 99)) == 3


def test_standardizer_uses_training_parameters_on_test():
    Xtr = np.array([[0.0, 10.0], [2.0, 10.0], [4.0, 10.0]])
    Xte = np.array([[10.0, 20.0]])
    std = sel.Standardizer.fit(Xtr)
    Ztr = std.transform(Xtr)
    Zte = std.transform(Xte)
    np.testing.assert_allclose(Ztr.mean(axis=0)[0], 0.0, atol=1e-12)
    # test row uses the TRAINING mean (2.0) and SD (2.0), not its own
    assert Zte[0, 0] == (10.0 - 2.0) / 2.0
    # constant training gene is centred but not divided by zero
    assert np.all(np.isfinite(Ztr)) and np.all(np.isfinite(Zte))
    np.testing.assert_allclose(Ztr[:, 1], 0.0)
