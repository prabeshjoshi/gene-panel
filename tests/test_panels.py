import numpy as np
import pandas as pd

from panel import select as sel


def test_random_panels_shape_unique_and_reproducible():
    a = sel.random_panels(50, 7, 20, np.random.default_rng([1, 7, 1]))
    b = sel.random_panels(50, 7, 20, np.random.default_rng([1, 7, 1]))
    assert a.shape == (20, 7)
    assert np.array_equal(a, b)
    assert all(len(set(row)) == 7 for row in a)
    assert a.min() >= 0 and a.max() < 50
    assert np.all(np.diff(a, axis=1) > 0)  # sorted


def test_prepare_split_fits_on_training_only():
    rng = np.random.default_rng(0)
    n_tr, n_te = 30, 10
    X = rng.normal(size=(n_tr + n_te, 4))
    X[:, 0] *= 5                     # highest training variance
    X[:, 1] *= 3                     # second-highest training variance
    X[n_tr:, 3] *= 1000              # high variance ONLY in test rows
    ids = [f"s{i}" for i in range(n_tr + n_te)]
    df = pd.DataFrame(X, index=ids, columns=["g0", "g1", "g2", "g3"])
    split = {"train": ids[:n_tr], "test": ids[n_tr:]}
    Ztr, Zte, genes = sel.prepare_split(df, split, n_genes=2)
    assert list(genes) == ["g0", "g1"]  # test-only variance in g3 must not drive selection
    np.testing.assert_allclose(Ztr.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Ztr.std(axis=0, ddof=1), 1.0)
    col = list(df.columns).index(genes[0])
    m, s = X[:n_tr, col].mean(), X[:n_tr, col].std(ddof=1)
    np.testing.assert_allclose(Zte[:, 0], (X[n_tr:, col] - m) / s)
