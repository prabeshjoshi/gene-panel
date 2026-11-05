import numpy as np
import pytest

from panel import evaluate as ev


def test_r2_all_perfect_and_zero_predictions():
    Z = np.arange(1.0, 13.0).reshape(3, 4)
    assert ev.r2_all(Z, Z) == pytest.approx(1.0)
    assert ev.r2_all(Z, np.zeros_like(Z)) == pytest.approx(0.0)


def test_r2_subset_matches_manual_ratio():
    Z = np.array([[1.0, 2.0, 3.0]])
    Zh = np.array([[1.0, 0.0, 3.0]])
    # non-panel columns [1]: 1 - 4/4 = 0
    assert ev.r2_subset(Z, Zh, [1]) == pytest.approx(0.0)
    assert ev.r2_subset(Z, Zh, [0, 2]) == pytest.approx(1.0)


def test_pca_reconstruction_full_rank_is_exact_and_monotone():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(60, 8))
    Ztr = A[:40] - A[:40].mean(axis=0)
    Zte = A[40:] - A[:40].mean(axis=0)
    r2 = ev.pca_reconstruction_r2(Ztr, Zte, [1, 2, 4, 8])
    vals = [r2[q] for q in [1, 2, 4, 8]]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))
    assert r2[8] == pytest.approx(1.0, abs=1e-10)


def test_pca_uses_only_training_components():
    # If the test rows are exactly in the training span, q = rank reconstructs them perfectly
    rng = np.random.default_rng(1)
    basis = rng.normal(size=(2, 10))
    Ztr = rng.normal(size=(30, 2)) @ basis
    Zte = rng.normal(size=(5, 2)) @ basis
    r2 = ev.pca_reconstruction_r2(Ztr, Zte, [2])
    assert r2[2] == pytest.approx(1.0, abs=1e-10)
