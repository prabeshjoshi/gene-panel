"""Reconstruction metrics and the low-rank (PCA) baseline."""
from __future__ import annotations

import numpy as np


def r2_all(Z: np.ndarray, Zhat: np.ndarray) -> float:
    """R2_all = 1 - ||Z - Zhat||^2 / ||Z||^2 over all genes.

    Z is on the scale used for fitting (training-centred and scaled).
    """
    return float(1.0 - ((Z - Zhat) ** 2).sum() / (Z ** 2).sum())


def r2_subset(Z: np.ndarray, Zhat: np.ndarray, cols) -> float:
    """Same ratio restricted to the given gene columns (e.g. non-panel genes for Q2_nonpanel)."""
    Zs, Zhs = Z[:, cols], Zhat[:, cols]
    return float(1.0 - ((Zs - Zhs) ** 2).sum() / (Zs ** 2).sum())


def pca_reconstruction_r2(Z_train: np.ndarray, Z_test: np.ndarray, qs) -> dict:
    """Test R2 when test rows are reconstructed from the top q training PCs.

    Z_train must already be centred and scaled with training parameters, and Z_test must be
    transformed with the same parameters. The PCs come from the SVD of Z_train only.
    """
    _, _, Vt = np.linalg.svd(Z_train, full_matrices=False)
    out = {}
    for q in qs:
        q_eff = min(int(q), Vt.shape[0])
        V = Vt[:q_eff].T
        Zhat = (Z_test @ V) @ V.T
        out[int(q)] = r2_all(Z_test, Zhat)
    return out
