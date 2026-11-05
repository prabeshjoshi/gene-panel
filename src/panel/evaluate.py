"""Reconstruction metrics, the low-rank (PCA) baseline and the shared ridge predictor."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from panel.select import Standardizer


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


def ridge_coef(Z_panel: np.ndarray, Z_all: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge coefficients B (k x genes) from panel genes to all genes, no intercept.

    Inputs are training-centred, so no intercept is needed:
    B = (S'S + alpha I)^-1 S'Z.
    """
    k = Z_panel.shape[1]
    G = Z_panel.T @ Z_panel + alpha * np.eye(k)
    return np.linalg.solve(G, Z_panel.T @ Z_all)


def ridge_predict(Z_train: np.ndarray, Z_test: np.ndarray, panel, alpha: float) -> np.ndarray:
    """Fit ridge on training rows, predict all genes for test rows from the panel genes."""
    B = ridge_coef(Z_train[:, panel], Z_train, alpha)
    return Z_test[:, panel] @ B


def score_panel(Z_train: np.ndarray, Z_test: np.ndarray, panel, alpha: float) -> dict:
    """Shared evaluation: test R2_all over all genes and Q2_nonpanel over non-panel genes."""
    Zhat = ridge_predict(Z_train, Z_test, panel, alpha)
    nonpanel = np.setdiff1d(np.arange(Z_test.shape[1]), panel)
    return {"r2_all": r2_all(Z_test, Zhat), "q2_nonpanel": r2_subset(Z_test, Zhat, nonpanel)}


def cv_ridge_alpha(Z_train: np.ndarray, panels, alphas, n_folds: int, seed: int):
    """Choose one ridge alpha by K-fold CV on training rows only.

    Each fold is re-centred and re-scaled with its own fold-training statistics (an affine
    re-standardisation of Z_train equals standardising the raw data). The score for each alpha
    is the CV R2_all (pooled SSE / pooled SS over folds), averaged over the given panels.

    Returns (best_alpha, {alpha: mean CV R2_all}).
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    sse = np.zeros((len(panels), len(alphas)))
    ss = np.zeros(len(panels))
    for tr, va in kf.split(Z_train):
        std = Standardizer.fit(Z_train[tr])
        Ztr, Zva = std.transform(Z_train[tr]), std.transform(Z_train[va])
        for p, panel in enumerate(panels):
            ss[p] += (Zva ** 2).sum()
            for a, alpha in enumerate(alphas):
                Zhat = Zva[:, panel] @ ridge_coef(Ztr[:, panel], Ztr, alpha)
                sse[p, a] += ((Zva - Zhat) ** 2).sum()
    scores = (1.0 - sse / ss[:, None]).mean(axis=0)
    table = {float(a): float(s) for a, s in zip(alphas, scores)}
    return float(alphas[int(np.argmax(scores))]), table
