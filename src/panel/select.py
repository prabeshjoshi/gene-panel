"""Gene ranking, centring/scaling and panel sampling. Always fitted on training samples only."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def top_variance_genes(X_train: np.ndarray, n: int) -> np.ndarray:
    """Column indices of the n highest-variance genes in the training matrix.

    Variance uses ddof=1. Ties keep the lower column index (stable sort).
    """
    var = X_train.var(axis=0, ddof=1)
    order = np.argsort(-var, kind="stable")
    return order[: min(n, X_train.shape[1])]


@dataclass
class Standardizer:
    """Per-gene centring and scaling with training mean and SD."""

    mean_: np.ndarray
    sd_: np.ndarray

    @classmethod
    def fit(cls, X_train: np.ndarray) -> "Standardizer":
        mean = X_train.mean(axis=0)
        sd = X_train.std(axis=0, ddof=1)
        sd = np.where(sd > 0, sd, 1.0)  # constant genes are only centred
        return cls(mean_=mean, sd_=sd)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the training parameters to any matrix (train or test)."""
        return (X - self.mean_) / self.sd_


def prepare_split(X: pd.DataFrame, split: dict, n_genes: int):
    """Per-split preparation shared by every phase.

    Rank genes by TRAINING variance of log-CPM, keep the top n_genes, then centre and scale
    each with training mean and SD; the test rows get the same transform.

    Returns (Z_train, Z_test, gene_ids).
    """
    Xtr = X.loc[split["train"]].to_numpy(dtype=np.float64)
    Xte = X.loc[split["test"]].to_numpy(dtype=np.float64)
    top = top_variance_genes(Xtr, n_genes)
    std = Standardizer.fit(Xtr[:, top])
    return std.transform(Xtr[:, top]), std.transform(Xte[:, top]), X.columns[top].to_numpy()


def random_panels(n_genes: int, k: int, n_panels: int, rng: np.random.Generator) -> np.ndarray:
    """n_panels random k-gene panels (column indices, sorted), drawn without replacement."""
    return np.array([np.sort(rng.choice(n_genes, size=k, replace=False)) for _ in range(n_panels)])


# --------------------------------------------------------------------------------------
# Panel-selection heuristics. Every function sees training data only (Z = training rows,
# centred and scaled with training parameters).
# --------------------------------------------------------------------------------------

def pc_targets(Z: np.ndarray, n_pc: int) -> np.ndarray:
    """Scores on the top n_pc training PCs (unscaled, so PCs keep their variance weights)."""
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    return Z @ Vt[:n_pc].T


def greedy_forward(Z: np.ndarray, k: int, eps: float = 1e-10):
    """Greedy forward selection maximising training R2_all under OLS.

    Works in gene-gene space: M = R'R where R is the residual of Z after projecting onto the
    selected genes. Adding gene j removes ||M[:, j]||^2 / M[j, j] of residual sum of squares;
    after choosing j, M <- M - m_j m_j' / M[j, j]. Returns (genes in selection order,
    training R2_all after each addition). Prefixes give the greedy panel for every smaller k.
    """
    M = Z.T @ Z
    total = np.trace(M)
    diag0 = np.diag(M).copy()
    sel: list[int] = []
    r2 = []
    for _ in range(k):
        d = np.diag(M)
        valid = d > eps * diag0
        valid[sel] = False
        gains = np.full(M.shape[0], -np.inf)
        gains[valid] = (M[:, valid] ** 2).sum(axis=0) / d[valid]
        j = int(np.argmax(gains))
        mj = M[:, j].copy()
        M -= np.outer(mj, mj) / mj[j]
        sel.append(j)
        r2.append(1.0 - np.trace(M) / total)
    return np.array(sel), np.array(r2)


def leverage_panel(Z: np.ndarray, k: int) -> np.ndarray:
    """The k genes with the largest leverage on the top-k right singular vectors."""
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    lev = (Vt[:k] ** 2).sum(axis=0)
    return np.argsort(-lev, kind="stable")[:k]


def cluster_medoid_panels(Z: np.ndarray, ks, method: str) -> dict:
    """Hierarchical clustering of genes on 1 - |r|, cut into k clusters; one medoid per cluster.

    The medoid is the gene with the smallest summed distance to the rest of its cluster.
    Returns {k: (medoid indices, number of clusters actually formed)}.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    D = 1.0 - np.abs(np.corrcoef(Z, rowvar=False))
    D = np.clip((D + D.T) / 2.0, 0.0, None)
    np.fill_diagonal(D, 0.0)
    L = linkage(squareform(D, checks=False), method=method)
    out = {}
    for k in ks:
        labels = fcluster(L, t=k, criterion="maxclust")
        medoids = []
        for c in np.unique(labels):
            idx = np.flatnonzero(labels == c)
            medoids.append(int(idx[np.argmin(D[np.ix_(idx, idx)].sum(axis=1))]))
        out[int(k)] = (np.array(medoids), int(len(np.unique(labels))))
    return out


def mtl_panels(Z: np.ndarray, T: np.ndarray, ks, n_alphas: int, alpha_min_ratio: float,
               max_iter: int, tol: float) -> dict:
    """MultiTaskLasso from all genes to the PC-score targets, with alpha tuned to k genes.

    Walks a log-spaced alpha path from alpha_max (no gene selected) downwards with warm starts.
    For each k, takes the first alpha at which >= k genes have non-zero coefficient rows and
    keeps the k genes with the largest row norms. Returns {k: (genes, info)}.
    """
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import MultiTaskLasso

    n = Z.shape[0]
    alpha_max = np.sqrt(((Z.T @ T) ** 2).sum(axis=1)).max() / n
    alphas = alpha_max * np.logspace(0, np.log10(alpha_min_ratio), n_alphas)[1:]
    model = MultiTaskLasso(alpha=alphas[0], fit_intercept=False, warm_start=True,
                           max_iter=max_iter, tol=tol)
    out: dict = {}
    n_warn = 0
    for a in alphas:
        model.set_params(alpha=a)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(Z, T)
        n_warn += sum(issubclass(x.category, ConvergenceWarning) for x in w)
        norms = np.linalg.norm(model.coef_, axis=0)  # coef_: (n_targets, n_genes)
        nnz = int((norms > 0).sum())
        for k in ks:
            if int(k) not in out and nnz >= k:
                genes = np.argsort(-norms, kind="stable")[:k]
                out[int(k)] = (genes, {"alpha": float(a), "alpha_over_max": float(a / alpha_max),
                                       "nnz_at_alpha": nnz})
        if len(out) == len(ks):
            break
    missing = [k for k in ks if int(k) not in out]
    if missing:
        raise RuntimeError(f"MultiTaskLasso path never reached k={missing}; lower alpha_min_ratio")
    for k in out:
        out[k][1]["convergence_warnings_on_path"] = n_warn
    return out


def pls_vip(Z: np.ndarray, T: np.ndarray, n_components: int, max_iter: int = 500) -> np.ndarray:
    """PLS2 VIP scores from all genes to the PC-score targets (no rescaling of X or Y).

    VIP_j = sqrt(p * sum_a SS_a (w_ja / ||w_a||)^2 / sum_a SS_a), SS_a = ||t_a||^2 ||q_a||^2.
    Raises if NIPALS does not converge within max_iter.
    """
    import warnings

    from sklearn.cross_decomposition import PLSRegression
    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pls = PLSRegression(n_components=n_components, scale=False, max_iter=max_iter).fit(Z, T)
    W, S, Q = pls.x_weights_, pls.x_scores_, pls.y_loadings_
    ss = (S ** 2).sum(axis=0) * (Q ** 2).sum(axis=0)
    Wn = W / np.linalg.norm(W, axis=0)
    return np.sqrt(Z.shape[1] * ((Wn ** 2) @ ss) / ss.sum())


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-scores, kind="stable")[:k]
