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
