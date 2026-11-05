"""Gene ranking and centring/scaling. Always fitted on training samples only."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
