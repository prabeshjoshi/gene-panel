"""Rerun arms: sex calls and the three per-split data arms (A reference, B sex-adjusted,
C disjoint candidate/target blocks). Everything fitted uses training samples only."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from panel.select import Standardizer, prepare_split, top_variance_genes


@dataclass
class ArmData:
    """One arm for one split. Z* are samples x genes, centred/scaled with training parameters.

    cand / targ are column indices into Z*: candidates a panel may use, targets it is scored on.
    Arms A and B: both are all genes. Arm C: disjoint halves.
    """

    arm: str
    Ztr: np.ndarray
    Zte: np.ndarray
    gene_ids: np.ndarray
    cand: np.ndarray
    targ: np.ndarray


def sex_calls(X: pd.DataFrame, male_col: str, female_col: str, thr: float) -> pd.DataFrame:
    """Male if log2CPM(male marker) > log2CPM(female marker), else female.

    Ambiguous if both markers are < thr (both low) or both >= thr (both high).
    X: samples x genes log2(CPM+1); male_col / female_col are its gene-id columns.
    """
    m = X[male_col].astype(float)
    f = X[female_col].astype(float)
    both_low = (m < thr) & (f < thr)
    both_high = (m >= thr) & (f >= thr)
    return pd.DataFrame({
        "male_marker_log2cpm": m, "female_marker_log2cpm": f,
        "sex": np.where(m > f, "male", "female"),
        "ambiguous": both_low | both_high,
        "ambiguous_reason": np.where(both_low, "both_low", np.where(both_high, "both_high", "")),
    }, index=X.index)


def filter_split(split: dict, keep) -> dict:
    """Drop excluded samples from a stored split, keeping order."""
    keep = set(keep)
    return {"train": [s for s in split["train"] if s in keep],
            "test": [s for s in split["test"] if s in keep]}


def sex_residualize(Xtr: np.ndarray, Xte: np.ndarray, male_tr: np.ndarray, male_te: np.ndarray):
    """Per gene OLS on [1, male] within training (= training group means), applied to both sets.

    Returns (R_train, R_test, intercept a, sex effect b).
    """
    male_tr = np.asarray(male_tr, dtype=bool)
    male_te = np.asarray(male_te, dtype=bool)
    if male_tr.all() or (~male_tr).all():
        raise ValueError("training split needs both sexes to estimate the sex effect")
    a = Xtr[~male_tr].mean(axis=0)
    b = Xtr[male_tr].mean(axis=0) - a
    return Xtr - a - np.outer(male_tr, b), Xte - a - np.outer(male_te, b), a, b


def arm_a(X: pd.DataFrame, split: dict, n_genes: int) -> ArmData:
    """Reference: exactly the pilot pipeline."""
    Ztr, Zte, ids = prepare_split(X, split, n_genes)
    allg = np.arange(Ztr.shape[1])
    return ArmData("A", Ztr, Zte, ids, allg, allg)


def arm_b(X: pd.DataFrame, split: dict, male: pd.Series, n_genes: int) -> ArmData:
    """Sex-adjusted: residualise on sex (training fit), then rank, select and scale on training."""
    Xtr = X.loc[split["train"]].to_numpy(dtype=np.float64)
    Xte = X.loc[split["test"]].to_numpy(dtype=np.float64)
    Rtr, Rte, _, _ = sex_residualize(Xtr, Xte, male.loc[split["train"]].to_numpy(),
                                     male.loc[split["test"]].to_numpy())
    top = top_variance_genes(Rtr, n_genes)
    std = Standardizer.fit(Rtr[:, top])
    allg = np.arange(len(top))
    return ArmData("B", std.transform(Rtr[:, top]), std.transform(Rte[:, top]),
                   X.columns[top].to_numpy(), allg, allg)


def arm_c(b: ArmData, seed: int, candidate_fraction: float, stream: int) -> ArmData:
    """Disjoint blocks: Arm B's genes split at random into candidates and targets (per seed)."""
    n = b.Ztr.shape[1]
    perm = np.random.default_rng([seed, stream]).permutation(n)
    n_cand = int(round(candidate_fraction * n))
    return ArmData("C", b.Ztr, b.Zte, b.gene_ids, np.sort(perm[:n_cand]), np.sort(perm[n_cand:]))


def build_arms(X: pd.DataFrame, split: dict, male: pd.Series, seed: int, cfg: dict) -> dict:
    """All three arms for one (already filtered) split."""
    a = arm_a(X, split, int(cfg["n_genes"]))
    b = arm_b(X, split, male, int(cfg["n_genes"]))
    c = arm_c(b, seed, float(cfg["arm_c_candidate_fraction"]), int(cfg["rng_stream_arm_c"]))
    return {"A": a, "B": b, "C": c}


def included_samples(X: pd.DataFrame, genes: pd.DataFrame, rcfg: dict):
    """Sex calls from the pre-registered markers; returns (calls, included ids, male Series)."""
    kept = genes[genes["status"] == "kept"].set_index("symbol")["gene_id"]
    ids = []
    for role in ("sex_male_marker", "sex_female_marker"):
        sym = rcfg[role]
        if sym not in kept.index or kept[sym] not in X.columns:
            raise SystemExit(f"marker {sym} is not in the processed matrix")
        ids.append(kept[sym])
    calls = sex_calls(X, ids[0], ids[1], float(rcfg["sex_ambiguous_log2cpm"]))
    included = calls.index[~calls["ambiguous"]]
    return calls, included, calls.loc[included, "sex"] == "male"


def sex_residualized_all(X: pd.DataFrame, male: pd.Series) -> pd.DataFrame:
    """Descriptive only (soft overlap): remove sex group means using all included samples."""
    Xi = X.loc[male.index]
    m = male.to_numpy()
    out = Xi.copy()
    out.loc[m] = Xi.loc[m] - Xi.loc[m].mean()
    out.loc[~m] = Xi.loc[~m] - Xi.loc[~m].mean()
    return out
