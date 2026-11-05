"""Gene and sample filters, CPM and log transforms. Nothing here is fitted to outcomes."""
from __future__ import annotations

import numpy as np
import pandas as pd


def library_sizes(counts: pd.DataFrame) -> pd.Series:
    """Total reads per sample over all genes in the file (genes x samples input)."""
    return counts.sum(axis=0).astype(np.float64)


def to_cpm(counts: pd.DataFrame, lib: pd.Series) -> pd.DataFrame:
    """Counts per million, dividing each sample column by its library size."""
    return counts.astype(np.float64).div(lib, axis=1) * 1e6


def log2cpm(cpm: pd.DataFrame | np.ndarray):
    return np.log2(cpm + 1.0)


def mt_keep_mask(symbols: pd.Series, prefix: str) -> pd.Series:
    """True for genes whose symbol does not start with the mitochondrial prefix."""
    return ~symbols.str.startswith(prefix)


def dedupe_symbol_keep_mask(symbols: pd.Series, score: pd.Series) -> pd.Series:
    """For each repeated non-empty symbol keep the gene with the highest score.

    Ties go to the first gene in index order. Empty symbols are never treated as duplicates.
    """
    keep = pd.Series(True, index=symbols.index)
    df = pd.DataFrame({"sym": symbols, "score": score})
    repeated = df[(df["sym"] != "") & df["sym"].duplicated(keep=False)]
    for _, grp in repeated.groupby("sym", sort=False):
        best = grp["score"].idxmax()
        keep.loc[grp.index[grp.index != best]] = False
    return keep


def expression_keep_mask(cpm: pd.DataFrame, min_cpm: float, min_frac: float) -> pd.Series:
    """Keep genes with CPM >= min_cpm in at least min_frac of samples (genes x samples input)."""
    frac = (cpm >= min_cpm).mean(axis=1)
    return frac >= min_frac


def donor_keep_mask(sample_ids, donors, lib, rule: str) -> np.ndarray:
    """Boolean mask keeping exactly one sample per donor.

    rule = "max_library_size": keep the sample with the largest library size;
    ties are broken by sample ID so the choice is deterministic.
    """
    if rule != "max_library_size":
        raise ValueError(f"Unknown donor_keep_rule: {rule}")
    df = pd.DataFrame(
        {"sample": np.asarray(sample_ids), "donor": np.asarray(donors), "lib": np.asarray(lib)}
    )
    ordered = df.sort_values(["donor", "lib", "sample"], ascending=[True, False, True])
    kept = set(ordered.drop_duplicates("donor", keep="first")["sample"])
    return df["sample"].isin(kept).to_numpy()


def library_qc_flags(lib: pd.Series, threshold: float) -> pd.Series:
    """Flag samples whose log10 library size is more than `threshold` raw MADs from the median.

    Flags only; no sample is dropped. A zero MAD flags any sample off the median.
    """
    x = np.log10(lib.astype(np.float64))
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return pd.Series(np.abs(x - med) > threshold * mad, index=lib.index)
