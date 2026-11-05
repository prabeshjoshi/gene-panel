"""Config loading, JSON writing and GTEx GCT parsing."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml (default: project root)."""
    path = Path(path) if path is not None else ROOT / "config.yaml"
    with open(path) as fh:
        return yaml.safe_load(fh)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_jsonable(obj), fh, indent=2)


def strip_version(ids) -> pd.Index:
    """ENSG00000223972.5 -> ENSG00000223972."""
    return pd.Index(ids).str.replace(r"\.\d+$", "", regex=True)


def donor_from_sample(sample_id: str) -> str:
    """GTEX-XXXXX-... -> GTEX-XXXXX (first two dash-separated fields)."""
    return "-".join(sample_id.split("-")[:2])


def read_gct(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a GTEx GCT (v1.2) gene-count file.

    Returns
    -------
    counts : genes x samples int64 DataFrame, index = Ensembl ID without version.
    gene_meta : DataFrame indexed by the same IDs with columns
        gene_id, gene_name_versioned, symbol.
    """
    with gzip.open(path, "rt") as fh:
        version = fh.readline().strip()
        dims = fh.readline().split()
    if version != "#1.2":
        raise ValueError(f"Unexpected GCT version line: {version!r}")
    # GCT dims line = (rows, samples); the table also has Name and Description columns.
    n_rows, n_samples = int(dims[0]), int(dims[1])

    raw = pd.read_csv(
        path,
        sep="\t",
        skiprows=2,
        compression="gzip",
        dtype={"Name": str, "Description": str},
    )
    if raw.shape != (n_rows, n_samples + 2):
        raise ValueError(f"GCT shape {raw.shape} does not match header rows={n_rows}, samples={n_samples}")

    gene_id = strip_version(raw["Name"])
    if gene_id.has_duplicates:
        raise ValueError("Duplicate Ensembl IDs after stripping version suffix")

    sample_ids = list(raw.columns[2:])
    counts = pd.DataFrame(
        raw.iloc[:, 2:].to_numpy(dtype=np.int64), index=gene_id, columns=sample_ids
    )
    gene_meta = pd.DataFrame(
        {
            "gene_id": gene_id,
            "gene_name_versioned": raw["Name"].to_numpy(),
            "symbol": raw["Description"].fillna("").to_numpy(),
        },
        index=gene_id,
    )
    return counts, gene_meta
