"""Config loading, JSON writing and GTEx GCT parsing."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _parse_value(text: str):
    """Parse one flat-YAML value: "quoted string", number, true/false, or [list]."""
    text = text.strip()
    if text.startswith('"'):
        end = text.index('"', 1)
        return text[1:end]
    text = text.split(" #", 1)[0].strip()  # trailing comment
    if text in ("true", "false"):
        return text == "true"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def load_config(path: str | Path | None = None) -> dict:
    """Load config.yaml (default: project root).

    The config is deliberately flat (one `key: value` per line; values are quoted strings,
    numbers, true/false or JSON-style lists), so it is parsed without a YAML dependency.
    """
    path = Path(path) if path is not None else ROOT / "config.yaml"
    cfg: dict = {}
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if ":" not in s:
                raise ValueError(f"{path}:{lineno}: expected 'key: value'")
            key, val = s.split(":", 1)
            key = key.strip()
            if key in cfg:
                raise ValueError(f"{path}:{lineno}: duplicate key {key!r}")
            cfg[key] = _parse_value(val)
    return cfg


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


def committed_config_hash(root: Path, path: Path) -> str:
    """Hash of the last commit touching `path`; refuse if it is uncommitted or edited.

    Used by every rerun script so each log can name the pre-registered config it ran under.
    """
    import subprocess

    rel = str(Path(path).resolve().relative_to(Path(root).resolve()))
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", rel],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"{rel} has uncommitted changes; commit it before running analysis")
    h = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%H", "--", rel],
                       capture_output=True, text=True, check=True).stdout.strip()
    if not h:
        raise SystemExit(f"{rel} is not committed")
    return h
