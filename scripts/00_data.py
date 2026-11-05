#!/usr/bin/env python3
"""Phase 0 setup: parse the GTEx GCT, filter, normalise, and write processed data and splits.

Outputs
-------
data/processed/X_logcpm.parquet   samples x genes, log2(CPM+1), after filters, NOT scaled
data/processed/genes.csv          every input gene with its status (kept / dropped reason)
data/processed/samples.csv        one row per kept sample (donor, library size, QC flag)
data/processed/splits.json        donor-level train/test sample IDs for every seed
data/processed/manifest.json      every threshold and the gene/sample count after each step
results/setup/log.json            config, seeds, timings, outputs
"""
from __future__ import annotations

import hashlib
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from panel import io as pio  # noqa: E402
from panel import preprocess as pp  # noqa: E402
from panel import split as sp  # noqa: E402


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    cfg = pio.load_config(ROOT / "config.yaml")
    proc = ROOT / cfg["processed_dir"]
    res = ROOT / cfg["results_dir"] / "setup"
    proc.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    t_all = time.perf_counter()

    # --- verify raw file -------------------------------------------------------------
    gct = ROOT / cfg["gct_file"]
    t = time.perf_counter()
    md5 = md5sum(gct)
    if md5 != cfg["gct_md5"]:
        raise SystemExit(f"md5 mismatch for {gct}: {md5} != {cfg['gct_md5']}")
    timings["md5_check_s"] = time.perf_counter() - t

    # --- parse -----------------------------------------------------------------------
    t = time.perf_counter()
    counts_all, gmeta = pio.read_gct(gct)
    timings["parse_s"] = time.perf_counter() - t
    n_samples_raw = counts_all.shape[1]

    manifest: dict = {
        "source": {"file": cfg["gct_file"], "md5": md5, "url": cfg["gct_url"]},
        "raw": {"genes": int(counts_all.shape[0]), "samples": int(n_samples_raw)},
        "steps": [],
    }

    def record(name, rule, n_before, n_after, **extra):
        entry = {"step": name, "rule": rule, "before": int(n_before), "after": int(n_after),
                 "removed": int(n_before - n_after)}
        entry.update(extra)
        manifest["steps"].append(entry)
        print(f"[{name}] {n_before} -> {n_after}  ({rule})")

    # Library size uses every gene in the file, before any gene is dropped.
    lib_all = pp.library_sizes(counts_all)

    # --- one sample per donor --------------------------------------------------------
    sample_ids_all = counts_all.columns.to_numpy()
    donors_all = np.array([pio.donor_from_sample(s) for s in sample_ids_all])
    donor_n = pd.Series(donors_all).value_counts()
    multi = donor_n[donor_n > 1]
    manifest["donors"] = {
        "n_donors_raw": int(donor_n.shape[0]),
        "n_donors_with_multiple_samples": int(multi.shape[0]),
        "multi_sample_donor_examples": {k: int(v) for k, v in multi.head(20).items()},
    }
    print(f"donors: {donor_n.shape[0]} unique; {multi.shape[0]} with >1 sample")

    keep_s = pp.donor_keep_mask(sample_ids_all, donors_all, lib_all.to_numpy(),
                                cfg["donor_keep_rule"])
    record("donor_dedup", cfg["donor_keep_rule"], n_samples_raw, int(keep_s.sum()),
           unit="samples")

    sample_ids = sample_ids_all[keep_s]
    donors = donors_all[keep_s]
    lib = lib_all[keep_s]
    counts = counts_all.loc[:, keep_s]
    del counts_all  # free memory; only the kept samples are used from here on

    # --- gene filters, in order ----------------------------------------------------
    cpm = pp.to_cpm(counts, lib)
    gene_status = pd.Series("kept", index=gmeta.index, dtype=object)

    # 1. drop mitochondrial genes
    n_before = len(gmeta)
    mt_keep = pp.mt_keep_mask(gmeta["symbol"], cfg["mt_prefix"])
    gene_status[~mt_keep.to_numpy()] = "dropped_mt"
    record("drop_mt", f"symbol startswith '{cfg['mt_prefix']}'", n_before, int(mt_keep.sum()),
           unit="genes")

    # 2. drop duplicated symbols (keep the higher mean CPM)
    alive = gene_status == "kept"
    mean_cpm = cpm.mean(axis=1)
    dedup_keep = pp.dedupe_symbol_keep_mask(gmeta.loc[alive, "symbol"], mean_cpm[alive])
    dropped_dup = dedup_keep.index[~dedup_keep.to_numpy()]
    gene_status.loc[dropped_dup] = "dropped_duplicate_symbol"
    record("drop_duplicate_symbols", "keep higher mean CPM", int(alive.sum()),
           int((gene_status == "kept").sum()), unit="genes")

    # 3. protein-coding filter: skipped for this pilot (no annotation fetched)
    if cfg.get("protein_coding_filter", False):
        raise SystemExit("protein_coding_filter=true is not implemented for this pilot")
    manifest["protein_coding_filter"] = {
        "applied": False,
        "note": "skipped: no gene annotation was fetched for this pilot, so non-protein-coding "
                "genes (lncRNA, pseudogenes, etc.) remain in the gene set",
    }

    # 4. expression filter: CPM >= min_cpm in >= min_frac_samples of kept samples
    alive = gene_status == "kept"
    expr_keep = pp.expression_keep_mask(cpm.loc[alive], cfg["min_cpm"], cfg["min_frac_samples"])
    dropped_expr = expr_keep.index[~expr_keep.to_numpy()]
    gene_status.loc[dropped_expr] = "dropped_low_expression"
    record("expression_filter",
           f"CPM>={cfg['min_cpm']} in >={cfg['min_frac_samples']} of samples",
           int(alive.sum()), int((gene_status == "kept").sum()), unit="genes")

    kept_genes = gene_status.index[gene_status == "kept"]
    print(f"genes kept: {len(kept_genes)}")

    # --- normalise -------------------------------------------------------------------
    logcpm = pp.log2cpm(cpm.loc[kept_genes])  # genes x samples
    X = logcpm.T.astype(np.float32)           # samples x genes, NOT scaled
    X.index.name = "sample_id"
    X.columns.name = "gene_id"

    # --- sample QC flags (flag, do not drop) -----------------------------------------
    qc = pp.library_qc_flags(lib, cfg["qc_mad_threshold"])
    samples = pd.DataFrame({
        "sample_id": sample_ids,
        "donor": donors,
        "library_size": lib.to_numpy().astype(np.int64),
        "log10_library_size": np.log10(lib.to_numpy()),
        "qc_flag_library": qc.to_numpy(),
    })
    manifest["qc"] = {
        "rule": f"|log10(lib) - median| > {cfg['qc_mad_threshold']} * MAD (raw MAD); flagged, not dropped",
        "n_flagged": int(samples["qc_flag_library"].sum()),
    }
    print(f"QC flagged samples: {manifest['qc']['n_flagged']}")

    # --- gene table ------------------------------------------------------------------
    genes_out = gmeta.assign(mean_cpm=mean_cpm.to_numpy(), status=gene_status.to_numpy())
    genes_out = genes_out[["gene_id", "gene_name_versioned", "symbol", "mean_cpm", "status"]]

    # --- splits (donor level, one per seed) ----------------------------------------
    seeds = sorted(set(cfg["seeds_gates"]) | set(cfg["seeds_stability"]))
    splits = {}
    for seed in seeds:
        s = sp.donor_split(samples["sample_id"], samples["donor"], cfg["test_frac"], seed)
        train_donors = set(samples.set_index("sample_id").loc[s["train"], "donor"])
        test_donors = set(samples.set_index("sample_id").loc[s["test"], "donor"])
        assert not (train_donors & test_donors), "donor leakage between train and test"
        splits[str(seed)] = {"n_train": len(s["train"]), "n_test": len(s["test"]),
                             "train": s["train"], "test": s["test"]}
    manifest["splits"] = {"seeds": seeds, "test_frac": cfg["test_frac"],
                          "rule": "round(test_frac * n_donors) test donors via default_rng(seed)"}

    # --- write -----------------------------------------------------------------------
    t = time.perf_counter()
    X.to_parquet(proc / "X_logcpm.parquet")
    genes_out.to_csv(proc / "genes.csv", index=False)
    samples.to_csv(proc / "samples.csv", index=False)
    pio.write_json({"seeds": seeds, "test_frac": cfg["test_frac"], "splits": splits},
                   proc / "splits.json")
    timings["write_s"] = time.perf_counter() - t

    manifest["final"] = {"samples": int(X.shape[0]), "genes": int(X.shape[1])}
    manifest["thresholds"] = {k: cfg[k] for k in
                              ["mt_prefix", "donor_keep_rule", "min_cpm", "min_frac_samples",
                               "test_frac", "qc_mad_threshold"]}
    manifest["versions"] = {"python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__}
    pio.write_json(manifest, proc / "manifest.json")

    timings["total_s"] = time.perf_counter() - t_all
    log = {
        "phase": "setup",
        "config": {k: cfg[k] for k in cfg},
        "seeds": {"gates": cfg["seeds_gates"], "stability": cfg["seeds_stability"]},
        "timings_s": timings,
        "manifest": manifest,
        "outputs": {
            "X_logcpm": {"path": "data/processed/X_logcpm.parquet", "shape": list(X.shape)},
            "genes": {"path": "data/processed/genes.csv", "rows": len(genes_out)},
            "samples": {"path": "data/processed/samples.csv", "rows": len(samples)},
            "splits": {"path": "data/processed/splits.json", "seeds": seeds},
            "manifest": {"path": "data/processed/manifest.json"},
        },
    }
    pio.write_json(log, res / "log.json")
    print(f"wrote processed data; total {timings['total_s']:.1f}s")


if __name__ == "__main__":
    main()
