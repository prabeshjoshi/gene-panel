#!/usr/bin/env python3
"""Rerun Step R0: sex calls, exclusions, and the three arms for every seed.

Reads config_rerun.yaml (must be committed and unmodified; its commit hash goes in the log) and
the pilot's config.yaml for data paths. Reuses data/processed/ and the stored splits.
Writes data/processed/sex_calls.csv (not committed) and results_rerun/r0/log.json.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from panel import arms  # noqa: E402
from panel import io as pio  # noqa: E402


def config_commit(path: Path) -> str:
    """Hash of the last commit touching the rerun config; refuse if it is uncommitted or edited."""
    rel = str(path.relative_to(ROOT))
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", rel],
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise SystemExit(f"{rel} has uncommitted changes; commit it before running analysis")
    h = subprocess.run(["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", rel],
                       capture_output=True, text=True, check=True).stdout.strip()
    if not h:
        raise SystemExit(f"{rel} is not committed")
    return h


def main() -> None:
    t0 = time.perf_counter()
    rcfg_path = ROOT / "config_rerun.yaml"
    commit = config_commit(rcfg_path)
    rcfg = pio.load_config(rcfg_path)
    pcfg = pio.load_config(ROOT / rcfg["pilot_config"])
    proc = ROOT / pcfg["processed_dir"]
    out_dir = ROOT / rcfg["results_dir"] / "r0"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"config_rerun.yaml commit {commit}")

    X = pd.read_parquet(proc / "X_logcpm.parquet")
    genes = pd.read_csv(proc / "genes.csv", keep_default_na=False)
    kept = genes[genes["status"] == "kept"].set_index("symbol")["gene_id"]
    with open(proc / "splits.json") as fh:
        splits = json.load(fh)["splits"]

    # --- sex calls --------------------------------------------------------------------------
    ids = {}
    for role in ("sex_male_marker", "sex_female_marker"):
        sym = rcfg[role]
        if sym not in kept.index or kept[sym] not in X.columns:
            raise SystemExit(f"marker {sym} is not in the processed matrix; stopping (ask the user)")
        ids[role] = kept[sym]
    calls = arms.sex_calls(X, ids["sex_male_marker"], ids["sex_female_marker"],
                           float(rcfg["sex_ambiguous_log2cpm"]))
    calls.index.name = "sample_id"
    calls.to_csv(proc / "sex_calls.csv")
    amb = calls[calls["ambiguous"]]
    included = calls.index[~calls["ambiguous"]]
    male = calls.loc[included, "sex"] == "male"
    n_m, n_f = int((calls.loc[included, "sex"] == "male").sum()), int((calls.loc[included, "sex"] == "female").sum())
    print(f"\nsex calls ({rcfg['sex_rule']}; ambiguous if both markers < or both >= "
          f"{rcfg['sex_ambiguous_log2cpm']} log2CPM):")
    print(f"  samples {len(calls)}: male {int((calls['sex'] == 'male').sum())}, "
          f"female {int((calls['sex'] == 'female').sum())}; ambiguous {len(amb)} "
          f"(both_low {int((amb['ambiguous_reason'] == 'both_low').sum())}, "
          f"both_high {int((amb['ambiguous_reason'] == 'both_high').sum())})")
    print(f"  included after exclusion: {len(included)} (male {n_m}, female {n_f})")
    for sid, r in amb.iterrows():
        print(f"    excluded {sid}: RPS4Y1 {r['male_marker_log2cpm']:.2f}, XIST {r['female_marker_log2cpm']:.2f} "
              f"({r['ambiguous_reason']})")
    margin = (calls.loc[included, "male_marker_log2cpm"] - calls.loc[included, "female_marker_log2cpm"]).abs()
    print(f"  smallest |RPS4Y1 - XIST| among included samples: {margin.min():.2f} log2CPM")

    # --- arms per seed -----------------------------------------------------------------------
    raw = X.loc[included]
    sex_num = male.astype(float)
    per_seed = {}
    for s in rcfg["seeds"]:
        sp = arms.filter_split(splits[str(s)], included)
        A = arms.build_arms(X.loc[included], sp, male, int(s), rcfg)
        tr = sp["train"]
        # report only: genes strongly associated with sex in TRAINING raw log2CPM
        xs = raw.loc[tr]
        r_sex = xs.apply(lambda col: np.corrcoef(col, sex_num.loc[tr])[0, 1] if col.std() > 0 else 0.0)
        strong = set(r_sex.index[r_sex.abs() > float(rcfg["sex_assoc_report_r"])])
        rec = {
            "n_train": len(sp["train"]), "n_test": len(sp["test"]),
            "excluded_from_train": len(splits[str(s)]["train"]) - len(sp["train"]),
            "excluded_from_test": len(splits[str(s)]["test"]) - len(sp["test"]),
            "male_train": int(male.loc[sp["train"]].sum()), "female_train": int((~male.loc[sp["train"]]).sum()),
            "male_test": int(male.loc[sp["test"]].sum()), "female_test": int((~male.loc[sp["test"]]).sum()),
            "arms": {k: {"n_genes": len(v.gene_ids), "n_candidates": len(v.cand), "n_targets": len(v.targ),
                         "n_sex_associated_genes": len(strong & set(v.gene_ids))}
                     for k, v in A.items()},
            "genes_A_and_B": len(set(A["A"].gene_ids) & set(A["B"].gene_ids)),
            "n_sex_associated_in_universe": len(strong),
            "arm_c_candidate_ids_head": A["C"].gene_ids[A["C"].cand[:5]].tolist(),
        }
        per_seed[str(s)] = rec

    print(f"\narms per seed (n_genes = {rcfg['n_genes']}; 'sex-assoc' = training |r(gene, male)| > "
          f"{rcfg['sex_assoc_report_r']} on raw log2CPM, report only):")
    print("  seed  train/test  excl tr/te  M/F train  M/F test   A genes (sex-assoc)  B genes (sex-assoc)  "
          "A&B shared  C cand/targ")
    for s, r in per_seed.items():
        a, b, c = r["arms"]["A"], r["arms"]["B"], r["arms"]["C"]
        print(f"  {s:>4}  {r['n_train']}/{r['n_test']}     {r['excluded_from_train']}/{r['excluded_from_test']}"
              f"        {r['male_train']}/{r['female_train']}    {r['male_test']}/{r['female_test']}    "
              f"{a['n_genes']} ({a['n_sex_associated_genes']:>2})          {b['n_genes']} ({b['n_sex_associated_genes']:>2})"
              f"           {r['genes_A_and_B']:>4}        {c['n_candidates']}/{c['n_targets']}")

    log = {
        "step": "R0",
        "config_rerun_commit": commit,
        "config_rerun": rcfg,
        "pilot_config": rcfg["pilot_config"],
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "markers": {"male": [rcfg["sex_male_marker"], ids["sex_male_marker"]],
                    "female": [rcfg["sex_female_marker"], ids["sex_female_marker"]]},
        "sex_calls": {
            "n_samples": len(calls),
            "male_all": int((calls["sex"] == "male").sum()), "female_all": int((calls["sex"] == "female").sum()),
            "ambiguous": len(amb),
            "ambiguous_samples": {sid: {"male_marker": float(r["male_marker_log2cpm"]),
                                        "female_marker": float(r["female_marker_log2cpm"]),
                                        "reason": r["ambiguous_reason"]} for sid, r in amb.iterrows()},
            "included": len(included), "included_male": n_m, "included_female": n_f,
            "min_marker_margin_included": float(margin.min()),
        },
        "per_seed": per_seed,
        "outputs": {"sex_calls": "data/processed/sex_calls.csv (not committed)"},
        "timings_s": {"total_s": time.perf_counter() - t0},
    }
    pio.write_json(log, out_dir / "log.json")
    print(f"\nwrote {out_dir / 'log.json'} ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
