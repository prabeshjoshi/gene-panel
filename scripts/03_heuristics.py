#!/usr/bin/env python3
"""Phase 2 — heuristics: do common panel-selection heuristics beat random panels?

At each k in k_heuristics, on every gate seed, select a panel from TRAINING data with
    greedy    greedy forward selection maximising training R2_all (OLS)
    leverage  top-k leverage on the top-k right singular vectors
    cluster   hierarchical clustering on 1 - |r| (cluster_linkage), k clusters, medoids
    mtl       MultiTaskLasso to the top n_pc_targets PC scores, alpha tuned to k genes
    pls_vip   PLS2 VIP to the same PC-score targets
then score every panel with the shared evaluation (ridge, alpha fixed in Phase 1): test R2_all
and Q2_nonpanel. The random baseline is the median of the Phase 1 random panels (same RNG
streams, so identical numbers).

GATE at k = gate_heuristics_k, with m = gate_noise_multiple_of_sd and paired per-seed differences:
    A. best method (highest mean test R2_all) minus random median: mean > m x SD
    B. at least one pair of methods differs: |mean difference| > m x SD
GO if A and B.

Writes results/phase2/log.json and results/phase2/heuristics.png.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from panel import evaluate as ev  # noqa: E402
from panel import io as pio  # noqa: E402
from panel import plots  # noqa: E402
from panel import select as sel  # noqa: E402

METHODS = sel.HEURISTICS
select_panels = sel.heuristic_panels


def paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    d = np.asarray(a) - np.asarray(b)
    return float(d.mean()), float(d.std(ddof=1))


def main() -> None:
    cfg = pio.load_config(ROOT / "config.yaml")
    proc = ROOT / cfg["processed_dir"]
    res = ROOT / cfg["results_dir"]
    out_dir = res / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in cfg["seeds_gates"]]
    ks = [int(k) for k in cfg["k_heuristics"]]
    gate_k = int(cfg["gate_heuristics_k"])
    m_sd = float(cfg["gate_noise_multiple_of_sd"])
    n_panels = int(cfg["n_random_panels"])
    t_all = time.perf_counter()

    with open(res / "phase1" / "log.json") as fh:
        alpha = float(json.load(fh)["ridge"]["alpha"])
    print(f"ridge alpha (fixed in Phase 1): {alpha:g}")

    X = pd.read_parquet(proc / "X_logcpm.parquet")
    with open(proc / "splits.json") as fh:
        splits = json.load(fh)["splits"]

    per_seed: dict = {}
    sel_secs: dict = {m: 0.0 for m in METHODS}
    for s in seeds:
        Ztr, Zte, gene_ids = sel.prepare_split(X, splits[str(s)], cfg["n_genes"])
        n_genes = Ztr.shape[1]
        panels, info, secs = select_panels(Ztr, ks, cfg)
        for m in METHODS:
            sel_secs[m] += secs[m]
        ceiling = ev.pca_reconstruction_r2(Ztr, Zte, ks)

        rec: dict = {"ceiling_r2": ceiling, "random": {}, "methods": {m: {} for m in METHODS}}
        for k in ks:
            rng = np.random.default_rng([s, k, int(cfg["rng_stream_panels"])])
            rp = sel.random_panels(n_genes, k, n_panels, rng)
            sc = [ev.score_panel(Ztr, Zte, p, alpha) for p in rp]
            r2 = np.array([d["r2_all"] for d in sc])
            q2 = np.array([d["q2_nonpanel"] for d in sc])
            rec["random"][k] = {"r2_all_median": np.median(r2), "r2_all_p5": np.percentile(r2, 5),
                                "r2_all_p95": np.percentile(r2, 95),
                                "q2_nonpanel_median": np.median(q2)}
            for m in METHODS:
                p = panels[m][k]
                test = ev.score_panel(Ztr, Zte, p, alpha)
                train = ev.score_panel(Ztr, Ztr, p, alpha)
                rec["methods"][m][k] = {
                    "test_r2_all": test["r2_all"], "test_q2_nonpanel": test["q2_nonpanel"],
                    "train_r2_all": train["r2_all"], "genes": gene_ids[p].tolist(),
                    **info[m].get(k, {}),
                }
        per_seed[s] = rec
        line = ", ".join(f"{m} {rec['methods'][m][gate_k]['test_r2_all']:.4f}" for m in METHODS)
        print(f"seed {s} k={gate_k}: random median {rec['random'][gate_k]['r2_all_median']:.4f}; {line}")

    # --- summaries -------------------------------------------------------------------
    def vec(fn):
        return np.array([fn(per_seed[s]) for s in seeds])

    summary: dict = {}
    for k in ks:
        rand = vec(lambda r: r["random"][k]["r2_all_median"])
        rand_q2 = vec(lambda r: r["random"][k]["q2_nonpanel_median"])
        ceil = vec(lambda r: r["ceiling_r2"][k])
        row = {"random_median": {"mean": rand.mean(), "sd": rand.std(ddof=1)},
               "random_q2_median": {"mean": rand_q2.mean(), "sd": rand_q2.std(ddof=1)},
               "random_p95_mean": vec(lambda r: r["random"][k]["r2_all_p95"]).mean(),
               "ceiling": {"mean": ceil.mean(), "sd": ceil.std(ddof=1)}, "methods": {}}
        for m in METHODS:
            r2 = vec(lambda r: r["methods"][m][k]["test_r2_all"])
            q2 = vec(lambda r: r["methods"][m][k]["test_q2_nonpanel"])
            tr = vec(lambda r: r["methods"][m][k]["train_r2_all"])
            d_mean, d_sd = paired(r2, rand)
            row["methods"][m] = {
                "test_r2_all_mean": r2.mean(), "test_r2_all_sd": r2.std(ddof=1),
                "test_q2_nonpanel_mean": q2.mean(), "test_q2_nonpanel_sd": q2.std(ddof=1),
                "train_r2_all_mean": tr.mean(),
                "vs_random_mean": d_mean, "vs_random_sd": d_sd,
                "gap_closed_frac": float(((r2 - rand) / (ceil - rand)).mean()),
            }
        summary[k] = row

    for k in ks:
        row = summary[k]
        print(f"\nk={k}: random median {row['random_median']['mean']:.4f}±{row['random_median']['sd']:.4f}"
              f"  (random 95th pct {row['random_p95_mean']:.4f}; ceiling {row['ceiling']['mean']:.4f})")
        print("  method     test R2_all      Q2_nonpanel      train R2_all  vs random (paired)   gap closed")
        for m in sorted(METHODS, key=lambda m: -row["methods"][m]["test_r2_all_mean"]):
            r = row["methods"][m]
            print(f"  {m:<9} {r['test_r2_all_mean']:.4f}±{r['test_r2_all_sd']:.4f}  "
                  f"{r['test_q2_nonpanel_mean']:.4f}±{r['test_q2_nonpanel_sd']:.4f}  "
                  f"{r['train_r2_all_mean']:.4f}        {r['vs_random_mean']:+.4f}±{r['vs_random_sd']:.4f}"
                  f"     {100 * r['gap_closed_frac']:5.1f}%")

    # --- gate ------------------------------------------------------------------------
    g = summary[gate_k]
    best = max(METHODS, key=lambda m: g["methods"][m]["test_r2_all_mean"])
    a_mean, a_sd = g["methods"][best]["vs_random_mean"], g["methods"][best]["vs_random_sd"]
    pass_a = a_mean > 0 and a_mean > m_sd * a_sd

    r2_by = {m: vec(lambda r: r["methods"][m][gate_k]["test_r2_all"]) for m in METHODS}
    pairs = []
    for m1, m2 in combinations(METHODS, 2):
        d_mean, d_sd = paired(r2_by[m1], r2_by[m2])
        pairs.append({"a": m1, "b": m2, "mean_diff": d_mean, "sd_diff": d_sd,
                      "differ": abs(d_mean) > m_sd * d_sd})
    n_differ = sum(p["differ"] for p in pairs)
    pass_b = n_differ >= 1
    verdict = "GO" if pass_a and pass_b else "STOP"

    print(f"\nGATE at k={gate_k} (noise = SD of paired per-seed differences, multiple {m_sd:g}):")
    print(f"  A. best method = {best}: {a_mean:+.4f} vs random median, SD {a_sd:.4f} "
          f"(ratio {a_mean / a_sd:.1f}) -> {'pass' if pass_a else 'fail'}")
    print(f"  B. method pairs that differ: {n_differ} of {len(pairs)} -> {'pass' if pass_b else 'fail'}")
    for p in sorted(pairs, key=lambda p: -abs(p["mean_diff"])):
        print(f"     {p['a']:<9} - {p['b']:<9} {p['mean_diff']:+.4f} ± {p['sd_diff']:.4f}"
              f"  {'differ' if p['differ'] else 'within noise'}")
    print(f"  VERDICT: {verdict}")

    plot_path = plots.plot_heuristics(
        ks, {m: [summary[k]["methods"][m]["test_r2_all_mean"] for k in ks] for m in METHODS},
        [summary[k]["random_median"]["mean"] for k in ks],
        [np.mean([per_seed[s]["random"][k]["r2_all_p5"] for s in seeds]) for k in ks],
        [summary[k]["random_p95_mean"] for k in ks],
        [summary[k]["ceiling"]["mean"] for k in ks],
        out_dir / "heuristics.png", gate_k=gate_k,
        title=f"Phase 2: heuristics vs random panels (ridge alpha={alpha:g})",
    )

    log = {
        "phase": "phase2_heuristics",
        "config": {k: cfg[k] for k in cfg},
        "seeds": seeds,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
        "timings_s": {"selection_by_method_total": sel_secs,
                      "total_s": time.perf_counter() - t_all},
        "ridge_alpha": alpha,
        "per_seed": {str(s): per_seed[s] for s in seeds},
        "summary": {str(k): summary[k] for k in ks},
        "gate": {
            "k": gate_k, "noise_definition": "SD across seeds of the paired per-seed difference",
            "multiple_of_sd": m_sd,
            "A_best_method": best, "A_mean_diff_vs_random": a_mean, "A_sd": a_sd, "A_pass": pass_a,
            "B_pairs": pairs, "B_n_differ": n_differ, "B_pass": pass_b,
            "verdict": verdict,
        },
        "outputs": {"plot": str(Path(plot_path).relative_to(ROOT))},
    }
    pio.write_json(log, out_dir / "log.json")
    print(f"wrote {out_dir / 'log.json'} ({time.perf_counter() - t_all:.1f}s)")


if __name__ == "__main__":
    main()
