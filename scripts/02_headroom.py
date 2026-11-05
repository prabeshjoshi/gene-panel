#!/usr/bin/env python3
"""Phase 1 — headroom: how far are random k-gene panels below the low-rank ceiling?

For each k in k_random, n_random_panels random panels per seed; ridge from panel genes to all
genes (alpha chosen once by CV on the first seed's training data); R2_all and Q2_nonpanel on test.
Ceiling: PCA reconstruction with q = k (Phase 0), same splits.

GATE at k = gate_headroom_k:
    headroom = mean over seeds of (ceiling - random median R2_all)
    noise    = SD across seeds of the random median R2_all
    GO if headroom >= gate_headroom_multiple_of_sd * noise.
The SD of the per-seed gap is reported as a second reading of "noise".

Writes results/phase1/log.json and results/phase1/headroom.png.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from panel import evaluate as ev  # noqa: E402
from panel import io as pio  # noqa: E402
from panel import plots  # noqa: E402
from panel import select as sel  # noqa: E402


def main() -> None:
    cfg = pio.load_config(ROOT / "config.yaml")
    proc = ROOT / cfg["processed_dir"]
    out_dir = ROOT / cfg["results_dir"] / "phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in cfg["seeds_gates"]]
    ks = [int(k) for k in cfg["k_random"]]
    n_panels = int(cfg["n_random_panels"])
    gate_k = int(cfg["gate_headroom_k"])
    mult = float(cfg["gate_headroom_multiple_of_sd"])
    timings: dict[str, float] = {}
    t_all = time.perf_counter()

    X = pd.read_parquet(proc / "X_logcpm.parquet")
    with open(proc / "splits.json") as fh:
        splits = json.load(fh)["splits"]
    prepared = {s: sel.prepare_split(X, splits[str(s)], cfg["n_genes"]) for s in seeds}
    n_genes = prepared[seeds[0]][0].shape[1]

    # --- choose ridge alpha once: CV on training data of the first seed -------------
    t = time.perf_counter()
    seed0 = seeds[0]
    Ztr0 = prepared[seed0][0]
    cv_panels = []
    for k in ks:
        rng = np.random.default_rng([seed0, k, int(cfg["rng_stream_alpha_cv"])])
        cv_panels.extend(sel.random_panels(n_genes, k, int(cfg["ridge_cv_panels_per_k"]), rng))
    alpha, alpha_table = ev.cv_ridge_alpha(Ztr0, cv_panels, cfg["ridge_alphas"],
                                           int(cfg["ridge_cv_folds"]), seed0)
    timings["alpha_cv_s"] = time.perf_counter() - t
    print(f"ridge alpha (CV on seed {seed0} train): {alpha}")
    for a, sc in alpha_table.items():
        print(f"   alpha={a:<8g} CV R2_all={sc:.4f}")
    if alpha in (min(cfg["ridge_alphas"]), max(cfg["ridge_alphas"])):
        print("   WARNING: chosen alpha is at the edge of the grid")

    # --- random panels -----------------------------------------------------------------
    t = time.perf_counter()
    per_seed: dict = {}
    for s in seeds:
        Ztr, Zte, _ = prepared[s]
        ceiling = ev.pca_reconstruction_r2(Ztr, Zte, ks)
        per_seed[s] = {"ceiling_r2": ceiling, "k": {}}
        for k in ks:
            rng = np.random.default_rng([s, k, int(cfg["rng_stream_panels"])])
            panels = sel.random_panels(n_genes, k, n_panels, rng)
            sc = [ev.score_panel(Ztr, Zte, p, alpha) for p in panels]
            r2 = np.array([d["r2_all"] for d in sc])
            q2 = np.array([d["q2_nonpanel"] for d in sc])
            per_seed[s]["k"][k] = {
                "r2_all": {"median": np.median(r2), "p5": np.percentile(r2, 5),
                           "p95": np.percentile(r2, 95), "max": r2.max(), "min": r2.min()},
                "q2_nonpanel": {"median": np.median(q2), "p5": np.percentile(q2, 5),
                                "p95": np.percentile(q2, 95), "max": q2.max()},
            }
        print(f"seed {s}: random median R2_all at k={gate_k} = "
              f"{per_seed[s]['k'][gate_k]['r2_all']['median']:.4f}, "
              f"ceiling = {ceiling[gate_k]:.4f}")
    timings["random_panels_s"] = time.perf_counter() - t

    # --- summaries across seeds -------------------------------------------------------
    def across(fn):
        v = np.array([[fn(s, k) for k in ks] for s in seeds])
        return v.mean(axis=0), v.std(axis=0, ddof=1)

    med_m, med_sd = across(lambda s, k: per_seed[s]["k"][k]["r2_all"]["median"])
    p5_m, _ = across(lambda s, k: per_seed[s]["k"][k]["r2_all"]["p5"])
    p95_m, _ = across(lambda s, k: per_seed[s]["k"][k]["r2_all"]["p95"])
    max_m, _ = across(lambda s, k: per_seed[s]["k"][k]["r2_all"]["max"])
    q2_m, q2_sd = across(lambda s, k: per_seed[s]["k"][k]["q2_nonpanel"]["median"])
    ceil_m, ceil_sd = across(lambda s, k: per_seed[s]["ceiling_r2"][k])
    gap_m, gap_sd = across(lambda s, k: per_seed[s]["ceiling_r2"][k]
                           - per_seed[s]["k"][k]["r2_all"]["median"])

    print("\nk    ceiling        random R2_all median (5%-95%)   best random   "
          "Q2_nonpanel median   gap (ceiling - median)")
    for i, k in enumerate(ks):
        print(f"{k:<4} {ceil_m[i]:.4f}±{ceil_sd[i]:.4f}  {med_m[i]:.4f}±{med_sd[i]:.4f} "
              f"({p5_m[i]:.4f}-{p95_m[i]:.4f})   {max_m[i]:.4f}        "
              f"{q2_m[i]:.4f}±{q2_sd[i]:.4f}      {gap_m[i]:.4f}±{gap_sd[i]:.4f}")

    gi = ks.index(gate_k)
    headroom = float(gap_m[gi])
    noise = float(med_sd[gi])
    verdict = "GO" if headroom >= mult * noise else "STOP"
    verdict_gap_sd = "GO" if headroom >= mult * float(gap_sd[gi]) else "STOP"
    print(f"\nGATE at k={gate_k}: headroom = {headroom:.4f}; "
          f"SD of random median across seeds = {noise:.4f}; "
          f"headroom / SD = {headroom / noise:.1f} (need >= {mult:g}) -> {verdict}")
    print(f"   second reading, SD of per-seed gap = {gap_sd[gi]:.4f}; "
          f"ratio = {headroom / gap_sd[gi]:.1f} -> {verdict_gap_sd}")

    plot_path = plots.plot_headroom(
        ks, med_m, p5_m, p95_m, ceil_m, ceil_sd, q2_m, out_dir / "headroom.png", gate_k=gate_k,
        title=f"Phase 1: random panels vs PCA ceiling (ridge alpha={alpha:g})",
    )

    timings["total_s"] = time.perf_counter() - t_all
    summary = {
        str(k): {
            "ceiling_mean": ceil_m[i], "ceiling_sd": ceil_sd[i],
            "random_r2_median_mean": med_m[i], "random_r2_median_sd": med_sd[i],
            "random_r2_p5_mean": p5_m[i], "random_r2_p95_mean": p95_m[i],
            "random_r2_max_mean": max_m[i],
            "random_q2_nonpanel_median_mean": q2_m[i], "random_q2_nonpanel_median_sd": q2_sd[i],
            "gap_mean": gap_m[i], "gap_sd": gap_sd[i],
        }
        for i, k in enumerate(ks)
    }
    log = {
        "phase": "phase1_headroom",
        "config": {k: cfg[k] for k in cfg},
        "seeds": seeds,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
        "timings_s": timings,
        "ridge": {"alpha": alpha, "chosen_on_seed": seed0, "cv_folds": cfg["ridge_cv_folds"],
                  "cv_panels": len(cv_panels), "cv_r2_all_by_alpha": alpha_table},
        "per_seed": {str(s): per_seed[s] for s in seeds},
        "summary": summary,
        "gate": {
            "name": "headroom_at_k",
            "k": gate_k,
            "headroom": headroom,
            "noise_sd": noise,
            "noise_definition": "SD across seeds of the random-panel median R2_all",
            "ratio": headroom / noise,
            "multiple_required": mult,
            "verdict": verdict,
            "alt_noise_sd_of_gap": float(gap_sd[gi]),
            "alt_verdict": verdict_gap_sd,
        },
        "outputs": {"plot": str(Path(plot_path).relative_to(ROOT))},
    }
    pio.write_json(log, out_dir / "log.json")
    print(f"wrote {out_dir / 'log.json'} ({timings['total_s']:.1f}s)")


if __name__ == "__main__":
    main()
