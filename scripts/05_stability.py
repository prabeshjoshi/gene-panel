#!/usr/bin/env python3
"""Phase 4 — stability: does the Phase 2 ranking hold across seeds, and are panels stable?

Repeats Phase 2 at k = stability_k over seeds_stability (same code: select.heuristic_panels,
ridge alpha fixed in Phase 1). Reports per-seed method ranking, the consensus ranking (mean test
R2_all over seeds), Kendall tau and top-1 agreement, and the overlap of panels across seeds:
    Jaccard       |A n B| / |A u B| on gene IDs, mean over seed pairs
    soft overlap  mean over genes of the best |r| to the other panel (symmetrised), on all
                  samples; descriptive only (nothing is fitted with it)
Random panels (same RNG streams as Phases 1-2) give the chance level for both.
Optional: the top pls_eval_top_methods consensus methods re-scored with PLS2 (LVs by inner CV).

GATE: per-seed ranking of the heuristics equals the consensus ranking in more than
gate_stability_share of seeds.

Writes results/phase4/log.json and results/phase4/stability.png.
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
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from panel import evaluate as ev  # noqa: E402
from panel import io as pio  # noqa: E402
from panel import plots  # noqa: E402
from panel import select as sel  # noqa: E402

METHODS = sel.HEURISTICS


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def soft_overlap(a, b, C: pd.DataFrame) -> float:
    """Symmetrised mean best |r| from each panel's genes to the other panel."""
    M = C.loc[list(a), list(b)].to_numpy()
    return 0.5 * (M.max(axis=1).mean() + M.max(axis=0).mean())


def main() -> None:
    cfg = pio.load_config(ROOT / "config.yaml")
    proc = ROOT / cfg["processed_dir"]
    res = ROOT / cfg["results_dir"]
    out_dir = res / "phase4"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in cfg["seeds_stability"]]
    k = int(cfg["stability_k"])
    n_rand = int(cfg["n_random_panels"])
    t_all = time.perf_counter()

    with open(res / "phase1" / "log.json") as fh:
        alpha = float(json.load(fh)["ridge"]["alpha"])
    X = pd.read_parquet(proc / "X_logcpm.parquet")
    with open(proc / "splits.json") as fh:
        splits = json.load(fh)["splits"]
    print(f"ridge alpha (fixed in Phase 1): {alpha:g}; k = {k}; seeds {seeds}")

    per_seed: dict = {}
    for s in seeds:
        Ztr, Zte, gene_ids = sel.prepare_split(X, splits[str(s)], cfg["n_genes"])
        panels, info, _ = sel.heuristic_panels(Ztr, [k], cfg)
        rng = np.random.default_rng([s, k, int(cfg["rng_stream_panels"])])
        rp = sel.random_panels(Ztr.shape[1], k, n_rand, rng)
        r2 = np.array([ev.score_panel(Ztr, Zte, p, alpha)["r2_all"] for p in rp])
        rec = {"random_median": float(np.median(r2)),
               "random_genes": [gene_ids[p].tolist() for p in rp],
               "methods": {}, "_Z": (Ztr, Zte), "_panels": panels}
        for m in METHODS:
            sc = ev.score_panel(Ztr, Zte, panels[m][k], alpha)
            rec["methods"][m] = {"test_r2_all": sc["r2_all"], "test_q2_nonpanel": sc["q2_nonpanel"],
                                 "genes": gene_ids[panels[m][k]].tolist()}
        order = sorted(METHODS, key=lambda m: -rec["methods"][m]["test_r2_all"])
        rec["ranking"] = order
        per_seed[s] = rec
        print(f"seed {s:>2}: " + " > ".join(f"{m} {rec['methods'][m]['test_r2_all']:.3f}" for m in order)
              + f"   (random median {rec['random_median']:.3f})")

    # --- ranking ----------------------------------------------------------------------
    mean_r2 = {m: float(np.mean([per_seed[s]["methods"][m]["test_r2_all"] for s in seeds])) for m in METHODS}
    sd_r2 = {m: float(np.std([per_seed[s]["methods"][m]["test_r2_all"] for s in seeds], ddof=1)) for m in METHODS}
    consensus = sorted(METHODS, key=lambda m: -mean_r2[m])
    cons_score = [mean_r2[m] for m in METHODS]
    rank_stats = {}
    for s in seeds:
        tau = kendalltau([per_seed[s]["methods"][m]["test_r2_all"] for m in METHODS], cons_score)[0]
        rank_stats[s] = {"ranking": per_seed[s]["ranking"],
                         "matches_consensus": per_seed[s]["ranking"] == consensus,
                         "top1_matches": per_seed[s]["ranking"][0] == consensus[0],
                         "kendall_tau": float(tau)}
    share = float(np.mean([r["matches_consensus"] for r in rank_stats.values()]))
    top1 = float(np.mean([r["top1_matches"] for r in rank_stats.values()]))
    taus = np.array([r["kendall_tau"] for r in rank_stats.values()])
    position_counts = {m: {i + 1: sum(per_seed[s]["ranking"].index(m) == i for s in seeds)
                           for i in range(len(METHODS))} for m in METHODS}
    rand_mean = float(np.mean([per_seed[s]["random_median"] for s in seeds]))

    print(f"\nconsensus ranking (mean test R2_all over {len(seeds)} seeds; random median {rand_mean:.3f}):")
    for i, m in enumerate(consensus, 1):
        pos = " ".join(f"{position_counts[m][j]}" for j in range(1, len(METHODS) + 1))
        print(f"  {i}. {m:<9} {mean_r2[m]:.4f} ± {sd_r2[m]:.4f}   seeds at rank 1..5: {pos}")
    swaps = {}
    for s in seeds:
        if not rank_stats[s]["matches_consensus"]:
            r = per_seed[s]["ranking"]
            swaps[s] = [f"{a}<->{b}" for a, b in zip(consensus, r) if a != b]
    print(f"  seeds matching consensus exactly: {share * len(seeds):.0f}/{len(seeds)}; top-1 match "
          f"{top1 * len(seeds):.0f}/{len(seeds)}; Kendall tau mean {taus.mean():.2f} (min {taus.min():.2f})")
    for s, sw in swaps.items():
        print(f"    seed {s}: {' > '.join(per_seed[s]['ranking'])}")

    # --- panel overlap ----------------------------------------------------------------
    t = time.perf_counter()
    n_soft_rand = 10
    union = set()
    for s in seeds:
        for m in METHODS:
            union |= set(per_seed[s]["methods"][m]["genes"])
        for g in per_seed[s]["random_genes"][:n_soft_rand]:
            union |= set(g)
    union = sorted(union)
    C = X[union].corr().abs()
    pairs = list(combinations(seeds, 2))
    overlap: dict = {}
    for m in METHODS:
        J = [jaccard(per_seed[a]["methods"][m]["genes"], per_seed[b]["methods"][m]["genes"]) for a, b in pairs]
        S = [soft_overlap(per_seed[a]["methods"][m]["genes"], per_seed[b]["methods"][m]["genes"], C)
             for a, b in pairs]
        counts = pd.Series([g for s in seeds for g in per_seed[s]["methods"][m]["genes"]]).value_counts()
        overlap[m] = {"jaccard_mean": float(np.mean(J)), "jaccard_min": float(np.min(J)),
                      "jaccard_max": float(np.max(J)), "soft_mean": float(np.mean(S)),
                      "distinct_genes": int(len(counts)),
                      "genes_in_all_seeds": int((counts == len(seeds)).sum()),
                      "genes_in_half_or_more": int((counts >= len(seeds) / 2).sum()),
                      "most_frequent": {g: int(c) for g, c in counts.head(10).items()}}
    Jr = [jaccard(per_seed[a]["random_genes"][i], per_seed[b]["random_genes"][i])
          for a, b in pairs for i in range(n_rand)]
    Sr = [soft_overlap(per_seed[a]["random_genes"][i], per_seed[b]["random_genes"][i], C)
          for a, b in pairs for i in range(n_soft_rand)]
    overlap["random"] = {"jaccard_mean": float(np.mean(Jr)), "soft_mean": float(np.mean(Sr))}
    overlap_secs = time.perf_counter() - t

    symbols = pd.read_csv(proc / "genes.csv", keep_default_na=False).set_index("gene_id")["symbol"]
    print(f"\npanel overlap across {len(pairs)} seed pairs (k = {k}):")
    print("  method     Jaccard mean (min-max)   soft overlap   distinct genes   in all seeds   in >=half")
    for m in METHODS + ["random"]:
        o = overlap[m]
        if m == "random":
            print(f"  {m:<9}  {o['jaccard_mean']:.3f}                    {o['soft_mean']:.3f}")
            continue
        print(f"  {m:<9}  {o['jaccard_mean']:.3f} ({o['jaccard_min']:.2f}-{o['jaccard_max']:.2f})"
              f"         {o['soft_mean']:.3f}          {o['distinct_genes']:>3}              "
              f"{o['genes_in_all_seeds']:>2}             {o['genes_in_half_or_more']:>2}")
    for m in consensus[:2]:
        top = overlap[m]["most_frequent"]
        print(f"  most frequent {m} genes: " + ", ".join(f"{symbols.get(g, g)}({c})" for g, c in top.items()))

    # --- optional: PLS2 instead of ridge for the top consensus methods ------------------
    t = time.perf_counter()
    top_methods = consensus[: int(cfg["pls_eval_top_methods"])]
    pls_res: dict = {m: {} for m in top_methods}
    for s in seeds:
        Ztr, Zte = per_seed[s]["_Z"]
        for m in top_methods:
            p = per_seed[s]["_panels"][m][k]
            n_lv, cv_tab = ev.pls_cv_n_lv(Ztr, p, int(cfg["pls_eval_max_lv"]), int(cfg["ridge_cv_folds"]),
                                          s, int(cfg["pls_max_iter"]))
            sc = ev.score_panel_pls(Ztr, Zte, p, n_lv, int(cfg["pls_max_iter"]))
            pls_res[m][s] = {"n_lv": n_lv, "test_r2_all": sc["r2_all"], "test_q2_nonpanel": sc["q2_nonpanel"],
                             "cv_r2_by_lv": cv_tab}
    pls_secs = time.perf_counter() - t
    a, b = top_methods[:2] if len(top_methods) >= 2 else (top_methods[0], None)
    print(f"\noptional: PLS2 (LVs by {cfg['ridge_cv_folds']}-fold inner CV) instead of ridge, top methods:")
    for m in top_methods:
        r = np.array([pls_res[m][s]["test_r2_all"] for s in seeds])
        lvs = [pls_res[m][s]["n_lv"] for s in seeds]
        print(f"  {m:<9} PLS2 {r.mean():.4f} ± {r.std(ddof=1):.4f} (ridge {mean_r2[m]:.4f});  "
              f"LVs chosen: {min(lvs)}-{max(lvs)}")
    pls_order_holds = None
    if b is not None:
        holds = [pls_res[a][s]["test_r2_all"] > pls_res[b][s]["test_r2_all"] for s in seeds]
        pls_order_holds = int(sum(holds))
        print(f"  {a} > {b} under PLS2 in {pls_order_holds}/{len(seeds)} seeds")

    # --- gate ---------------------------------------------------------------------------
    thr = float(cfg["gate_stability_share"])
    verdict = "GO" if share > thr else "STOP"
    print(f"\nGATE: per-seed ranking equals consensus in {share * len(seeds):.0f}/{len(seeds)} seeds "
          f"= {share:.2f} (need > {thr:g}) -> {verdict}")

    plot_path = plots.plot_stability(
        seeds, {m: [per_seed[s]["methods"][m]["test_r2_all"] for s in seeds] for m in consensus},
        [per_seed[s]["random_median"] for s in seeds],
        {m: overlap[m]["jaccard_mean"] for m in consensus + ["random"]},
        {m: overlap[m]["soft_mean"] for m in consensus + ["random"]},
        out_dir / "stability.png", title=f"Phase 4: stability over {len(seeds)} seeds (k = {k})",
    )

    for s in seeds:
        for key in ("_Z", "_panels", "random_genes"):
            per_seed[s].pop(key)
    log = {
        "phase": "phase4_stability",
        "config": {kk: cfg[kk] for kk in cfg},
        "seeds": seeds,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
        "timings_s": {"overlap_s": overlap_secs, "pls_s": pls_secs, "total_s": time.perf_counter() - t_all},
        "ridge_alpha": alpha,
        "per_seed": {str(s): per_seed[s] for s in seeds},
        "ranking": {"consensus": consensus, "mean_test_r2_all": mean_r2, "sd_test_r2_all": sd_r2,
                    "random_median_mean": rand_mean, "per_seed": {str(s): rank_stats[s] for s in seeds},
                    "position_counts": position_counts, "share_matching_consensus": share,
                    "top1_share": top1, "kendall_tau_mean": float(taus.mean())},
        "overlap": overlap,
        "pls_swap": {"methods": top_methods,
                     "per_seed": {m: {str(s): pls_res[m][s] for s in seeds} for m in top_methods},
                     "order_holds_seeds": pls_order_holds},
        "gate": {"statistic": "share of seeds whose full ranking equals the consensus",
                 "value": share, "threshold_exclusive": thr, "verdict": verdict},
        "outputs": {"plot": str(Path(plot_path).relative_to(ROOT))},
    }
    pio.write_json(log, out_dir / "log.json")
    print(f"wrote {out_dir / 'log.json'} ({time.perf_counter() - t_all:.1f}s)")


if __name__ == "__main__":
    main()
