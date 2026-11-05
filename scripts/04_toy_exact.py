#!/usr/bin/env python3
"""Phase 3 — toy exact optimum: does greedy miss the true best panel, and does it matter?

For each toy seed and each of toy_n_pools random pools of toy_pool_size genes (targets stay ALL
genes), using training data only:
    f(S) = tr(Z'Z) - tr(inv(G_SS) H_SS)    (OLS residual sum of squares; minimised)
    validated against direct OLS on toy_validate_n random subsets (relative tol toy_validate_rtol)
For each k in toy_k: exact optimum by exhaustive search; greedy (on f), leverage (full-data
top-k right singular vectors, restricted to the pool) and clustering (pool genes, k clusters,
medoids). Record hit/miss and training-objective gap; score every panel on test with the shared
ridge evaluation (alpha fixed in Phase 1).

GATE (each toy seed, cases with k >= gate_toy_k_min):
    A. greedy misses the optimum in >= gate_toy_miss_share of pool x k cases
    B. over those misses, mean test cost R2_all(opt) - R2_all(greedy) > gate_toy_cost_multiple_of_se
       x its bootstrap SE (resampling test samples, toy_n_boot replicates)
GO if A and B hold on every toy seed.

Writes results/phase3/log.json and results/phase3/toy.png.
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

METHODS = ["optimum", "greedy", "leverage", "cluster"]


def validate_objective(Ztr, pool, obj, ks, n, rtol, rng) -> dict:
    """f must equal the direct OLS residual sum of squares on random subsets."""
    worst = 0.0
    for _ in range(n):
        k = int(rng.choice(ks))
        S = np.sort(rng.choice(len(pool), size=k, replace=False))
        P = Ztr[:, pool[S]]
        B, *_ = np.linalg.lstsq(P, Ztr, rcond=None)
        direct = float(((Ztr - P @ B) ** 2).sum())
        worst = max(worst, abs(obj(S) - direct) / direct)
    if worst > rtol:
        raise SystemExit(f"objective validation FAILED: worst relative error {worst:.2e} > {rtol}")
    return {"n_subsets": n, "worst_relative_error": worst, "rtol": rtol, "passed": True}


def row_errors(Ztr, Zte, genes, alpha):
    """Per-test-sample squared error summed over genes, shared ridge evaluation."""
    Zhat = ev.ridge_predict(Ztr, Zte, genes, alpha)
    return ((Zte - Zhat) ** 2).sum(axis=1)


def main() -> None:
    cfg = pio.load_config(ROOT / "config.yaml")
    proc = ROOT / cfg["processed_dir"]
    res = ROOT / cfg["results_dir"]
    out_dir = res / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in cfg["toy_seeds"]]
    ks = [int(k) for k in cfg["toy_k"]]
    pool_size = int(cfg["toy_pool_size"])
    n_pools = int(cfg["toy_n_pools"])
    k_min = int(cfg["gate_toy_k_min"])
    t_all = time.perf_counter()

    with open(res / "phase1" / "log.json") as fh:
        alpha = float(json.load(fh)["ridge"]["alpha"])
    X = pd.read_parquet(proc / "X_logcpm.parquet")
    with open(proc / "splits.json") as fh:
        splits = json.load(fh)["splits"]
    print(f"ridge alpha (fixed in Phase 1): {alpha:g}; pools of {pool_size}, k in {ks}")

    per_seed: dict = {}
    gate_seeds: dict = {}
    for s in seeds:
        t_seed = time.perf_counter()
        Ztr, Zte, gene_ids = sel.prepare_split(X, splits[str(s)], cfg["n_genes"])
        n_genes = Ztr.shape[1]
        _, _, Vt = np.linalg.svd(Ztr, full_matrices=False)
        row_ss = (Zte ** 2).sum(axis=1)
        cases = []
        miss_err = {"opt": [], "greedy": []}
        validation = None

        for p in range(n_pools):
            rng = np.random.default_rng([s, p, int(cfg["rng_stream_toy_pool"])])
            pool = np.sort(rng.choice(n_genes, size=pool_size, replace=False))
            obj = sel.PoolObjective.fit(Ztr, pool)
            if p == 0:
                validation = validate_objective(
                    Ztr, pool, obj, ks, int(cfg["toy_validate_n"]), float(cfg["toy_validate_rtol"]),
                    np.random.default_rng([s, int(cfg["rng_stream_toy_validate"])]))
            g_order = sel.greedy_on_objective(obj, pool_size, max(ks))

            for k in ks:
                opt, f_opt, f_all = sel.exhaustive_optimum(obj, pool_size, k, int(cfg["toy_batch_size"]))
                lev = (Vt[:k] ** 2).sum(axis=0)[pool]
                subsets = {
                    "optimum": opt,
                    "greedy": g_order[:k],
                    "leverage": sel.top_k(lev, k),
                    "cluster": sel.cluster_medoid_panels(Ztr[:, pool], [k], cfg["cluster_linkage"])[k][0],
                }
                case = {"pool": p, "k": k, "n_subsets": len(f_all), "methods": {}}
                for m, sub in subsets.items():
                    f = obj(sub)
                    test = ev.score_panel(Ztr, Zte, pool[sub], alpha)
                    case["methods"][m] = {
                        "hit": bool(set(sub.tolist()) == set(opt.tolist())),
                        "train_r2_ols": obj.r2(f),
                        "train_r2_gap_vs_opt": obj.r2(f_opt) - obj.r2(f),
                        # rank 1 = optimum; ties count as equal
                        "rank": int((f_all < f - 1e-9 * abs(f_opt)).sum()) + 1,
                        "test_r2_all": test["r2_all"], "test_q2_nonpanel": test["q2_nonpanel"],
                        "genes": gene_ids[pool[sub]].tolist(),
                    }
                g = case["methods"]["greedy"]
                g["rank_percentile"] = 100.0 * g["rank"] / len(f_all)
                g["test_cost_vs_opt"] = case["methods"]["optimum"]["test_r2_all"] - g["test_r2_all"]
                if k >= k_min and not g["hit"]:
                    miss_err["opt"].append(row_errors(Ztr, Zte, pool[opt], alpha))
                    miss_err["greedy"].append(row_errors(Ztr, Zte, pool[g_order[:k]], alpha))
                cases.append(case)

        # --- gate for this seed -------------------------------------------------------
        gated = [c for c in cases if c["k"] >= k_min]
        n_miss = sum(not c["methods"]["greedy"]["hit"] for c in gated)
        share = n_miss / len(gated)
        pass_a = share >= float(cfg["gate_toy_miss_share"])
        if n_miss:
            E_opt, E_gr = np.array(miss_err["opt"]), np.array(miss_err["greedy"])
            cost_point = float(((E_gr.sum(1) - E_opt.sum(1)) / row_ss.sum()).mean())
            brng = np.random.default_rng([s, int(cfg["rng_stream_toy_boot"])])
            n_te = Zte.shape[0]
            W = np.stack([np.bincount(brng.integers(0, n_te, n_te), minlength=n_te)
                          for _ in range(int(cfg["toy_n_boot"]))]).astype(float)  # (B, n_te)
            boot_each = ((E_gr - E_opt) @ W.T) / (W @ row_ss)[None, :]           # (misses, B)
            se = float(boot_each.mean(axis=0).std(ddof=1))
            se_each = boot_each.std(axis=1, ddof=1)
            each = (E_gr.sum(1) - E_opt.sum(1)) / row_ss.sum()
            n_sig = int((each > float(cfg["gate_toy_cost_multiple_of_se"]) * se_each).sum())
            n_neg = int((each < 0).sum())
        else:
            cost_point, se, n_sig, n_neg = 0.0, float("nan"), 0, 0
        pass_b = n_miss > 0 and cost_point > float(cfg["gate_toy_cost_multiple_of_se"]) * se
        gate_seeds[s] = {"cases": len(gated), "misses": n_miss, "miss_share": share, "A_pass": pass_a,
                         "mean_test_cost": cost_point, "bootstrap_se": se,
                         "ratio": cost_point / se if n_miss else None,
                         "misses_individually_beyond_noise": n_sig,
                         "misses_where_greedy_better_on_test": n_neg, "B_pass": pass_b}
        per_seed[s] = {"validation": validation, "cases": cases,
                       "seconds": time.perf_counter() - t_seed}
        print(f"seed {s}: objective check worst rel. error {validation['worst_relative_error']:.1e} "
              f"(passed); {time.perf_counter() - t_seed:.0f}s")

    # --- summary tables ----------------------------------------------------------------
    summary: dict = {}
    for s in seeds:
        summary[s] = {}
        for k in ks:
            cs = [c for c in per_seed[s]["cases"] if c["k"] == k]
            row = {"n_pools": len(cs)}
            for m in METHODS[1:]:
                misses = [c for c in cs if not c["methods"][m]["hit"]]
                row[m] = {
                    "hit_rate": 1 - len(misses) / len(cs),
                    "mean_train_r2_gap_when_missed": float(np.mean(
                        [c["methods"][m]["train_r2_gap_vs_opt"] for c in misses])) if misses else 0.0,
                    "mean_test_r2_all": float(np.mean([c["methods"][m]["test_r2_all"] for c in cs])),
                }
            gm = [c for c in cs if not c["methods"]["greedy"]["hit"]]
            row["greedy"]["median_rank_when_missed"] = float(np.median(
                [c["methods"]["greedy"]["rank"] for c in gm])) if gm else 1.0
            row["greedy"]["mean_test_cost_when_missed"] = float(np.mean(
                [c["methods"]["greedy"]["test_cost_vs_opt"] for c in gm])) if gm else 0.0
            row["optimum_mean_test_r2_all"] = float(np.mean(
                [c["methods"]["optimum"]["test_r2_all"] for c in cs]))
            summary[s][k] = row

    for s in seeds:
        print(f"\nseed {s}: hit rate (share of {n_pools} pools where the method finds the exact optimum)")
        print("  k   greedy  leverage  cluster | greedy when missed: train R2 gap  median rank  "
              "test cost | test R2_all opt / greedy / lev / clus")
        for k in ks:
            r = summary[s][k]
            print(f"  {k}   {r['greedy']['hit_rate']:.2f}    {r['leverage']['hit_rate']:.2f}      "
                  f"{r['cluster']['hit_rate']:.2f}    |   {r['greedy']['mean_train_r2_gap_when_missed']:.5f}"
                  f"         {r['greedy']['median_rank_when_missed']:>5.0f}      "
                  f"{r['greedy']['mean_test_cost_when_missed']:+.5f} | "
                  f"{r['optimum_mean_test_r2_all']:.4f} / {r['greedy']['mean_test_r2_all']:.4f} / "
                  f"{r['leverage']['mean_test_r2_all']:.4f} / {r['cluster']['mean_test_r2_all']:.4f}")

    mult = float(cfg["gate_toy_cost_multiple_of_se"])
    print(f"\nGATE (k >= {k_min}; A: miss share >= {cfg['gate_toy_miss_share']}; "
          f"B: mean test cost of misses > {mult:g} x bootstrap SE)")
    for s in seeds:
        g = gate_seeds[s]
        ratio = f"{g['ratio']:.1f}" if g["ratio"] is not None else "n/a"
        print(f"  seed {s}: misses {g['misses']}/{g['cases']} = {g['miss_share']:.2f} -> "
              f"{'pass' if g['A_pass'] else 'fail'};  mean test cost {g['mean_test_cost']:+.5f} "
              f"(SE {g['bootstrap_se']:.5f}, ratio {ratio}) -> {'pass' if g['B_pass'] else 'fail'};  "
              f"misses beyond own noise {g['misses_individually_beyond_noise']}, "
              f"greedy better on test {g['misses_where_greedy_better_on_test']}")
    verdict = "GO" if all(g["A_pass"] and g["B_pass"] for g in gate_seeds.values()) else "STOP"
    print(f"  VERDICT: {verdict}")

    misses = [(c["methods"]["greedy"]["train_r2_gap_vs_opt"], c["methods"]["greedy"]["test_cost_vs_opt"], c["k"])
              for s in seeds for c in per_seed[s]["cases"] if not c["methods"]["greedy"]["hit"]]
    plot_path = plots.plot_toy(
        ks, {s: [summary[s][k]["greedy"]["hit_rate"] for k in ks] for s in seeds},
        {m: [np.mean([summary[s][k][m]["hit_rate"] for s in seeds]) for k in ks]
         for m in ("leverage", "cluster")},
        misses, out_dir / "toy.png",
        title=f"Phase 3: greedy vs exact optimum ({n_pools} pools of {pool_size} genes per seed)",
    )

    log = {
        "phase": "phase3_toy_exact",
        "config": {k: cfg[k] for k in cfg},
        "seeds": seeds,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
        "timings_s": {"per_seed": {str(s): per_seed[s]["seconds"] for s in seeds},
                      "total_s": time.perf_counter() - t_all},
        "ridge_alpha": alpha,
        "objective_validation": {str(s): per_seed[s]["validation"] for s in seeds},
        "summary": {str(s): {str(k): summary[s][k] for k in ks} for s in seeds},
        "gate": {"k_min": k_min, "miss_share_required": cfg["gate_toy_miss_share"],
                 "cost_multiple_of_se": mult, "per_seed": {str(s): gate_seeds[s] for s in seeds},
                 "verdict": verdict},
        "cases": {str(s): per_seed[s]["cases"] for s in seeds},
        "outputs": {"plot": str(Path(plot_path).relative_to(ROOT))},
    }
    pio.write_json(log, out_dir / "log.json")
    print(f"wrote {out_dir / 'log.json'} ({time.perf_counter() - t_all:.1f}s)")


if __name__ == "__main__":
    main()
