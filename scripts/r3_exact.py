#!/usr/bin/env python3
"""Rerun R3: exact-optimum check re-judged against split-to-split SD, plus 1-swap local search.

Arms in r3_arms (B: pools from all 1000 genes, targets all genes; C: pools from the candidate
block, targets = target block). toy_n_pools pools of toy_pool_size per seed, k in toy_k,
OLS objective f(S) = tr(T'T) - tr(inv(G_SS) H_SS) (validated against direct OLS).
Greedy (on f, within the pool) vs the exhaustive optimum; both scored on test with the ridge
evaluation. Gate, cost and noise exactly as pre-registered in config_rerun.yaml.

Local search: from the greedy panel at local_search_k, best-improvement 1-swap on the training
OLS objective; training and test gains.

Writes results_rerun/r3/log.json and results_rerun/r3/costs.csv.
"""
from __future__ import annotations

import argparse
import json
import os

# One BLAS thread per worker process (seeds run in parallel; see r2_compare.py).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from panel import arms  # noqa: E402
from panel import io as pio  # noqa: E402
from panel import rerun as rr  # noqa: E402
from panel import select as sel  # noqa: E402

_G: dict = {}


def _init(rcfg, xpath, gpath, spath):
    X = pd.read_parquet(xpath)
    genes = pd.read_csv(gpath, keep_default_na=False)
    _, included, male = arms.included_samples(X, genes, rcfg)
    with open(spath) as fh:
        splits = json.load(fh)["splits"]
    _G.update(rcfg=rcfg, X=X.loc[included], included=included, male=male, splits=splits)


def run_seed(seed: int) -> dict:
    rc, X, male = _G["rcfg"], _G["X"], _G["male"]
    alpha = float(rc["ridge_alpha"])
    ks = [int(k) for k in rc["toy_k"]]
    split = arms.filter_split(_G["splits"][str(seed)], _G["included"])
    built = arms.build_arms(X, split, male, seed, rc)
    out = {"seed": seed, "toy": {}, "validation": {}, "local": {}}
    for arm in rc["r3_arms"]:
        a = built[arm]
        disjoint = arm == "C"
        cases = []
        for p in range(int(rc["toy_n_pools"])):
            rng = np.random.default_rng([seed, p, int(rc["rng_stream_toy_pool"])])
            pool = a.cand[np.sort(rng.choice(len(a.cand), size=int(rc["toy_pool_size"]), replace=False))]
            obj = sel.PoolObjective.fit(a.Ztr, pool, a.targ)
            if p == 0:   # objective check against direct OLS
                vr = np.random.default_rng([seed, int(rc["rng_stream_toy_validate"])])
                worst = 0.0
                for _ in range(int(rc["toy_validate_n"])):
                    S = np.sort(vr.choice(len(pool), size=int(vr.choice(ks)), replace=False))
                    direct = rr.ols_rss(a.Ztr, pool[S], a.targ)
                    worst = max(worst, abs(obj(S) - direct) / direct)
                if worst > float(rc["toy_validate_rtol"]):
                    raise SystemExit(f"objective validation failed: arm {arm} seed {seed} rel err {worst:.2e}")
                out["validation"][arm] = worst
            g_order = sel.greedy_on_objective(obj, len(pool), max(ks))
            for k in ks:
                opt, f_opt, _ = sel.exhaustive_optimum(obj, len(pool), k, int(rc["toy_batch_size"]))
                g = g_order[:k]
                hit = set(g.tolist()) == set(opt.tolist())
                te_opt = rr.score_on_targets(a.Ztr, a.Zte, pool[opt], a.targ, alpha, disjoint)["r2"]
                te_g = rr.score_on_targets(a.Ztr, a.Zte, pool[g], a.targ, alpha, disjoint)["r2"]
                cases.append({"arm": arm, "seed": seed, "pool": p, "k": k, "hit": hit,
                              "train_r2_opt": obj.r2(f_opt), "train_r2_greedy": obj.r2(obj(g)),
                              "test_r2_opt": te_opt, "test_r2_greedy": te_g,
                              "test_cost": te_opt - te_g})
        out["toy"][arm] = cases

        out["local"][arm] = {}
        order, _ = rr.greedy_targets(a.Ztr, a.cand, a.targ, max(int(k) for k in rc["local_search_k"]))
        for k in rc["local_search_k"]:
            k = int(k)
            start = order[:k]
            panel, info = rr.swap_local_search(a.Ztr, a.cand, a.targ, start, int(rc["local_search_max_iter"]))
            te0 = rr.score_on_targets(a.Ztr, a.Zte, start, a.targ, alpha, disjoint)["r2"]
            te1 = rr.score_on_targets(a.Ztr, a.Zte, panel, a.targ, alpha, disjoint)["r2"]
            out["local"][arm][k] = {
                "iterations": info["iterations"], "hit_max_iter": info["iterations"] >= int(rc["local_search_max_iter"]),
                "train_r2_start": info["train_r2_start"], "train_r2_end": info["train_r2_end"],
                "train_gain": info["train_r2_end"] - info["train_r2_start"],
                "test_r2_start": te0, "test_r2_end": te1, "test_gain": te1 - te0,
                "genes_swapped": int(len(set(start.tolist()) - set(panel.tolist()))),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = ap.parse_args()
    t0 = time.perf_counter()
    rpath = ROOT / "config_rerun.yaml"
    commit = pio.committed_config_hash(ROOT, rpath)
    rc = pio.load_config(rpath)
    pc = pio.load_config(ROOT / rc["pilot_config"])
    proc = ROOT / pc["processed_dir"]
    out_dir = ROOT / rc["results_dir"] / "r3"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in rc["toy_seeds"]]
    ks = [int(k) for k in rc["toy_k"]]
    kmin = int(rc["r3_gate_k_min"])
    thr, mult = float(rc["practical_r2_threshold"]), float(rc["sd_multiple"])
    print(f"config_rerun.yaml commit {commit}; arms {rc['r3_arms']}; seeds {seeds}; workers {args.workers}")

    paths = (str(proc / "X_logcpm.parquet"), str(proc / "genes.csv"), str(proc / "splits.json"))
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(rc, *paths)) as ex:
        for fut in as_completed([ex.submit(run_seed, sd) for sd in seeds]):
            results.append(fut.result())
            print(f"  seed {results[-1]['seed']} done at {time.perf_counter() - t0:.0f}s", flush=True)
    results.sort(key=lambda r: r["seed"])

    cases = pd.DataFrame([c for r in results for arm in r["toy"] for c in r["toy"][arm]])
    cases.to_csv(out_dir / "costs.csv", index=False)
    worst_val = max(v for r in results for v in r["validation"].values())
    print(f"objective validation: worst relative error {worst_val:.1e} over {len(seeds)} seeds x "
          f"{len(rc['r3_arms'])} arms (tolerance {rc['toy_validate_rtol']})")

    per_k: dict = {}
    gate: dict = {}
    for arm in rc["r3_arms"]:
        per_k[arm] = {}
        print(f"\nArm {arm}: greedy vs exact optimum ({rc['toy_n_pools']} pools x {len(seeds)} seeds per k)")
        print("  k  miss rate  mean cost of a miss (SD over seeds)  seeds w/o miss  greedy better on test  "
              "greedy test R2 mean (split SD)")
        for k in ks:
            ck = cases[(cases.arm == arm) & (cases.k == k)]
            misses = ck[~ck.hit]
            seed_cost = misses.groupby("seed")["test_cost"].mean()
            greedy_seed = ck.groupby("seed")["test_r2_greedy"].mean()
            row = {"miss_rate": float(1 - ck.hit.mean()), "n_cases": len(ck), "n_misses": len(misses),
                   "cost_mean": float(seed_cost.mean()) if len(seed_cost) else float("nan"),
                   "cost_sd": float(seed_cost.std(ddof=1)) if len(seed_cost) > 1 else float("nan"),
                   "seeds_without_miss": int(len(seeds) - len(seed_cost)),
                   "share_misses_greedy_better": float((misses.test_cost < 0).mean()) if len(misses) else float("nan"),
                   "greedy_test_r2_mean": float(greedy_seed.mean()),
                   "split_sd": float(greedy_seed.std(ddof=1))}
            per_k[arm][k] = row
            print(f"  {k}  {row['miss_rate']:.2f}       {row['cost_mean']:+.5f} ({row['cost_sd']:.5f})"
                  f"                  {row['seeds_without_miss']}               "
                  f"{row['share_misses_greedy_better']:.2f}                  "
                  f"{row['greedy_test_r2_mean']:.4f} ({row['split_sd']:.5f})")
        checks = {}
        for k in [k for k in ks if k >= kmin]:
            r = per_k[arm][k]
            a_ok = r["cost_mean"] >= thr
            b_ok = r["cost_mean"] >= mult * r["split_sd"]
            checks[k] = {"cost_mean": r["cost_mean"], "practical_threshold": thr, "passes_practical": bool(a_ok),
                         "split_sd": r["split_sd"], "sd_bar": mult * r["split_sd"], "passes_sd": bool(b_ok)}
        verdict = "GO" if all(c["passes_practical"] and c["passes_sd"] for c in checks.values()) else "STOP"
        gate[arm] = {"per_k": checks, "verdict": verdict}

    milp = "continue" if any(g["verdict"] == "GO" for g in gate.values()) else "STOP the MILP direction"
    print(f"\nGATE (pre-registered): at every k >= {kmin}, mean cost >= {thr} AND >= {mult:g} x split-to-split SD")
    for arm, g in gate.items():
        for k, c in g["per_k"].items():
            print(f"  Arm {arm} k={k}: cost {c['cost_mean']:+.5f} vs {thr} -> {'pass' if c['passes_practical'] else 'fail'}; "
                  f"vs {mult:g} x SD = {c['sd_bar']:.5f} -> {'pass' if c['passes_sd'] else 'fail'}")
        print(f"  Arm {arm}: {g['verdict']}")
    print(f"  MILP direction: {milp}")

    local: dict = {}
    print("\nlocal search from the greedy panel (1-swap, training OLS objective); mean ± SD over seeds")
    print("  arm  k   train R2 gain        test R2 gain         iterations  genes swapped  seeds hitting max_iter")
    for arm in rc["local_search_arms"]:
        local[arm] = {}
        for k in rc["local_search_k"]:
            k = int(k)
            v = pd.DataFrame([r["local"][arm][k] for r in results])
            local[arm][k] = {"train_gain_mean": float(v.train_gain.mean()), "train_gain_sd": float(v.train_gain.std(ddof=1)),
                             "test_gain_mean": float(v.test_gain.mean()), "test_gain_sd": float(v.test_gain.std(ddof=1)),
                             "seeds_test_gain_positive": int((v.test_gain > 0).sum()),
                             "iterations_mean": float(v.iterations.mean()), "genes_swapped_mean": float(v.genes_swapped.mean()),
                             "seeds_hit_max_iter": int(v.hit_max_iter.sum()),
                             "per_seed": {str(r["seed"]): r["local"][arm][k] for r in results}}
            L = local[arm][k]
            print(f"  {arm}    {k:<3} {L['train_gain_mean']:+.5f} ± {L['train_gain_sd']:.5f}   "
                  f"{L['test_gain_mean']:+.5f} ± {L['test_gain_sd']:.5f}   {L['iterations_mean']:>5.1f}       "
                  f"{L['genes_swapped_mean']:>5.1f}          {L['seeds_hit_max_iter']}"
                  f"   (test gain > 0 in {L['seeds_test_gain_positive']}/{len(seeds)} seeds)")

    log = {"step": "R3", "config_rerun_commit": commit, "config_rerun": rc,
           "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
           "workers": args.workers, "timings_s": {"total": time.perf_counter() - t0},
           "objective_validation": {str(r["seed"]): r["validation"] for r in results},
           "per_k": per_k, "gate": gate, "milp_direction": milp, "local_search": local,
           "outputs": {"cases_csv": "results_rerun/r3/costs.csv"}}
    pio.write_json(log, out_dir / "log.json")
    print(f"\nwrote {out_dir / 'log.json'} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
