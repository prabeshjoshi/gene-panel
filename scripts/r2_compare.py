#!/usr/bin/env python3
"""Rerun R1 + R2: corrected methods on the three arms, 10 seeds, k in k_values.

For every arm, method, k and seed: test R2 on the arm's targets (Arm C: the target block;
Arms A/B: R2_all and Q2_nonpanel), ridge with the pilot's fixed alpha. Then stability at
stability_k (Jaccard and soft overlap over the 45 seed pairs) and the per-seed ranking of the
pre-registered ranking_methods against the consensus.

Seeds run in parallel worker processes (results do not depend on the number of workers).
Outputs: results_rerun/comparison.csv, results_rerun/r2/arm_{A,B,C}.png, results_rerun/r2/log.json.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from panel import arms  # noqa: E402
from panel import io as pio  # noqa: E402
from panel import plots  # noqa: E402
from panel import rerun as rr  # noqa: E402
from panel import select as sel  # noqa: E402

PRIMARY = ["greedy", "pivoted_qr", "clustering", "multitask_lasso", "pls2_vip_iterative"]
REFERENCE = ["leverage_naive", "pls2_vip_once"]
REPORT_ONLY = ["clustering_average"]
METHODS = PRIMARY + REFERENCE + REPORT_ONLY

_G: dict = {}


def _init(rcfg: dict, xpath: str, gpath: str, spath: str) -> None:
    X = pd.read_parquet(xpath)
    genes = pd.read_csv(gpath, keep_default_na=False)
    _, included, male = arms.included_samples(X, genes, rcfg)
    with open(spath) as fh:
        splits = json.load(fh)["splits"]
    _G.update(rcfg=rcfg, X=X.loc[included], included=included, male=male, splits=splits)


def note_for(arm: str, method: str) -> str:
    if arm in ("A", "B") and method in ("pls2_vip_iterative", "pls2_vip_once"):
        return "degenerate: candidates and targets are the same block"
    if arm in ("A", "B") and method == "multitask_lasso":
        return "limitation: PC targets of the same genes as the candidates"
    return ""


def run_seed(seed: int) -> dict:
    rc, X, male = _G["rcfg"], _G["X"], _G["male"]
    ks = [int(k) for k in rc["k_values"]]
    alpha = float(rc["ridge_alpha"])
    split = arms.filter_split(_G["splits"][str(seed)], _G["included"])
    built = arms.build_arms(X, split, male, seed, rc)
    rows, panels_out, info_out, secs = [], {}, {}, {}
    for arm, a in built.items():
        Ztr, Zte, cand, targ = a.Ztr, a.Zte, a.cand, a.targ
        disjoint = arm == "C"
        panels: dict = {m: {} for m in METHODS}
        info: dict = {m: {} for m in METHODS}
        t = time.perf_counter()

        order, path = rr.greedy_targets(Ztr, cand, targ, max(ks))
        qr_order = rr.pivoted_qr_order(Ztr, cand)
        for k in ks:
            panels["greedy"][k] = order[:k]
            info["greedy"][k] = {"train_ols_r2": float(path[k - 1])}
            panels["pivoted_qr"][k] = qr_order[:k]
            panels["leverage_naive"][k] = cand[sel.leverage_panel(Ztr[:, cand], k)]
        for name, link in (("clustering", rc["cluster_linkage_primary"]),
                           ("clustering_average", rc["cluster_linkage_report_only"])):
            for k, (idx, ncl) in sel.cluster_medoid_panels(Ztr[:, cand], ks, link).items():
                panels[name][k] = cand[idx]
                info[name][k] = {"n_clusters": ncl, "linkage": link}
        T = sel.pc_targets(Ztr[:, targ], int(rc["n_pc_targets"]))
        mtl = sel.mtl_panels(Ztr[:, cand], T, ks, int(rc["mtl_n_alphas"]), float(rc["mtl_alpha_min_ratio"]),
                             int(rc["mtl_max_iter"]), float(rc["mtl_tol"]))
        for k, (idx, meta) in mtl.items():
            panels["multitask_lasso"][k] = cand[idx]
            info["multitask_lasso"][k] = meta
        vip_once = sel.pls_vip(Ztr[:, cand], T, int(rc["pls_vip_once_n_components"]), int(rc["pls_max_iter"]))
        for k in ks:
            panels["pls2_vip_once"][k] = cand[sel.top_k(vip_once, k)]
        t_iter = time.perf_counter()
        it_panels, rounds = rr.pls2_vip_iterative(Ztr, cand, targ, ks, float(rc["pls_iter_drop_fraction"]),
                                                  int(rc["pls_max_lv"]), int(rc["pls_cv_folds"]), seed)
        for k in ks:
            panels["pls2_vip_iterative"][k] = it_panels[k]
            info["pls2_vip_iterative"][k] = {"rounds": len(rounds),
                                             "n_lv_last_round": rounds[-1]["n_lv"],
                                             "n_lv_range": [min(r["n_lv"] for r in rounds),
                                                            max(r["n_lv"] for r in rounds)]}
        secs[arm] = {"total": time.perf_counter() - t, "pls2_vip_iterative": time.perf_counter() - t_iter}

        for m in METHODS:
            for k in ks:
                p = np.asarray(panels[m][k])
                assert len(set(p.tolist())) == k and set(p.tolist()) <= set(cand.tolist()), (arm, m, k)
                te = rr.score_on_targets(Ztr, Zte, p, targ, alpha, disjoint)
                tr = rr.score_on_targets(Ztr, Ztr, p, targ, alpha, disjoint)
                rows.append({"arm": arm, "method": m, "k": k, "seed": seed,
                             "r2_test": te["r2"], "q2_nonpanel_test": te["q2_nonpanel"],
                             "r2_train": tr["r2"], "r2_p5": np.nan, "r2_p95": np.nan,
                             "role": "primary" if m in PRIMARY else ("reference" if m in REFERENCE else "report_only"),
                             "note": note_for(arm, m)})
        lo_p, hi_p = rc["random_band_percentiles"]
        for k in ks:
            rng = np.random.default_rng([seed, k, int(rc["rng_stream_panels"])])
            rp = sel.random_panels(len(cand), k, int(rc["n_random_panels"]), rng)
            sc = [rr.score_on_targets(Ztr, Zte, cand[p], targ, alpha, disjoint) for p in rp]
            r2 = np.array([d["r2"] for d in sc])
            q2 = np.array([d["q2_nonpanel"] for d in sc])
            rows.append({"arm": arm, "method": "random", "k": k, "seed": seed,
                         "r2_test": float(np.median(r2)),
                         "q2_nonpanel_test": float(np.median(q2)) if not disjoint else np.nan,
                         "r2_train": np.nan, "r2_p5": float(np.percentile(r2, lo_p)),
                         "r2_p95": float(np.percentile(r2, hi_p)), "role": "baseline", "note": "median of 100"})
        panels_out[arm] = {m: {k: a.gene_ids[np.asarray(panels[m][k])].tolist() for k in ks} for m in METHODS}
        info_out[arm] = info
    return {"seed": seed, "rows": rows, "panels": panels_out, "info": info_out, "secs": secs}


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def soft_overlap(a, b, C: pd.DataFrame) -> float:
    M = C.loc[list(a), list(b)].to_numpy()
    return 0.5 * (M.max(axis=1).mean() + M.max(axis=0).mean())


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
    out_root = ROOT / rc["results_dir"]
    out_dir = out_root / "r2"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in rc["seeds"]]
    ks = [int(k) for k in rc["k_values"]]
    kstab = int(rc["stability_k"])
    print(f"config_rerun.yaml commit {commit}; seeds {seeds}; k {ks}; workers {args.workers}")

    paths = (str(proc / "X_logcpm.parquet"), str(proc / "genes.csv"), str(proc / "splits.json"))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(rc, *paths)) as ex:
        results = list(ex.map(run_seed, seeds))
    t_run = time.perf_counter() - t0
    for r in results:
        print(f"  seed {r['seed']}: " + ", ".join(f"{a} {v['total']:.0f}s" for a, v in r["secs"].items()))

    df = pd.DataFrame([row for r in results for row in r["rows"]])
    df.to_csv(out_root / "comparison.csv", index=False)
    panels = {r["seed"]: r["panels"] for r in results}

    # --- summaries ------------------------------------------------------------------------
    summ = (df.groupby(["arm", "method", "k"])
              .agg(r2_mean=("r2_test", "mean"), r2_sd=("r2_test", "std"),
                   q2_mean=("q2_nonpanel_test", "mean"), q2_sd=("q2_nonpanel_test", "std"),
                   train_mean=("r2_train", "mean"), p5_mean=("r2_p5", "mean"), p95_mean=("r2_p95", "mean"))
              .reset_index())
    order = ["random"] + METHODS
    for arm in ("A", "B", "C"):
        lab = "target-block R2" if arm == "C" else "R2_all"
        print(f"\nArm {arm} at k={kstab}: test {lab}, mean ± SD over {len(seeds)} seeds"
              + ("" if arm == "C" else "  (Q2_nonpanel)"))
        s = summ[(summ.arm == arm) & (summ.k == kstab)].set_index("method").loc[order]
        for m, r in s.iterrows():
            extra = "" if arm == "C" else f"  ({r.q2_mean:.4f} ± {r.q2_sd:.4f})"
            band = f"  [5–95%: {r.p5_mean:.4f}–{r.p95_mean:.4f}]" if m == "random" else ""
            print(f"  {m:<20} {r.r2_mean:.4f} ± {r.r2_sd:.4f}{extra}{band}  {note_for(arm, m)}")

    # --- stability at stability_k -------------------------------------------------------------
    X = pd.read_parquet(proc / "X_logcpm.parquet")
    genes = pd.read_csv(proc / "genes.csv", keep_default_na=False)
    _, included, male = arms.included_samples(X, genes, rc)
    data_for = {"A": X.loc[included], "B": arms.sex_residualized_all(X, male)}
    data_for["C"] = data_for["B"]
    pairs = list(combinations(seeds, 2))
    stability: dict = {}
    print(f"\nstability at k={kstab} over {len(pairs)} seed pairs: Jaccard / soft overlap")
    for arm in ("A", "B", "C"):
        union = sorted({g for s in seeds for m in METHODS for g in panels[s][arm][m][kstab]})
        C = data_for[arm][union].corr().abs()
        stability[arm] = {}
        for m in METHODS:
            J = [jaccard(panels[a][arm][m][kstab], panels[b][arm][m][kstab]) for a, b in pairs]
            S = [soft_overlap(panels[a][arm][m][kstab], panels[b][arm][m][kstab], C) for a, b in pairs]
            stability[arm][m] = {"jaccard_mean": float(np.mean(J)), "jaccard_sd": float(np.std(J, ddof=1)),
                                 "soft_mean": float(np.mean(S)), "soft_sd": float(np.std(S, ddof=1)),
                                 "distinct_genes": len({g for s in seeds for g in panels[s][arm][m][kstab]})}
        print(f"  Arm {arm}: " + "; ".join(f"{m} {v['jaccard_mean']:.2f}/{v['soft_mean']:.2f}"
                                          for m, v in stability[arm].items()))

    # --- ranking at stability_k ---------------------------------------------------------------
    rmethods = list(rc["ranking_methods"])
    need = int(rc["stability_seeds_required"])
    ranking: dict = {}
    print(f"\nranking at k={kstab} ({', '.join(rmethods)}); holds if matches >= {need} of {len(seeds)}")
    for arm in ("A", "B", "C"):
        sub = df[(df.arm == arm) & (df.k == kstab) & (df.method.isin(rmethods))]
        means = sub.groupby("method")["r2_test"].mean()
        consensus = sorted(rmethods, key=lambda m: -means[m])
        per_seed = {}
        for s in seeds:
            v = sub[sub.seed == s].set_index("method")["r2_test"]
            per_seed[str(s)] = sorted(rmethods, key=lambda m: -v[m])
        matches = sum(per_seed[str(s)] == consensus for s in seeds)
        ranking[arm] = {"consensus": consensus, "consensus_mean_r2": means.to_dict(), "per_seed": per_seed,
                        "matches": matches, "required": need, "holds": matches >= need}
        print(f"  Arm {arm}: consensus {' > '.join(consensus)}; matches {matches}/{len(seeds)} -> "
              f"{'holds' if matches >= need else 'does not hold'}")
        for s in seeds:
            if per_seed[str(s)] != consensus:
                print(f"     seed {s}: {' > '.join(per_seed[str(s)])}")

    # --- figures ----------------------------------------------------------------------------
    figs = {}
    for arm in ("A", "B", "C"):
        s = summ[summ.arm == arm]
        rnd = s[s.method == "random"].set_index("k").loc[ks]
        means = {m: s[s.method == m].set_index("k").loc[ks, "r2_mean"].tolist() for m in METHODS}
        lab = "test R2 on target block" if arm == "C" else "test R2_all"
        figs[arm] = str(Path(plots.plot_arm(
            ks, means, rnd["r2_mean"].tolist(), rnd["p5_mean"].tolist(), rnd["p95_mean"].tolist(),
            out_dir / f"arm_{arm}.png", f"{lab} (mean of {len(seeds)} seeds)",
            title=f"Arm {arm}: test R2 vs panel size")).relative_to(ROOT))

    log = {
        "step": "R1+R2", "config_rerun_commit": commit, "config_rerun": rc,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "workers": args.workers,
        "timings_s": {"seeds_parallel": t_run, "total": time.perf_counter() - t0,
                      "per_seed": {str(r["seed"]): r["secs"] for r in results}},
        "methods": {"primary": PRIMARY, "reference": REFERENCE, "report_only": REPORT_ONLY},
        "summary": summ.to_dict(orient="records"),
        "stability": stability, "ranking": ranking,
        "panels": {str(s): panels[s] for s in seeds},
        "method_info": {str(r["seed"]): r["info"] for r in results},
        "outputs": {"comparison_csv": "results_rerun/comparison.csv", "figures": figs},
    }
    pio.write_json(log, out_dir / "log.json")
    print(f"\nwrote results_rerun/comparison.csv and {out_dir / 'log.json'} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
