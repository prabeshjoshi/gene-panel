#!/usr/bin/env python3
"""Phase 0 — low rank: how well do top-q PCs of the training data reconstruct the test data?

GATE: mean test R2_all at q = gate_lowrank_q must be >= gate_lowrank_r2_at_20.
Writes results/phase0/log.json and results/phase0/r2_vs_q.png.
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
    out_dir = ROOT / cfg["results_dir"] / "phase0"
    out_dir.mkdir(parents=True, exist_ok=True)
    qs = [int(q) for q in cfg["lowrank_q"]]
    gate_q = int(cfg["gate_lowrank_q"])
    gate_val = float(cfg["gate_lowrank_r2_at_20"])

    t_all = time.perf_counter()
    X = pd.read_parquet(proc / "X_logcpm.parquet")
    with open(proc / "splits.json") as fh:
        splits = json.load(fh)["splits"]

    per_seed = {}
    for seed in cfg["seeds_gates"]:
        # Gene ranking, centring and scaling are fitted on training samples only.
        Ztr, Zte, _ = sel.prepare_split(X, splits[str(seed)], cfg["n_genes"])
        per_seed[seed] = {
            "n_train": int(Ztr.shape[0]),
            "n_test": int(Zte.shape[0]),
            "n_genes_used": int(Ztr.shape[1]),
            "test_r2": ev.pca_reconstruction_r2(Ztr, Zte, qs),
            "train_r2": ev.pca_reconstruction_r2(Ztr, Ztr, qs),
        }
        print(f"seed {seed}: test R2 at q={gate_q} = {per_seed[seed]['test_r2'][gate_q]:.4f}")

    seeds = list(cfg["seeds_gates"])
    test_mat = np.array([[per_seed[s]["test_r2"][q] for q in qs] for s in seeds])
    train_mat = np.array([[per_seed[s]["train_r2"][q] for q in qs] for s in seeds])
    mean, sd = test_mat.mean(axis=0), test_mat.std(axis=0, ddof=1)

    print("\nq    test R2 mean   SD      train R2 mean")
    for i, q in enumerate(qs):
        print(f"{q:<4} {mean[i]:.4f}        {sd[i]:.4f}   {train_mat.mean(axis=0)[i]:.4f}")

    gate_idx = qs.index(gate_q)
    gate_stat = float(mean[gate_idx])
    verdict = "GO" if gate_stat >= gate_val else "STOP"
    print(f"\nGATE (mean test R2_all at q={gate_q} = {gate_stat:.4f} vs threshold {gate_val}): {verdict}")

    plot_path = plots.plot_r2_vs_q(
        qs, mean, sd, out_dir / "r2_vs_q.png", gate_q=gate_q, gate_value=gate_val,
        title=f"Phase 0: test R2 vs PCs (muscle, {cfg['n_genes']} genes)",
    )

    log = {
        "phase": "phase0_lowrank",
        "config": {k: cfg[k] for k in cfg},
        "seeds": seeds,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
        "timings_s": {"total_s": time.perf_counter() - t_all},
        "per_seed": {str(s): per_seed[s] for s in seeds},
        "summary": {
            "qs": qs,
            "test_r2_mean": dict(zip(map(str, qs), mean.tolist())),
            "test_r2_sd": dict(zip(map(str, qs), sd.tolist())),
        },
        "gate": {
            "name": "lowrank_r2_at_q",
            "q": gate_q,
            "statistic": "mean over seeds of test R2_all",
            "value": gate_stat,
            "threshold": gate_val,
            "verdict": verdict,
        },
        "outputs": {"plot": str(Path(plot_path).relative_to(ROOT))},
    }
    pio.write_json(log, out_dir / "log.json")
    print(f"wrote {out_dir / 'log.json'}")


if __name__ == "__main__":
    main()
