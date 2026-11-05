"""Plots. Each function writes a PNG and returns its path."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_r2_vs_q(qs, mean, sd, out_path, gate_q=None, gate_value=None, title=""):
    qs = np.asarray(qs)
    mean = np.asarray(mean)
    sd = np.asarray(sd)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(qs, mean, marker="o", color="#1f4e79", label="test R2 (mean over seeds)")
    ax.fill_between(qs, mean - sd, mean + sd, color="#1f4e79", alpha=0.2, label="±1 SD across seeds")
    if gate_q is not None and gate_value is not None:
        ax.axhline(gate_value, color="#b22222", ls="--", lw=1, label=f"gate: {gate_value}")
        ax.axvline(gate_q, color="grey", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xticks(qs)
    ax.set_xticklabels([str(q) for q in qs])
    ax.set_xlabel("number of PCs q")
    ax.set_ylabel("R2_all on test")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, out_path)


def _save(fig, out_path):
    for ax in fig.axes:  # log x-axes: keep only the explicit k / q tick labels
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


# Okabe-Ito colour-blind-safe palette, one colour per heuristic.
METHOD_COLORS = {"greedy": "#D55E00", "leverage": "#0072B2", "cluster": "#009E73",
                 "mtl": "#CC79A7", "pls_vip": "#E69F00"}


def plot_stability(seeds, r2_by_method: dict, rand_median, jaccard: dict, soft: dict, out_path,
                   title=""):
    """Left: test R2_all per seed for each method. Right: panel overlap across seeds."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    x = np.arange(len(seeds))
    a1.plot(x, rand_median, color="#4d4d4d", marker="o", ms=3, ls="--", label="random median")
    for m, vals in r2_by_method.items():
        a1.plot(x, vals, marker="o", ms=4, color=METHOD_COLORS.get(m), label=m)
    a1.set_xticks(x)
    a1.set_xticklabels([str(s) for s in seeds])
    a1.set_xlabel("seed (train/test split)")
    a1.set_ylabel("test R2_all")
    a1.legend(fontsize=8, ncol=3, loc="center", bbox_to_anchor=(0.5, 0.36), frameon=False)
    names = list(jaccard)
    xb = np.arange(len(names))
    a2.bar(xb - 0.2, [jaccard[n] for n in names], width=0.4, color="#1f4e79", label="Jaccard (same genes)")
    a2.bar(xb + 0.2, [soft[n] for n in names], width=0.4, color="#9ecae1",
           label="soft overlap (best |r| stand-in)")
    a2.set_xticks(xb)
    a2.set_xticklabels(names, rotation=30, ha="right")
    a2.set_ylim(0, 1.22)
    a2.set_yticks(np.arange(0, 1.01, 0.2))
    a2.set_ylabel("mean over seed pairs")
    a2.legend(fontsize=8, loc="upper left", ncol=1, frameon=False)
    fig.suptitle(title, fontsize=11)
    return _save(fig, out_path)


def plot_toy(ks, greedy_hit_by_seed: dict, other_hit: dict, misses, out_path, title=""):
    """Left: share of pools where each method finds the exact optimum, by k.
    Right: for every greedy miss, training-R2 gap vs test-R2 cost (optimum minus greedy)."""
    ks = np.asarray(ks)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    shades = ["#D55E00", "#E8875A", "#F2B08F"]
    for i, (s, hits) in enumerate(greedy_hit_by_seed.items()):
        a1.plot(ks, hits, marker="o", color=shades[i % len(shades)], label=f"greedy, seed {s}")
    for m, hits in other_hit.items():
        a1.plot(ks, hits, marker="s", ls="--", color=METHOD_COLORS[m], label=f"{m} (mean of seeds)")
    a1.set_ylim(-0.03, 1.03)
    a1.set_xticks(ks)
    a1.set_xlabel("panel size k")
    a1.set_ylabel("share of pools that hit the exact optimum")
    a1.legend(fontsize=8)
    if misses:
        tr, te, kk = (np.array(v) for v in zip(*misses))
        for k in np.unique(kk):
            sel_ = kk == k
            a2.scatter(tr[sel_], te[sel_], s=18, alpha=0.75, label=f"k = {k}")
        a2.axhline(0, color="grey", lw=1)
        a2.legend(fontsize=8)
    a2.set_xlabel("training R2 lost by greedy (OLS)")
    a2.set_ylabel("test R2_all cost of greedy (optimum − greedy)")
    a2.set_title("greedy misses", fontsize=10)
    fig.suptitle(title, fontsize=11)
    return _save(fig, out_path)


def plot_heuristics(ks, method_means: dict, rand_med, rand_lo, rand_hi, ceiling, out_path,
                    gate_k=None, title=""):
    """Test R2_all vs k for each heuristic, against random panels and the PCA ceiling."""
    ks = np.asarray(ks)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.fill_between(ks, rand_lo, rand_hi, color="#7f7f7f", alpha=0.25,
                    label="random panels, 5–95%")
    ax.plot(ks, rand_med, color="#4d4d4d", marker="o", ms=4, label="random median")
    ax.plot(ks, ceiling, color="#1f4e79", ls="--", lw=1, label="PCA ceiling, q = k")
    for m, vals in method_means.items():
        ax.plot(ks, vals, marker="o", color=METHOD_COLORS.get(m), label=m)
    if gate_k is not None:
        ax.axvline(gate_k, color="grey", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("panel size k")
    ax.set_ylabel("test R2_all (mean over seeds)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, out_path)


def plot_headroom(ks, rand_med, rand_lo, rand_hi, ceil_mean, ceil_sd, q2_med, out_path,
                  gate_k=None, title=""):
    """Random-panel R2_all (median, 5-95% band) vs k, with the PCA ceiling (q = k)."""
    ks = np.asarray(ks)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.fill_between(ks, rand_lo, rand_hi, color="#7f7f7f", alpha=0.25,
                    label="random panels, 5–95% (R2_all)")
    ax.plot(ks, rand_med, marker="o", color="#4d4d4d", label="random median (R2_all)")
    ax.plot(ks, q2_med, marker="s", ms=4, ls="--", color="#4d4d4d", lw=1,
            label="random median (Q2_nonpanel)")
    ceil_mean, ceil_sd = np.asarray(ceil_mean), np.asarray(ceil_sd)
    ax.plot(ks, ceil_mean, marker="o", color="#1f4e79", label="PCA ceiling, q = k")
    ax.fill_between(ks, ceil_mean - ceil_sd, ceil_mean + ceil_sd, color="#1f4e79", alpha=0.2)
    if gate_k is not None:
        ax.axvline(gate_k, color="grey", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("panel size k")
    ax.set_ylabel("test R2 (mean over seeds)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    return _save(fig, out_path)
