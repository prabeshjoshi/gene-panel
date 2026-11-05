"""Plots. Each function writes a PNG and returns its path."""
from __future__ import annotations

from pathlib import Path

import matplotlib
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
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


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
