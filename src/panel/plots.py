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
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)
