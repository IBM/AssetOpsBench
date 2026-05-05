"""Reproduce fig2_kmeans_sweep.pdf and fig2_kmeans_sweep.png.

Self-contained: all K-means sweep numbers are embedded inline, so this
script has no dependency on any CSV files. Style (fonts, palette,
grid, marker scheme, legend placement) is identical to the figure used
in the paper and is intended to be the reference style for the rest
of the paper's figures.

Style summary (reuse as is for other figures):
    - font.family       : DejaVu Serif
    - font.size         : 9  (labels 9, titles 10, legend 7, ticks 8)
    - pdf.fonttype      : 42          (editable in Illustrator)
    - grid              : alpha 0.25, solid, 0.4 linewidth
    - palette           : #ca0020 (red, planning) / #0571b0 (blue, execution)
    - encoder encoding  : solid line = BGE, dashed+open markers = MiniLM
    - figure size       : (6.8, 2.8) inches for a 2-panel horizontal layout
    - DPI               : 300

Usage:
    python reproduce_fig2.py
    # outputs ./figures/fig2_kmeans_sweep.pdf and .png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Data (K-means sweep, K in [2..10], per track, per encoder)
# ------------------------------------------------------------
SWEEP = {
    "planning": {
        "k":             list(range(2, 11)),
        "silhouette_bge":    [0.119, 0.095, 0.109, 0.070, 0.051, 0.070, 0.081, 0.052, 0.069],
        "dbi_bge":           [4.43,  3.84,  3.60,  3.41,  3.32,  3.31,  3.01,  2.87,  2.91],
        "silhouette_minilm": [0.086, 0.078, 0.071, 0.077, 0.084, 0.071, 0.068, 0.059, 0.061],
        "dbi_minilm":        [4.13,  3.66,  3.48,  3.54,  3.23,  3.28,  3.08,  3.12,  3.10],
    },
    "execution": {
        "k":             list(range(2, 11)),
        "silhouette_bge":    [0.145, 0.135, 0.156, 0.155, 0.159, 0.140, 0.138, 0.110, 0.138],
        "dbi_bge":           [3.24,  3.00,  2.62,  3.06,  2.75,  2.77,  2.77,  2.54,  2.60],
        "silhouette_minilm": [0.198, 0.143, 0.146, 0.140, 0.131, 0.129, 0.154, 0.147, 0.123],
        "dbi_minilm":        [2.73,  2.66,  2.61,  2.85,  2.81,  2.63,  2.60,  2.42,  2.35],
    },
}

# ------------------------------------------------------------
# Paper style (reuse this rcParams block for all paper figures)
# ------------------------------------------------------------
PAPER_RC = {
    "font.family":      "DejaVu Serif",
    "font.size":        9,
    "axes.labelsize":   9,
    "axes.titlesize":   10,
    "legend.fontsize":  7,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "-",
    "grid.linewidth":   0.4,
}

# Track palette (colour-blind-safe red/blue pair used throughout the paper)
TRACK_STYLE = {
    "planning":  {"color": "#ca0020", "marker": "o"},
    "execution": {"color": "#0571b0", "marker": "s"},
}


def _plot_one_metric(ax, metric_key_bge, metric_key_mini, ylabel, title):
    """Plot one metric (silhouette or DBI) on `ax` for both tracks, both encoders."""
    for track, style in TRACK_STYLE.items():
        c = style["color"]
        m = style["marker"]
        d = SWEEP[track]

        # BGE = solid line, filled marker
        ax.plot(
            d["k"], d[metric_key_bge],
            linestyle="-", marker=m, color=c,
            markersize=4, linewidth=1.2,
            label=f"{track} / BGE",
        )
        # MiniLM = dashed line, open (white-filled) marker
        ax.plot(
            d["k"], d[metric_key_mini],
            linestyle="--", marker=m,
            color=c, markeredgecolor=c, markerfacecolor="white",
            markersize=4, linewidth=1.2,
            label=f"{track} / MiniLM",
        )

    ax.set_xlabel("K (number of K-means clusters)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(2, 11))
    ax.axvline(5, color="#444444", linestyle=":", linewidth=0.6, alpha=0.8)
    ax.legend(
        loc="upper right", frameon=True, fancybox=False,
        framealpha=0.9, edgecolor="#888888", ncol=2,
    )


def make_figure(out_dir: Path):
    plt.rcParams.update(PAPER_RC)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8), constrained_layout=True)

    _plot_one_metric(
        axes[0],
        metric_key_bge="silhouette_bge",
        metric_key_mini="silhouette_minilm",
        ylabel="Silhouette (cosine)",
        title="(a) Silhouette vs K",
    )
    # annotate the K=5 operating point on the silhouette panel only
    axes[0].text(
        5.1, axes[0].get_ylim()[1] * 0.02,
        "K=5 (operating point)",
        fontsize=7, color="#444444",
    )

    _plot_one_metric(
        axes[1],
        metric_key_bge="dbi_bge",
        metric_key_mini="dbi_minilm",
        ylabel="Davies--Bouldin (lower is better)",
        title="(b) Davies--Bouldin vs K",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out_path = out_dir / f"fig2_kmeans_sweep.{ext}"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    make_figure(Path("figures"))
