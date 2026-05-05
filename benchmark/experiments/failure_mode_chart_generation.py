"""
AssetOpsBench — Failure Mode Distribution Chart
================================================
Generates a two-panel publication-quality figure from the clustered
failure mode CSV produced by the benchmark pipeline.

Panel (a): Nested donut chart
    - Inner ring  = failure mode (one colour per mode)
    - Outer ring  = title variants (lightened shade of parent colour)
    - Centre text = total instance count

Panel (b): Horizontal stacked bar chart
    - One bar per failure mode, sorted by frequency (descending)
    - Segments show title variants within each failure mode
    - Count labels annotated inside each segment and as totals at bar end

Requirements:
    pip install pandas matplotlib

Input:
    additional_fm_clustered.csv   (columns: cluster, failure mode, title, description)

Output (written to ./figs/):
    failure_mode_distribution.png
    failure_mode_distribution.pdf

Usage:
    python generate_failure_mode_chart.py
    python generate_failure_mode_chart.py --data path/to/file.csv --out path/to/figs/
"""

import argparse
import os
import warnings

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Generate AssetOpsBench failure mode distribution chart."
)
parser.add_argument(
    "--data",
    default="additional_fm_clustered.csv",
    help="Path to clustered failure mode CSV (default: additional_fm_clustered.csv)",
)
parser.add_argument(
    "--out",
    default="figs",
    help="Output directory for figures (default: ./figs/)",
)
parser.add_argument(
    "--dpi",
    type=int,
    default=180,
    help="DPI for PNG export (default: 180)",
)
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(args.data)

# Validate expected columns
required = {"cluster", "failure mode", "title"}
if not required.issubset(df.columns):
    raise ValueError(
        f"CSV must contain columns: {required}. Found: {set(df.columns)}"
    )

# ═════════════════════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

# Sort failure modes by frequency (descending) for consistent ordering
fm_counts = df["failure mode"].value_counts().sort_values(ascending=False)
fm_list   = fm_counts.index.tolist()

# One distinct colour per failure mode
PALETTE = [
    "#3A7DCC",  # blue
    "#E05C2A",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B2",  # purple
    "#937860",  # brown
    "#DA8BC3",  # pink
    "#17BECF",  # teal
    "#BCBD22",  # yellow-green
    "#E377C2",  # magenta
]

# Assign colours — cycle if more failure modes than palette entries
fm_color = {fm: PALETTE[i % len(PALETTE)] for i, fm in enumerate(fm_list)}

GREY = "#222222"

plt.rcParams.update({
    "font.size":        10,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9.5,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "savefig.facecolor":"white",
})


# ── HELPER: lighten a hex colour ──────────────────────────────────────────────
def lighten(hex_col: str, factor: float = 0.38) -> str:
    """Blend a hex colour toward white by `factor` (0 = no change, 1 = white)."""
    hex_col = hex_col.lstrip("#")
    r, g, b = [int(hex_col[i: i + 2], 16) for i in (0, 2, 4)]
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


# ── HELPER: wrap long strings ─────────────────────────────────────────────────
def wrap(s: str, width: int = 22) -> str:
    """Wrap a string to at most `width` characters per line."""
    words = s.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 <= width:
            current = current + " " + word if current else word
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return "\n".join(lines)


# ── HELPER: save figure ───────────────────────────────────────────────────────
def save_fig(fig: plt.Figure, name: str) -> None:
    base = os.path.join(args.out, name)
    fig.savefig(base + ".png", dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".pdf",              bbox_inches="tight", facecolor="white")
    print(f"  Saved {base}.png / .pdf")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# 3. BUILD FIGURE
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="white")
fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.06, wspace=0.35)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL A — Nested donut
# ─────────────────────────────────────────────────────────────────────────────
ax1 = axes[0]
ax1.set_aspect("equal")

# Inner ring: one wedge per failure mode
inner_vals   = [fm_counts[fm] for fm in fm_list]
inner_colors = [fm_color[fm]  for fm in fm_list]

ax1.pie(
    inner_vals,
    radius=0.55,
    colors=inner_colors,
    startangle=90,
    wedgeprops=dict(width=0.32, edgecolor="white", linewidth=2),
)

# Outer ring: one wedge per title variant, ordered by parent failure mode
outer_data = []
for fm in fm_list:
    for title, cnt in df[df["failure mode"] == fm]["title"].value_counts().items():
        outer_data.append({"fm": fm, "title": title, "count": cnt})

outer_vals         = [d["count"] for d in outer_data]
outer_colors_light = [lighten(fm_color[d["fm"]]) for d in outer_data]

ax1.pie(
    outer_vals,
    radius=0.90,
    colors=outer_colors_light,
    startangle=90,
    wedgeprops=dict(width=0.32, edgecolor="white", linewidth=2),
)

# Centre annotation
ax1.text(
    0, 0,
    f"{sum(inner_vals)}\nfailure\ninstances",
    ha="center", va="center",
    fontsize=11, fontweight="bold",
    color=GREY, linespacing=1.5,
)

# Legend: failure modes only (inner ring)
legend_handles = [
    mpatches.Patch(facecolor=fm_color[fm], label=fm, edgecolor="white")
    for fm in fm_list
]
ax1.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.20),
    ncol=2,
    fontsize=8.5,
    frameon=False,
    handlelength=1.2,
    handletextpad=0.5,
    columnspacing=0.8,
)

# Panel label
ax1.text(
    0.5, 1.02, "(a)",
    transform=ax1.transAxes,
    ha="center", va="bottom",
    fontsize=12, fontweight="bold", color=GREY,
)

# ─────────────────────────────────────────────────────────────────────────────
# PANEL B — Horizontal stacked bar
# ─────────────────────────────────────────────────────────────────────────────
ax2 = axes[1]

# Pre-compute per-failure-mode title counts
bar_data = {
    fm: df[df["failure mode"] == fm]["title"].value_counts()
    for fm in fm_list
}

y_pos = np.arange(len(fm_list))
lefts = np.zeros(len(fm_list))

# One pass per unique title (variant), stacked left-to-right
for title in df["title"].unique():
    vals     = [bar_data[fm].get(title, 0) for fm in fm_list]
    fm_owner = df[df["title"] == title]["failure mode"].iloc[0]
    col      = fm_color[fm_owner]

    ax2.barh(
        y_pos, vals, left=lefts, height=0.55,
        color=col, edgecolor="white", linewidth=0.8,
    )

    # Annotate count inside non-zero segments
    for i, (v, l) in enumerate(zip(vals, lefts)):
        if v > 0:
            ax2.text(
                l + v / 2, y_pos[i], str(v),
                ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold",
            )

    lefts += np.array(vals, dtype=float)

# Total count at the end of each bar
for i, fm in enumerate(fm_list):
    ax2.text(
        fm_counts[fm] + 0.4, y_pos[i], str(fm_counts[fm]),
        va="center", ha="left",
        fontsize=9, color=GREY, fontweight="bold",
    )

# Axis formatting
ax2.set_yticks(y_pos)
ax2.set_yticklabels([wrap(fm) for fm in fm_list], fontsize=9.5, color=GREY)
ax2.set_xlabel("Number of Instances", fontsize=11, color=GREY)
ax2.set_xlim(0, max(fm_counts) * 1.14)
ax2.spines[["top", "right"]].set_visible(False)
ax2.xaxis.grid(True, linestyle="--", alpha=0.4, color="#CCCCCC", zorder=0)
ax2.set_axisbelow(True)
ax2.set_facecolor("white")
ax2.tick_params(axis="y", length=0)

# Panel label
ax2.text(
    0.0, 1.02, "(b)",
    transform=ax2.transAxes,
    ha="left", va="bottom",
    fontsize=12, fontweight="bold", color=GREY,
)

# ═════════════════════════════════════════════════════════════════════════════
# 4. SAVE
# ═════════════════════════════════════════════════════════════════════════════
save_fig(fig, "failure_mode_distribution")
print("Done.")