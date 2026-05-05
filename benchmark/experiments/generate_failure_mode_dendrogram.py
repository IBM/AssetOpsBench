"""
AssetOpsBench — Failure Mode Hierarchy Dendrogram
==================================================
Generates a publication-quality hierarchical tree visualisation
of failure modes, styled like a clustering dendrogram.

Hierarchy:
    Root → Failure Mode Cluster → Title Variant (leaf)

Requirements:
    pip install pandas matplotlib numpy

Input:
    additional_fm_clustered.csv
    (columns: cluster, failure mode, title, description)

Output (written to ./figs/):
    failure_mode_dendrogram.png
    failure_mode_dendrogram.pdf

Usage:
    python generate_failure_mode_dendrogram.py
    python generate_failure_mode_dendrogram.py --data path/to/file.csv --out figs/
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
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="additional_fm_clustered.csv",
                    help="Path to clustered failure mode CSV")
parser.add_argument("--out",  default="figs",
                    help="Output directory (default: ./figs/)")
parser.add_argument("--dpi",  type=int, default=180,
                    help="PNG export DPI (default: 180)")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# 1. LOAD & VALIDATE
# ═════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(args.data)
required = {"cluster", "failure mode", "title"}
if not required.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {required}. Found: {set(df.columns)}")

# ═════════════════════════════════════════════════════════════════════════════
# 2. CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

# Stable cluster order
cluster_order = (df[["cluster", "failure mode"]]
                 .drop_duplicates()
                 .sort_values("cluster"))
fm_list = cluster_order["failure mode"].tolist()

# Count instances per node
fm_counts    = df.groupby("failure mode").size().to_dict()
title_counts = df.groupby(["failure mode", "title"]).size().to_dict()

# Leaf title → failure mode lookup
fm_to_titles = {
    fm: df[df["failure mode"] == fm]["title"].unique().tolist()
    for fm in fm_list
}

# One distinct colour per failure mode
PALETTE = [
    "#3A7DCC", "#E05C2A", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#17BECF",
    "#BCBD22", "#E377C2",
]
fm_color = {fm: PALETTE[i % len(PALETTE)] for i, fm in enumerate(fm_list)}


def lighten(hex_col: str, factor: float = 0.40) -> str:
    """Blend a hex colour toward white by `factor`."""
    hex_col = hex_col.lstrip("#")
    r, g, b = int(hex_col[:2], 16), int(hex_col[2:4], 16), int(hex_col[4:], 16)
    return "#{:02x}{:02x}{:02x}".format(
        int(r + (255 - r) * factor),
        int(g + (255 - g) * factor),
        int(b + (255 - b) * factor),
    )


GREY    = "#1A1A2E"
Y_STEP  = 1.2   # vertical spacing between leaf nodes

# X positions of the three hierarchy levels
X_ROOT   = 0.5
X_FM     = 2.0
X_LEAF   = 3.0

# ═════════════════════════════════════════════════════════════════════════════
# 3. COMPUTE LAYOUT
# ═════════════════════════════════════════════════════════════════════════════
# Build leaf list in cluster order
leaves = []
for fm in fm_list:
    for t in fm_to_titles[fm]:
        leaves.append((t, fm))

n_leaves = len(leaves)
leaf_y   = {leaves[i][0]: i * Y_STEP for i in range(n_leaves)}
fm_y     = {fm: np.mean([leaf_y[t] for t in fm_to_titles[fm]])
            for fm in fm_list}
root_y   = np.mean(list(fm_y.values()))

# ═════════════════════════════════════════════════════════════════════════════
# 4. BUILD FIGURE
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(15, 10), facecolor="white")
ax.set_facecolor("white")
fig.subplots_adjust(left=0.01, right=0.58, top=0.93, bottom=0.04)


def draw_elbow(ax, x1, y1, x2, y2, color, lw=1.8, alpha=0.85):
    """Draw an L-shaped connector: horizontal then vertical to target."""
    ax.plot([x1, x2], [y1, y1], color=color, lw=lw, alpha=alpha,
            solid_capstyle="round")
    ax.plot([x2, x2], [y1, y2], color=color, lw=lw, alpha=alpha,
            solid_capstyle="round")


# ── Root vertical spine ───────────────────────────────────────────────────────
y_all = list(fm_y.values())
ax.plot([X_ROOT, X_ROOT], [min(y_all), max(y_all)],
        color="#888888", lw=2.5, alpha=0.6, solid_capstyle="round")

# ── Branches: root → failure mode → title variant ────────────────────────────
for fm in fm_list:
    col  = fm_color[fm]
    fmy  = fm_y[fm]
    cnt  = fm_counts[fm]
    titles = fm_to_titles[fm]

    # Root → FM horizontal arm
    ax.plot([X_ROOT, X_FM], [fmy, fmy],
            color=col, lw=2.2, alpha=0.80, solid_capstyle="round")

    # FM node dot
    ax.plot(X_FM, fmy, "o", color=col, markersize=10, zorder=5,
            markeredgecolor="white", markeredgewidth=1.2)

    # Instance count badge above FM node
    bbox_props = dict(boxstyle="round,pad=0.3",
                      facecolor=lighten(col, 0.55),
                      edgecolor=col, linewidth=1.2)
    ax.text(X_FM - 0.08, fmy + 0.28, f"{cnt} inst.",
            ha="center", va="bottom", fontsize=7.5,
            color=col, fontweight="bold",
            bbox=bbox_props, clip_on=False)

    # Vertical bar connecting leaf arms at FM level (if >1 title)
    if len(titles) > 1:
        leaf_ys = [leaf_y[t] for t in titles]
        ax.plot([X_FM, X_FM], [min(leaf_ys), max(leaf_ys)],
                color=lighten(col, 0.2), lw=1.6, alpha=0.70,
                solid_capstyle="round")

    # FM → each leaf
    for t in titles:
        ty   = leaf_y[t]
        tcnt = title_counts.get((fm, t), 0)
        draw_elbow(ax, X_FM, fmy, X_LEAF, ty,
                   color=lighten(col, 0.15), lw=1.5, alpha=0.75)
        ax.plot(X_LEAF, ty, "o", color=lighten(col, 0.1),
                markersize=7, zorder=5,
                markeredgecolor="white", markeredgewidth=0.8)

# ── Root node ─────────────────────────────────────────────────────────────────
ax.plot(X_ROOT, root_y, "s", color="#555555", markersize=12,
        zorder=6, markeredgecolor="white", markeredgewidth=1.5)
ax.text(X_ROOT - 0.12, root_y, "Root",
        ha="right", va="center", fontsize=9,
        color="#555555", fontweight="bold")

# ── Leaf labels ───────────────────────────────────────────────────────────────
for title, fm in leaves:
    ty   = leaf_y[title]
    tcnt = title_counts.get((fm, title), 0)
    ax.text(X_LEAF + 0.08, ty, f"{title}  ({tcnt})",
            va="center", ha="left", fontsize=9.5,
            color="#222222", clip_on=False)

# ── Right-margin bracket + failure mode label ─────────────────────────────────
X_BRACKET = X_LEAF + 3.8
for fm in fm_list:
    col    = fm_color[fm]
    titles = fm_to_titles[fm]
    ys     = [leaf_y[t] for t in titles]
    y_min, y_max = min(ys), max(ys)
    y_mid  = (y_min + y_max) / 2

    xb = X_BRACKET
    ax.plot([xb, xb], [y_min - 0.1, y_max + 0.1],
            color=col, lw=2.0, alpha=0.80,
            solid_capstyle="round", clip_on=False)
    ax.plot([xb - 0.05, xb], [y_min - 0.1, y_min - 0.1],
            color=col, lw=2.0, alpha=0.80, clip_on=False)
    ax.plot([xb - 0.05, xb], [y_max + 0.1, y_max + 0.1],
            color=col, lw=2.0, alpha=0.80, clip_on=False)

    short = fm if len(fm) <= 24 else fm[:22] + "…"
    ax.text(xb + 0.12, y_mid, short,
            ha="left", va="center", fontsize=9,
            color=col, fontweight="bold", clip_on=False)

# ═════════════════════════════════════════════════════════════════════════════
# 5. AXES STYLE
# ═════════════════════════════════════════════════════════════════════════════
ax.set_xlim(-0.3, X_LEAF + 4.2)
ax.set_ylim(-0.8, (n_leaves - 1) * Y_STEP + 0.8)
ax.set_xlabel("Abstraction Level  (Root → Cluster → Variant)",
              fontsize=10, color="#555555", labelpad=8)
ax.set_xticks([X_ROOT, X_FM, X_LEAF])
ax.set_xticklabels(["Root", "Failure Mode\nCluster", "Title Variant"],
                    fontsize=9, color="#555555")
ax.tick_params(axis="y", left=False, labelleft=False)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color("#CCCCCC")
ax.xaxis.grid(True, linestyle="--", alpha=0.30, color="#DDDDDD")
ax.set_axisbelow(True)
ax.set_title("Failure Mode Hierarchy", fontsize=14,
             fontweight="bold", color=GREY, pad=12, loc="left")

# ── Legend ────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(facecolor=fm_color[fm], label=fm,
                   edgecolor="white", linewidth=0.5)
    for fm in fm_list
]
ax.legend(handles=handles, loc="lower left", fontsize=8.2,
          frameon=True, framealpha=0.92, edgecolor="#CCCCCC",
          handlelength=1.1, ncol=1,
          bbox_to_anchor=(0.0, 0.0))

# ═════════════════════════════════════════════════════════════════════════════
# 6. SAVE
# ═════════════════════════════════════════════════════════════════════════════
base = os.path.join(args.out, "failure_mode_dendrogram")
fig.savefig(base + ".png", dpi=args.dpi, bbox_inches="tight", facecolor="white")
fig.savefig(base + ".pdf",              bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {base}.png / .pdf")
print("Done.")
