import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"        : "serif",
    "font.serif"         : ["Georgia", "Times New Roman", "DejaVu Serif"],
    "font.size"          : 11,
    "axes.spines.top"    : False,
    "axes.spines.right"  : False,
    "axes.spines.left"   : True,
    "axes.spines.bottom" : True,
    "axes.linewidth"     : 0.8,
    "axes.grid"          : True,
    "grid.color"         : "#e8e8e8",
    "grid.linewidth"     : 0.6,
    "grid.linestyle"     : "--",
    "xtick.major.width"  : 0.8,
    "ytick.major.width"  : 0.8,
    "xtick.minor.visible": False,
    "figure.dpi"         : 300,
    "savefig.dpi"        : 300,
    "savefig.bbox"       : "tight",
    "savefig.facecolor"  : "white",
})

# ── Palette ───────────────────────────────────────────────────────────────────
C_PIPE     = "#2563EB"   # blue  — full pipeline
C_PIPE_L   = "#BFDBFE"   # light fill
C_BASE     = "#6B7280"   # gray  — baseline
C_BASE_L   = "#E5E7EB"   # light fill
C_TARGET   = "#059669"   # green — target line
C_ANNOT    = "#1E3A5F"   # dark  — annotations

ks               = list(range(1, 11))
baseline_recalls = [57.11, 72.30, 78.92, 82.84, 85.54,
                    86.76, 88.24, 88.73, 89.22, 89.95]
pipeline_recalls = [63.24, 78.19, 84.80, 88.48, 92.89,
                    94.61, 95.10, 96.08, 97.06, 97.06]
target           = 99.0


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — Recall@k curve
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.5, 4.2))

# Shaded fill — pipeline to target gap
ax.fill_between(ks, pipeline_recalls, target,
                alpha=0.12, color=C_TARGET, zorder=1)

# Shaded fill under each curve (area)
ax.fill_between(ks, baseline_recalls, alpha=0.10,
                color=C_BASE, zorder=2)
ax.fill_between(ks, pipeline_recalls, alpha=0.12,
                color=C_PIPE, zorder=2)

# Baseline curve
ax.plot(ks, baseline_recalls,
        color=C_BASE, linewidth=1.8, linestyle="--",
        marker="s", markersize=5.5,
        markerfacecolor="white", markeredgewidth=1.6,
        markeredgecolor=C_BASE,
        label="Baseline (bi-encoder only)", zorder=4)

# Full pipeline curve
ax.plot(ks, pipeline_recalls,
        color=C_PIPE, linewidth=2.2,
        marker="o", markersize=6,
        markerfacecolor="white", markeredgewidth=1.8,
        markeredgecolor=C_PIPE,
        label="Full pipeline (BM25 + bi-encoder + cross-encoder)", zorder=5)

# Target line
ax.axhline(target, color=C_TARGET, linewidth=1.4,
           linestyle=":", zorder=3, label="Target (99%)")

# ── Annotate key points ───────────────────────────────────────────────────────
annot_pipe  = {1: (0, 10), 5: (0, 10), 9: (-8, 10)}
annot_base  = {1: (0, -18), 5: (0, -18), 10: (0, -18)}

for k, r in zip(ks, pipeline_recalls):
    if k in annot_pipe:
        dx, dy = annot_pipe[k]
        ax.annotate(f"{r:.1f}%",
                    xy=(k, r), xytext=(dx, dy),
                    textcoords="offset points",
                    ha="center", fontsize=8.5,
                    color=C_PIPE, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=C_PIPE,
                                   lw=0.6, alpha=0.5))

for k, r in zip(ks, baseline_recalls):
    if k in annot_base:
        dx, dy = annot_base[k]
        ax.annotate(f"{r:.1f}%",
                    xy=(k, r), xytext=(dx, dy),
                    textcoords="offset points",
                    ha="center", fontsize=8.5,
                    color=C_BASE,
                    arrowprops=dict(arrowstyle="-", color=C_BASE,
                                   lw=0.6, alpha=0.5))

# ── Delta annotations (gain arrows between curves at k=1, k=5, k=9) ──────────
for k, gain_label in [(1, "+6.1pp"), (5, "+7.4pp"), (9, "+7.8pp")]:
    b = baseline_recalls[k - 1]
    p = pipeline_recalls[k - 1]
    mid = (b + p) / 2
    ax.annotate("", xy=(k + 0.15, p), xytext=(k + 0.15, b),
                arrowprops=dict(arrowstyle="<->", color="#9CA3AF",
                                lw=1.0))
    ax.text(k + 0.32, mid, gain_label,
            ha="left", va="center", fontsize=7.5,
            color="#6B7280", style="italic")

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_xlim(0.5, 10.8)
ax.set_ylim(45, 103)
ax.set_xticks(ks)
ax.set_xlabel("Retrieval depth  $k$", fontsize=11, labelpad=6)
ax.set_ylabel("Recall@$k$  (%)", fontsize=11, labelpad=6)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: f"{int(v)}%"))

ax.set_title("Recall@$k$ — Diagnostic Step Alignment Benchmark",
             fontsize=12, fontweight="bold", pad=12, loc="left")
ax.text(0, 1.01,
        "Full pipeline consistently outperforms baseline by 6–8pp across all $k$",
        transform=ax.transAxes, fontsize=9,
        color="#6B7280", style="italic")

# Legend
leg = ax.legend(frameon=True, fontsize=9, loc="lower right",
                framealpha=0.95, edgecolor="#E5E7EB",
                borderpad=0.8, handlelength=2.2)
leg.get_frame().set_linewidth(0.6)

plt.tight_layout()
plt.savefig("recall_curve.pdf")
plt.savefig("recall_curve.png")
print("Saved recall_curve.pdf / .png")
plt.show()


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — Grouped bar chart
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 4.0))

metrics       = ["Recall@1", "Recall@5", "Recall@10"]
base_vals     = [57.11, 85.54, 89.95]
pipeline_vals = [63.24, 92.89, 97.06]
deltas        = ["+6.1pp", "+7.4pp", "+7.1pp"]

x     = np.arange(len(metrics))
width = 0.32
gap   = 0.04

bars_b = ax.bar(x - width/2 - gap/2, base_vals, width,
                label="Baseline", color=C_BASE_L,
                edgecolor=C_BASE, linewidth=0.9, zorder=3)
bars_p = ax.bar(x + width/2 + gap/2, pipeline_vals, width,
                label="Full pipeline", color=C_PIPE_L,
                edgecolor=C_PIPE, linewidth=0.9, zorder=3)

# Target line
ax.axhline(target, color=C_TARGET, linewidth=1.2,
           linestyle=":", zorder=2, label="Target (99%)")

# Value labels above bars
for bar, val in zip(bars_b, base_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + 0.6, f"{val:.1f}%",
            ha="center", va="bottom",
            fontsize=8.5, color=C_BASE, fontweight="bold")

for bar, val in zip(bars_p, pipeline_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            val + 0.6, f"{val:.1f}%",
            ha="center", va="bottom",
            fontsize=8.5, color=C_PIPE, fontweight="bold")

# Delta brackets between bar pairs
for i, (bv, pv, delta) in enumerate(zip(base_vals, pipeline_vals, deltas)):
    bx = x[i] - width/2 - gap/2 + width/2
    px = x[i] + width/2 + gap/2 + width/2
    mid_x = (bx + px) / 2
    top_y  = max(bv, pv) + 4.5
    ax.annotate("", xy=(px - width, top_y),
                xytext=(bx - width + 0.02, top_y),
                arrowprops=dict(arrowstyle="-", color="#9CA3AF", lw=0.8))
    ax.text(mid_x - width + 0.02, top_y + 0.6,
            delta, ha="center", va="bottom",
            fontsize=8, color="#374151",
            style="italic", fontweight="bold")

# Grid only on y
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#e8e8e8")
ax.xaxis.grid(False)

# Axes
ax.set_ylim(40, 106)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylabel("Recall  (%)", fontsize=11, labelpad=6)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: f"{int(v)}%"))

ax.set_title("Method Comparison: Baseline vs Full Pipeline",
             fontsize=12, fontweight="bold", pad=12, loc="left")
ax.text(0, 1.01,
        "Consistent gains of 6–8pp across all retrieval depths",
        transform=ax.transAxes, fontsize=9,
        color="#6B7280", style="italic")

leg = ax.legend(frameon=True, fontsize=9, loc="lower right",
                framealpha=0.95, edgecolor="#E5E7EB",
                borderpad=0.8)
leg.get_frame().set_linewidth(0.6)

plt.tight_layout()
plt.savefig("method_comparison.pdf")
plt.savefig("method_comparison.png")
print("Saved method_comparison.pdf / .png")
plt.show()