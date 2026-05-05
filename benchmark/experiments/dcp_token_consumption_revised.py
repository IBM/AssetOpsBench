"""generate_cost_figures.py
Produces all execution-cost figures for the NeurIPS evaluation paper.

Style follows reproduce_fig2.py exactly (DejaVu Serif, 9 pt base,
grid alpha 0.25, pdf.fonttype 42).  The only additions from the cost
module are DPI / facecolor / bbox save-fig settings and the spine
suppression that was already in the original cost script.

Input:  trajectory_stats.csv
Output: figures/fig_cost_*.pdf  and  figures/fig_cost_*.png

Usage:
    python generate_cost_figures.py
    python generate_cost_figures.py --data path/to/trajectory_stats.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="trajectory_stats.csv")
parser.add_argument("--out",  default="figures")
args = parser.parse_args()

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

# ── Paper style (identical to reproduce_fig2.py) ───────────────────────────────
# Reuse this block unchanged across all paper figures.
PAPER_RC: dict = {
    # typography
    "font.family":      "DejaVu Serif",
    "font.size":        9,
    "axes.labelsize":   9,
    "axes.titlesize":   10,
    "legend.fontsize":  7,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    # PDF embedding (editable in Illustrator / Inkscape)
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    # grid
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "-",
    "grid.linewidth":   0.4,
    # save settings (additions over fig2 for consistent export)
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.facecolor": "white",
    "savefig.bbox":     "tight",
    # spine cleanup
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.linewidth":   0.8,
}

plt.rcParams.update(PAPER_RC)

# ── Colour palettes ─────────────────────────────────────────────────────────────
AGENT_COLORS: dict[str, str] = {
    "IoT":  "#2563EB",
    "FMSA": "#7C3AED",
    "TSFM": "#059669",
    "WO":   "#D97706",
    "E2E":  "#DC2626",
}
CLASS_COLORS: dict[str, str] = {
    "Single-Agent": "#2563EB",
    "Multi-Agent":  "#DC2626",
}
PHASE_COLORS: dict[str, str] = {
    "P1": "#1D4ED8",
    "P2": "#B45309",
}
PHASE_LABELS: dict[str, str] = {
    "P1": "Phase 1\n(Development)",
    "P2": "Phase 2\n(Evaluation)",
}

# Reusable legend handles
LEGEND_AGENT = [
    mpatches.Patch(color=v, label=k, alpha=0.85)
    for k, v in AGENT_COLORS.items()
]
LEGEND_CLASS = [
    mpatches.Patch(color=v, label=k, alpha=0.85)
    for k, v in CLASS_COLORS.items()
]

# Scenario ID ranges
PHASE1_IDS = [5, 8, 106, 114, 203, 204, 400, 405, 424, 604, 607]
PHASE2_IDS = [7, 11, 107, 108, 201, 205, 403, 410, 411, 605, 606]


# =============================================================================
# DATA
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Correct for request/response double-counting
    df["api_calls"] = df["api_calls"] / 2

    # Derive phase and agent domain from scenario ID
    df["phase"] = df["id"].apply(
        lambda x: "P1" if x in PHASE1_IDS else "P2"
    )

    def _agent(sid: int) -> str:
        if sid in [5, 7, 8, 11]:
            return "IoT"
        if 100 <= sid <= 199:
            return "FMSA"
        if 200 <= sid <= 299:
            return "TSFM"
        if 400 <= sid <= 499:
            return "WO"
        if 600 <= sid <= 699:
            return "E2E"
        return "Unknown"

    df["agent"]       = df["id"].map(_agent)
    df["agent_class"] = df["agent"].map(
        lambda a: "Multi-Agent" if a == "E2E" else "Single-Agent"
    )
    return df


# =============================================================================
# SHARED HELPERS
# =============================================================================

def savefig(name: str) -> None:
    for ext in ("pdf", "png"):
        p = OUT / f"{name}.{ext}"
        plt.savefig(p)
        print(f"  wrote {p}")
    plt.close()


def bar_panel(
    ax: plt.Axes,
    labels: list[str],
    vals: np.ndarray,
    errs: np.ndarray,
    colors: list[str],
    ylabel: str,
    title: str,
    pval: float | None = None,
    rotation: int = 0,
) -> None:
    """Horizontal grouped bar with error bars and optional p-value annotation."""
    x    = np.arange(len(labels))
    bars = ax.bar(
        x, vals,
        color=colors, alpha=0.85,
        edgecolor="white", linewidth=0.5, width=0.55,
    )
    ax.errorbar(
        x, vals, yerr=errs,
        fmt="none", color="#374151",
        linewidth=1.0, capsize=4,
    )
    # Value labels above bars
    err_max = float(np.max(errs)) if len(errs) else 0.0
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + err_max * 0.05,
            f"{val:.1f}",
            ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="#1F2937",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotation, ha="center")
    ax.set_ylabel(ylabel, labelpad=6)
    ax.set_title(title, fontweight="bold", pad=8, loc="left")
    ax.set_axisbelow(True)

    if pval is not None:
        sig = (
            "***" if pval < 0.001
            else "**"  if pval < 0.01
            else "*"   if pval < 0.05
            else "n.s."
        )
        ax.text(
            0.97, 0.97,
            f"$p$ = {pval:.3f}  {sig}",
            transform=ax.transAxes,
            fontsize=7, ha="right", va="top",
            style="italic", color="#6B7280",
        )


# =============================================================================
# FIGURE 1 — Phase comparison
# =============================================================================

def fig_phase_comparison(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)

    specs = [
        ("tokens_sent",      1_000, "Mean tokens sent (thousands)"),
        ("api_calls",        1,     "Mean API calls (adjusted)"),
        ("duration_seconds", 1,     "Mean duration (seconds)"),
    ]
    titles = [
        "(a) Token consumption\nPhase 1 vs Phase 2",
        "(b) API call depth\nPhase 1 vs Phase 2",
        "(c) Wall-clock duration\nPhase 1 vs Phase 2",
    ]

    for ax, (col, scale, ylabel), title in zip(axes, specs, titles):
        grp = (
            df.groupby("phase")[col]
            .agg(["mean", "std"])
            .reindex(["P1", "P2"])
        )
        _, p = stats.ttest_ind(
            df[df["phase"] == "P1"][col],
            df[df["phase"] == "P2"][col],
        )
        bar_panel(
            ax,
            labels=[PHASE_LABELS[ph] for ph in grp.index],
            vals=grp["mean"].values / scale,
            errs=grp["std"].values / scale,
            colors=[PHASE_COLORS[ph] for ph in grp.index],
            ylabel=ylabel, title=title, pval=p,
        )

    savefig("fig_cost_phase_comparison")


# =============================================================================
# FIGURE 2 — Single-agent vs multi-agent
# =============================================================================

def fig_single_vs_multi(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)

    order = ["Single-Agent", "Multi-Agent"]
    specs = [
        ("tokens_sent",      1_000, "Mean tokens sent (thousands)"),
        ("api_calls",        1,     "Mean API calls (adjusted)"),
        ("duration_seconds", 1,     "Mean duration (seconds)"),
    ]
    titles = [
        "(a) Token consumption\nSingle-Agent vs Multi-Agent",
        "(b) API call depth\nSingle-Agent vs Multi-Agent",
        "(c) Wall-clock duration\nSingle-Agent vs Multi-Agent",
    ]

    for ax, (col, scale, ylabel), title in zip(axes, specs, titles):
        grp = (
            df.groupby("agent_class")[col]
            .agg(["mean", "std"])
            .reindex(order)
        )
        _, p = stats.ttest_ind(
            df[df["agent_class"] == "Single-Agent"][col],
            df[df["agent_class"] == "Multi-Agent"][col],
        )
        bar_panel(
            ax,
            labels=order,
            vals=grp["mean"].values / scale,
            errs=grp["std"].values / scale,
            colors=[CLASS_COLORS[c] for c in order],
            ylabel=ylabel, title=title, pval=p,
        )

    savefig("fig_cost_single_vs_multi")


# =============================================================================
# FIGURE 3 — Agent domain cost profile
# =============================================================================

def fig_agent_domain(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.8), constrained_layout=True)

    order = ["IoT", "FMSA", "TSFM", "WO", "E2E"]
    specs = [
        ("tokens_sent",      1_000, "Mean tokens sent (thousands)"),
        ("api_calls",        1,     "Mean API calls (adjusted)"),
        ("duration_seconds", 1,     "Mean duration (seconds)"),
    ]
    titles = [
        "(a) Token consumption\nby agent domain",
        "(b) API call depth\nby agent domain",
        "(c) Wall-clock duration\nby agent domain",
    ]

    for ax, (col, scale, ylabel), title in zip(axes, specs, titles):
        grp = (
            df.groupby("agent")[col]
            .agg(["mean", "std"])
            .reindex(order)
        )
        bar_panel(
            ax,
            labels=order,
            vals=grp["mean"].values / scale,
            errs=grp["std"].values / scale,
            colors=[AGENT_COLORS[a] for a in order],
            ylabel=ylabel, title=title,
        )

    savefig("fig_cost_agent_domain")


# =============================================================================
# FIGURE 4 — Phase × agent heatmap
# =============================================================================

def fig_heatmap(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 3.4), constrained_layout=True)

    agent_order = ["IoT", "FMSA", "TSFM", "WO", "E2E"]
    phase_order = ["P1", "P2"]
    phase_tick  = ["Phase 1\n(Dev.)", "Phase 2\n(Eval.)"]

    specs = [
        ("tokens_sent",      1_000, "Mean tokens (K)"),
        ("api_calls",        1,     "Mean API calls"),
        ("duration_seconds", 1,     "Mean duration (s)"),
    ]

    for ax, (col, scale, cbar_label) in zip(axes, specs):
        pivot = (
            df.groupby(["phase", "agent"])[col]
            .mean()
            .unstack()
            .reindex(index=phase_order, columns=agent_order) / scale
        )
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(agent_order)))
        ax.set_xticklabels(agent_order)
        ax.set_yticks(range(2))
        ax.set_yticklabels(phase_tick)
        # Disable the grid that PAPER_RC enables — heatmap cells handle this
        ax.grid(False)

        vmax = float(np.nanmax(pivot.values))
        for i in range(2):
            for j in range(len(agent_order)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(
                        j, i, f"{val:.1f}",
                        ha="center", va="center",
                        fontsize=7, fontweight="bold",
                        color="white" if val > vmax * 0.6 else "#1F2937",
                    )
        cb = plt.colorbar(im, ax=ax, shrink=0.78, pad=0.03)
        cb.set_label(cbar_label, fontsize=7)
        ax.set_title(cbar_label, fontweight="bold", pad=6, loc="left")

    savefig("fig_cost_heatmap")


# =============================================================================
# FIGURE 5 — Scenario difficulty ordered by token consumption
# =============================================================================

def fig_difficulty_cv(df: pd.DataFrame) -> None:
    scen = (
        df.groupby(["id", "agent", "phase"])["tokens_sent"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_tokens", "std": "std_tokens", "count": "n"})
    )
    scen["label"] = scen["id"].map(lambda x: f"Q{x}")

    p1 = (scen[scen["phase"] == "P1"]
          .sort_values("mean_tokens", ascending=True)
          .reset_index(drop=True))
    p2 = (scen[scen["phase"] == "P2"]
          .sort_values("mean_tokens", ascending=True)
          .reset_index(drop=True))

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True)

    for ax, sub, title in zip(
        axes,
        [p1, p2],
        [
            "(a) Token consumption per scenario — Phase 1 (Development)",
            "(b) Token consumption per scenario — Phase 2 (Evaluation)",
        ],
    ):
        colors = [AGENT_COLORS[a] for a in sub["agent"]]
        ax.barh(
            sub["label"], sub["mean_tokens"] / 1_000,
            color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.4,
        )
        ax.errorbar(
            sub["mean_tokens"] / 1_000, sub["label"],
            xerr=sub["std_tokens"] / 1_000,
            fmt="none", color="#6B7280",
            linewidth=0.8, capsize=2.5,
        )
        for _, row in sub.iterrows():
            x_pos = (row["mean_tokens"] + row["std_tokens"]) / 1_000 + 2
            ax.text(x_pos, row["label"], row["agent"],
                    va="center", fontsize=6, color="#6B7280")

        ax.set_xlabel("Mean tokens sent (thousands)", labelpad=6)
        ax.set_xlim(left=0)
        ax.set_title(title, fontweight="bold", pad=8, loc="left")
        ax.set_axisbelow(True)
        ax.legend(
            handles=LEGEND_AGENT, frameon=True,
            framealpha=0.95, edgecolor="#e5e7eb",
            loc="lower right",
        )
        # Horizontal grid only
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)

    savefig("fig_cost_difficulty_cv")


# =============================================================================
# FIGURE 6 — Cross-run boxplot, both phases side by side
# =============================================================================

def fig_boxplot_both_phases(df: pd.DataFrame) -> None:
    combined_max = max(
        max(df[df["id"] == sid]["tokens_sent"].max() / 1_000
            for sid in PHASE1_IDS),
        max(df[df["id"] == sid]["tokens_sent"].max() / 1_000
            for sid in PHASE2_IDS),
    )

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.0),
                             sharey=True, constrained_layout=True)

    for ax, phase_ids, panel_label, n_label in zip(
        axes,
        [PHASE1_IDS, PHASE2_IDS],
        ["(a) Phase 1 (Development)", "(b) Phase 2 (Evaluation)"],
        [r"$n \approx 180$ executions per scenario",
         r"$n \approx 23$ executions per scenario"],
    ):
        order = sorted(
            phase_ids,
            key=lambda x: df[df["id"] == x]["tokens_sent"].mean(),
            reverse=True,
        )
        data   = [df[df["id"] == sid]["tokens_sent"].values / 1_000
                  for sid in order]
        colors = [AGENT_COLORS[df[df["id"] == sid]["agent"].iloc[0]]
                  for sid in order]
        labels = [
            f"Q{sid}\n({df[df['id'] == sid]['agent'].iloc[0]})"
            for sid in order
        ]

        bp = ax.boxplot(
            data, vert=True, patch_artist=True,
            medianprops=dict(color="white", linewidth=1.2),
            whiskerprops=dict(linewidth=0.7, color="#6B7280"),
            capprops=dict(linewidth=0.7, color="#6B7280"),
            flierprops=dict(
                marker="o", markersize=2,
                markerfacecolor="#9CA3AF",
                alpha=0.4, linestyle="none",
            ),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.80)
            patch.set_linewidth(0.4)

        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Tokens sent (thousands)", labelpad=6)
        ax.set_title(
            f"Cross-run token distribution — {panel_label}\n({n_label})",
            fontweight="bold", pad=8, loc="left",
        )
        ax.set_axisbelow(True)
        ax.legend(
            handles=LEGEND_AGENT, frameon=True,
            framealpha=0.95, edgecolor="#e5e7eb",
            loc="upper right",
        )
        ax.set_ylim(0, combined_max * 1.08)

    savefig("fig_cost_boxplot_both_phases")


# =============================================================================
# FIGURE 7 — Overall cost distributions (Phase 1)
# =============================================================================

def fig_experiment_distributions(df: pd.DataFrame) -> None:
    p1 = df[df["phase"] == "P1"]

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)

    specs = [
        ("tokens_sent",      1_000, "Tokens sent (thousands)", "#1D4ED8",
         "(a) Token consumption\nPhase 1 executions"),
        ("api_calls",        1,     "API calls (adjusted)",    "#7C3AED",
         "(b) API call depth\nPhase 1 executions"),
        ("duration_seconds", 1,     "Duration (seconds)",      "#D97706",
         "(c) Wall-clock duration\nPhase 1 executions"),
    ]

    for ax, (col, scale, xlabel, color, title) in zip(axes, specs):
        vals = p1[col] / scale
        ax.hist(vals, bins=35, color=color, alpha=0.80,
                edgecolor="white", linewidth=0.5)
        med = float(vals.median())
        ax.axvline(med, color="#DC2626", linewidth=1.0, linestyle="--")
        ylim = ax.get_ylim()
        ax.text(
            med * 1.03, ylim[1] * 0.88,
            f"Median\n{med:.1f}",
            fontsize=7, color="#DC2626", va="top",
        )
        ax.set_xlabel(xlabel, labelpad=6)
        ax.set_ylabel("Number of executions", labelpad=6)
        ax.set_title(title, fontweight="bold", pad=8, loc="left")
        ax.set_axisbelow(True)
        # horizontal grid only
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

    savefig("fig_cost_distributions")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"Loading {args.data} …")
    df = load_data(args.data)
    n_traces  = len(df)
    n_scen    = df["id"].nunique()
    n_folders = df["source_folder"].nunique()
    print(f"  {n_traces} execution traces | "
          f"{n_scen} scenarios | "
          f"{n_folders} experiment folders\n")

    print("Generating figures …")
    fig_phase_comparison(df)
    fig_single_vs_multi(df)
    fig_agent_domain(df)
    fig_heatmap(df)
    fig_difficulty_cv(df)
    fig_experiment_distributions(df)
    fig_boxplot_both_phases(df)

    # ── Summary statistics ─────────────────────────────────────────────────
    print("\n── Paper-ready numbers ──────────────────────────────────────")
    for phase, label in [("P1", "Phase 1"), ("P2", "Phase 2")]:
        sub = df[df["phase"] == phase]
        print(
            f"{label}: {len(sub)} executions | "
            f"mean tokens = {sub['tokens_sent'].mean():.0f} | "
            f"mean calls = {sub['api_calls'].mean():.1f} | "
            f"mean dur = {sub['duration_seconds'].mean():.1f} s"
        )

    t, p = stats.ttest_ind(
        df[df["phase"] == "P1"]["tokens_sent"],
        df[df["phase"] == "P2"]["tokens_sent"],
    )
    print(f"\nPhase token t-test: t = {t:.2f}, p = {p:.4f}")

    sa = df[df["agent_class"] == "Single-Agent"]["tokens_sent"]
    ma = df[df["agent_class"] == "Multi-Agent"]["tokens_sent"]
    t, p = stats.ttest_ind(sa, ma)
    print(
        f"Single vs Multi-Agent tokens: "
        f"{sa.mean():.0f} vs {ma.mean():.0f} "
        f"({ma.mean() / sa.mean():.2f}x), t = {t:.2f}, p = {p:.4f}"
    )

    top10 = (
        df[df["tokens_sent"] > df["tokens_sent"].quantile(0.9)]["tokens_sent"].sum()
        / df["tokens_sent"].sum() * 100
    )
    print(f"Top 10 % executions = {top10:.1f} % of total tokens")

    print(f"\nAll figures saved to {OUT}/")