#!/usr/bin/env python3
"""StirrupAgent leaderboard charts: one file per panel.

Reads the CSV that benchmarks/consolidate_results.py writes:

    python benchmarks/consolidate_results.py "$LEADERBOARD_DIR" --csv results.csv
    python make_charts.py --csv results.csv

Without --csv it falls back to the FALLBACK tables at the bottom of this file, so
the script still runs standalone.

    python make_charts.py --csv results.csv --palette cb
    python make_charts.py --csv results.csv --only pass_rate cost_per_pass
    python make_charts.py --list
"""

from __future__ import annotations

import argparse
import csv as csvmod
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# ═══════════════════════ what the consolidator gives us ═════════════════════
#
# consolidate_results.py --csv writes one row per model with these columns:
#
#   model            raw model id
#   n                scenarios scored for this model
#   passed           scenarios passed
#   pass_rate        passed / n
#   score_avg        mean per-scenario score (may be blank)
#   turns_avg        mean turns per scenario
#   calls_avg        mean tool calls per scenario
#   tokens_in_avg    mean input tokens PER SCENARIO
#   tokens_out_avg   mean output tokens PER SCENARIO
#   cost_total       summed est_cost_usd (blank if the harness reported no cost)
#   cost_per_pass    cost_total / passed
#
# Note what is NOT in there: wall-clock duration, Macro-F1, and the input/output
# cost split. Those live in EXTRAS below and are only charted if you supply them.

# Raw model ids are long; map them to what should appear under the bars. Any id
# not listed here is shortened automatically.
DISPLAY = {
    # "anthropic/claude-opus-5-20260514": ("Opus 5", "high"),
}

# metrics the consolidator does not emit; leave empty to skip those panels
EXTRAS: dict[str, dict[str, float]] = {
    # "duration_p50": {"Opus 5": 92, "GPT-5.6": 50, ...},          # seconds
    # "macro_f1":     {"Opus 5": 0.468, "GPT-5.6": 0.460, ...},
    # "input_cost":   {"Opus 5": 94.20, ...},   # both needed for the cost table
    # "output_cost":  {"Opus 5": 8.69, ...},
}

# ═══════════════════════════════ style ══════════════════════════════════════

INK, MUTED, GRID = "#161616", "#525252", "#DCDCDC"
TABLE_BG, TABLE_HEAD, TABLE_RULE = "#0D0D0D", "#F1C21B", "#3A3A3A"

PALETTES = {
    "default": ["#1F1F1F", "#2E7CF6", "#2CA84F", "#C08672", "#FF7A0D"],
    # Okabe-Ito derived: adjacent pairs stay separable under deuteranopia,
    # protanopia and greyscale printing
    "cb": ["#1F1F1F", "#0072B2", "#009E73", "#CC79A7", "#E69F00"],
    "carbon": ["#161616", "#0F62FE", "#007D79", "#9F1853", "#D02670"],
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.monospace": ["DejaVu Sans Mono", "Courier New"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,          # embed TrueType: PDF text stays selectable
    "figure.dpi": 110,
})


# ═══════════════════════════════ loading ════════════════════════════════════

def _num(v):
    """CSV cell to float, or None for a blank the consolidator left empty."""
    if v is None or str(v).strip() in ("", "-", "None"):
        return None
    return float(v)


def _shorten(model_id: str) -> tuple[str, str]:
    """Fall back to a readable (name, effort) when DISPLAY has no entry."""
    name = model_id.rsplit("/", 1)[-1]
    effort = ""
    for tag in ("high", "max", "medium", "low", "minimal"):
        for sep in (f"-{tag}", f"_{tag}", f":{tag}"):
            if name.endswith(sep):
                name, effort = name[: -len(sep)], tag
                break
        if effort:
            break
    return name.replace("_", " "), effort


class Results:
    """Everything the charts need, loaded from one consolidated CSV."""

    def __init__(self, rows: list[dict], order: list[str] | None = None):
        self.raw = {r["model"]: r for r in rows}
        ids = order or [r["model"] for r in rows]
        self.ids = [m for m in ids if m in self.raw]

        self.name, self.effort = {}, {}
        for mid in self.ids:
            n, e = DISPLAY.get(mid) or _shorten(mid)
            self.name[mid], self.effort[mid] = n, e

        g = lambda mid, k: _num(self.raw[mid].get(k))
        self.n = {m: int(g(m, "n") or 0) for m in self.ids}
        self.passed = {m: int(g(m, "passed") or 0) for m in self.ids}

        self.pass_rate = self._series("pass_rate")
        self.score_avg = self._series("score_avg")
        self.turns = self._series("turns_avg")
        self.tool_calls = self._series("calls_avg")
        self.cost_total = self._series("cost_total")
        self.cost_per_pass = self._series("cost_per_pass")

        # the consolidator reports tokens PER SCENARIO; totals are what the
        # leaderboard slide shows, so derive them rather than re-typing them
        ti, to = self._series("tokens_in_avg"), self._series("tokens_out_avg")
        self.tokens_in_m = {m: v * self.n[m] / 1e6 for m, v in ti.items()}
        self.tokens_out_m = {m: v * self.n[m] / 1e6 for m, v in to.items()}

        self.cost_per_scenario = {m: c / self.n[m]
                                  for m, c in self.cost_total.items() if self.n[m]}
        # recompute rather than trusting the column, so a blank passed count
        # cannot silently produce a wrong bar
        self.cost_per_pass = {m: self.cost_total[m] / self.passed[m]
                              for m in self.cost_total if self.passed.get(m)}

    def _series(self, col: str) -> dict[str, float]:
        """Column as a dict, silently dropping models where it is blank."""
        out = {}
        for mid in self.ids:
            v = _num(self.raw[mid].get(col))
            if v is not None:
                out[mid] = v
        return out

    def extra(self, key: str) -> dict[str, float]:
        """An EXTRAS series, keyed by display name, remapped onto model ids."""
        src = EXTRAS.get(key) or {}
        return {mid: src[self.name[mid]] for mid in self.ids
                if self.name[mid] in src}

    @property
    def scenarios(self) -> int:
        return max(self.n.values()) if self.n else 0


def load_csv(path: Path, order: list[str] | None = None) -> Results:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csvmod.DictReader(fh))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    missing = {"model", "n", "passed", "pass_rate"} - set(rows[0])
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(sorted(missing))}"
                         "\nis this the output of consolidate_results.py --csv?")
    return Results(rows, order)


def load_fallback() -> Results:
    """Rebuild the consolidator's schema from the hardcoded tables below."""
    rows = []
    for m in FALLBACK_ORDER:
        n = FALLBACK_N
        rows.append({
            "model": m, "n": n, "passed": round(FB_PASS_RATE[m] * n),
            "pass_rate": FB_PASS_RATE[m], "score_avg": FB_MACRO_F1[m],
            "turns_avg": "", "calls_avg": FB_TOOL_CALLS[m],
            "tokens_in_avg": FB_TOKENS_IN[m] * 1e6 / n,
            "tokens_out_avg": FB_TOKENS_OUT[m] * 1e6 / n,
            "cost_total": FB_INPUT_COST[m] + FB_OUTPUT_COST[m],
            "cost_per_pass": "",
        })
    EXTRAS.setdefault("duration_p50", dict(FB_DURATION))
    EXTRAS.setdefault("macro_f1", dict(FB_MACRO_F1))
    EXTRAS.setdefault("input_cost", dict(FB_INPUT_COST))
    EXTRAS.setdefault("output_cost", dict(FB_OUTPUT_COST))
    return Results(rows, FALLBACK_ORDER)


# ═══════════════════════════════ charts ═════════════════════════════════════

def bar_chart(res: Results, values: dict[str, float], *, title, fmt, name,
              outdir: Path, palette: list[str], caption=None,
              title_align="left", ymax=None, headroom=1.28,
              figsize=(7.4, 4.6), bar_width=0.62, label_fs=12, value_fs=14,
              baseline=True, also_pdf=True, close=True):
    """One metric, its own axes, its own file. No horizontal gridlines."""
    ids = [m for m in res.ids if m in values]
    if not ids:
        print(f"skip {name}: no data")
        return None
    heights = [values[m] for m in ids]
    colors = [palette[res.ids.index(m) % len(palette)] for m in ids]
    top = ymax if ymax is not None else max(heights) * headroom

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(ids)), heights, width=bar_width, color=colors, zorder=3)

    for i, h in enumerate(heights):
        ax.text(i, h + top * 0.025, fmt(h), ha="center", va="bottom",
                fontsize=value_fs, fontweight="bold", color=INK, zorder=4)

    ax.set_ylim(0, top)
    ax.set_yticks([])          # no y scale and no horizontal rules
    ax.grid(False)

    labels = []
    for m in ids:
        lab = "\n".join(textwrap.wrap(res.name[m], 12, break_long_words=False))
        if res.effort[m]:
            lab += f"\n({res.effort[m]})"
        labels.append(lab)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(labels, fontsize=label_fs, color=INK)
    ax.tick_params(axis="x", length=0, pad=8)

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    if baseline:
        ax.spines["bottom"].set_color(GRID)
        ax.spines["bottom"].set_linewidth(1.0)
    else:
        ax.spines["bottom"].set_visible(False)

    if title_align == "center":
        ax.set_title(title, fontsize=17, color=MUTED, pad=18)
    else:
        ax.set_title(title, fontsize=17, color=INK, fontweight="bold",
                     loc="left", pad=18)
    if caption:
        ax.set_title(caption, fontsize=11, color="#8D8D8D", loc="right", pad=20)

    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{name}.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    if also_pdf:
        fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    print(f"wrote {outdir / name}.png")
    if close:
        plt.close(fig)
    return fig


def cost_table(res: Results, outdir: Path, *, also_pdf=True,
               figsize=(7.6, 2.6), close=True):
    """Spend table. Splits input/output when EXTRAS supplies them."""
    inp, outp = res.extra("input_cost"), res.extra("output_cost")
    split = bool(inp and outp)
    ids = sorted(res.cost_total, key=lambda m: -res.cost_total[m])
    if not ids:
        print("skip cost_table: the harness reported no cost")
        return None

    if split:
        headers = ("Model", "Input cost", "Output cost", "Total", "$/pass")
        colx = (0.030, 0.470, 0.640, 0.805, 0.970)
        rows = [(res.name[m], f"${inp[m]:,.2f}", f"${outp[m]:,.2f}",
                 f"${res.cost_total[m]:,.2f}",
                 f"${res.cost_per_pass.get(m, float('nan')):,.2f}") for m in ids]
    else:
        headers = ("Model", "Scenarios", "Total cost", "$/pass")
        colx = (0.030, 0.560, 0.780, 0.970)
        rows = [(res.name[m], str(res.n[m]), f"${res.cost_total[m]:,.2f}",
                 f"${res.cost_per_pass.get(m, float('nan')):,.2f}") for m in ids]

    n = len(rows)
    row_h = 1.0 / (n + 2.2)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(TABLE_BG); fig.patch.set_facecolor(TABLE_BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    y = 1 - row_h * 1.1
    for i, h in enumerate(headers):
        ax.text(colx[i], y, h, color=TABLE_HEAD, fontsize=11.5,
                family="monospace", fontweight="bold",
                ha="left" if i == 0 else "right", va="center")
    ax.plot([0.02, 0.985], [y - row_h * 0.55] * 2, color=TABLE_HEAD, lw=1.4)

    for r, cells in enumerate(rows):
        y -= row_h
        for i, c in enumerate(cells):
            ax.text(colx[i], y, c, color="#F4F4F4", fontsize=11,
                    family="monospace", ha="left" if i == 0 else "right",
                    va="center")
        if r < n - 1:
            ax.plot([0.02, 0.985], [y - row_h * 0.5] * 2, color=TABLE_RULE, lw=0.8)

    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "cost_table.png", dpi=300, bbox_inches="tight",
                facecolor=TABLE_BG)
    if also_pdf:
        fig.savefig(outdir / "cost_table.pdf", bbox_inches="tight",
                    facecolor=TABLE_BG)
    print(f"wrote {outdir / 'cost_table'}.png")
    if close:
        plt.close(fig)
    return fig


def panels(res: Results) -> dict[str, dict]:
    """Panel name -> bar_chart kwargs. Only panels with data are returned."""
    ns = res.scenarios
    spec = {
        "pass_rate": dict(values=res.pass_rate, title="Pass rate",
                          fmt=lambda v: f"{v * 100:.1f}%",
                          title_align="center", figsize=(8.0, 5.0),
                          bar_width=0.52),
        "score_avg": dict(values=res.score_avg, title="Average score",
                          caption="mean per-scenario score",
                          fmt=lambda v: f"{v:.3f}"),
        "macro_f1": dict(values=res.extra("macro_f1"), title="Macro-F1",
                         fmt=lambda v: f"{v:.3f}", title_align="center",
                         figsize=(8.0, 5.0), bar_width=0.52),
        "tokens_in": dict(values=res.tokens_in_m, title="Tokens In",
                          caption="total, millions",
                          fmt=lambda v: f"{v:.1f}M"),
        "tokens_out": dict(values=res.tokens_out_m, title="Tokens Out",
                           caption="total, millions",
                           fmt=lambda v: f"{v:.2f}M"),
        "turns": dict(values=res.turns, title="Turns",
                      caption="avg per scenario", fmt=lambda v: f"{v:.1f}"),
        "tool_calls": dict(values=res.tool_calls, title="Tool Calls",
                           caption="avg per scenario", fmt=lambda v: f"{v:.1f}"),
        "duration": dict(values=res.extra("duration_p50"), title="Duration",
                         caption="p50, seconds", fmt=lambda v: f"{v:.0f}s"),
        "cost_total": dict(values=res.cost_total, title="Total cost",
                           caption=f"{ns} scenarios, USD",
                           fmt=lambda v: f"${v:,.0f}"),
        "cost_per_scenario": dict(values=res.cost_per_scenario,
                                  title="Cost per scenario",
                                  caption="USD", fmt=lambda v: f"${v:,.2f}"),
        "cost_per_pass": dict(values=res.cost_per_pass,
                              title="Cost per successful pass",
                              caption="USD, what a usable answer costs",
                              fmt=lambda v: f"${v:,.2f}"),
    }
    return {k: v for k, v in spec.items() if v["values"]}


# ═══════════════════════════════ fallback data ══════════════════════════════
# Used only when --csv is omitted. Keeps the script runnable on its own.

FALLBACK_N = 50
FALLBACK_ORDER = ["Opus 5", "GPT-5.6", "Gemini 3.6", "Sonnet 5", "MiniMax"]
FB_PASS_RATE = {"Opus 5": 0.320, "GPT-5.6": 0.300, "Gemini 3.6": 0.320,
                "Sonnet 5": 0.280, "MiniMax": 0.280}
FB_MACRO_F1 = {"Opus 5": 0.468, "GPT-5.6": 0.460, "Gemini 3.6": 0.459,
               "Sonnet 5": 0.402, "MiniMax": 0.400}
FB_TOKENS_IN = {"Opus 5": 18.8, "GPT-5.6": 6.9, "Gemini 3.6": 32.3,
                "Sonnet 5": 36.3, "MiniMax": 19.9}       # millions, total
FB_TOKENS_OUT = {"Opus 5": 0.35, "GPT-5.6": 0.13, "Gemini 3.6": 0.65,
                 "Sonnet 5": 2.31, "MiniMax": 0.42}
FB_DURATION = {"Opus 5": 92, "GPT-5.6": 50, "Gemini 3.6": 171,
               "Sonnet 5": 441, "MiniMax": 110}          # p50 seconds
FB_TOOL_CALLS = {"Opus 5": 15.6, "GPT-5.6": 12.1, "Gemini 3.6": 25.7,
                 "Sonnet 5": 31.9, "MiniMax": 18.5}
FB_INPUT_COST = {"Opus 5": 94.20, "GPT-5.6": 34.74, "Gemini 3.6": 48.46,
                 "Sonnet 5": 72.61, "MiniMax": 5.98}
FB_OUTPUT_COST = {"Opus 5": 8.69, "GPT-5.6": 3.85, "Gemini 3.6": 4.85,
                  "Sonnet 5": 23.10, "MiniMax": 0.50}


# ═══════════════════════════════ cli ════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path,
                    help="output of consolidate_results.py --csv")
    ap.add_argument("--outdir", type=Path, default=Path("figs"))
    ap.add_argument("--palette", default="default", choices=sorted(PALETTES))
    ap.add_argument("--order", nargs="*", metavar="MODEL_ID",
                    help="model ids in the order to plot them")
    ap.add_argument("--only", nargs="*", metavar="PANEL")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    res = load_csv(args.csv, args.order) if args.csv else load_fallback()
    spec = panels(res)

    if args.list:
        print("\n".join(list(spec) + ["cost_table"]))
        return

    pal = PALETTES[args.palette]
    for name in (args.only or list(spec) + ["cost_table"]):
        if name == "cost_table":
            cost_table(res, args.outdir, also_pdf=not args.no_pdf)
        elif name in spec:
            bar_chart(res, name=name, outdir=args.outdir, palette=pal,
                      also_pdf=not args.no_pdf, **spec[name])
        else:
            print(f"unknown panel: {name} (try --list)")

    print(f"\n{res.scenarios} scenarios, {len(res.ids)} models")
    for m in res.ids:
        cost = res.cost_total.get(m)
        per = res.cost_per_pass.get(m)
        print(f"  {res.name[m]:<14} pass {res.pass_rate.get(m, 0):.1%}"
              f"   total {'$%.2f' % cost if cost else '-':>9}"
              f"   $/pass {'$%.2f' % per if per else '-':>8}")


if __name__ == "__main__":
    main()
