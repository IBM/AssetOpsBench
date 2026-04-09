"""Generate leaderboard tables (JSON + LaTeX) from evaluation results."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)


def _fmt(val: float, digits: int = 1) -> str:
    return f"{val * 100:.{digits}f}"


def _fmt_ci(lo: float, hi: float, digits: int = 1) -> str:
    return f"({lo * 100:.{digits}f}–{hi * 100:.{digits}f})"


def generate_json(metrics: dict[str, Any], output_dir: Path) -> Path:
    """Write *metrics* as ``leaderboard.json``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "leaderboard.json"
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    _log.info("Wrote %s", path)
    return path


def generate_latex_main(metrics: dict[str, Any], output_dir: Path) -> Path:
    """Produce ``leaderboard.tex`` — the main results table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"  \caption{Industrial Skills Leaderboard. Pass rates (\%) across agent-model "
        r"configurations under three evaluation conditions. Results averaged over $N$ runs "
        r"with 95\% bootstrapped CIs.}",
        r"  \label{tab:main_results}",
        r"  \centering",
        r"  \begin{tabular}{l c c c c}",
        r"    \toprule",
        r"    Agent-Model & No Skills & MCP Only & MCP+Skills & $\Delta$ \\",
        r"    \midrule",
    ]
    for model, data in metrics.items():
        ns = data.get("no_skills", {})
        mo = data.get("mcp_only", {})
        ms = data.get("mcp_skills", {})
        delta = data.get("normalized_gain", 0.0)

        ns_str = _fmt(ns.get("pass_rate", 0)) if ns else "--"
        mo_str = _fmt(mo.get("pass_rate", 0)) if mo else "--"
        ms_str = _fmt(ms.get("pass_rate", 0)) if ms else "--"
        d_str = _fmt(delta)

        short_name = model.rsplit("/", 1)[-1] if "/" in model else model
        lines.append(
            f"    {short_name} & {ns_str} & {mo_str} & {ms_str} & {d_str} \\\\"
        )

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    path = output_dir / "leaderboard.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log.info("Wrote %s", path)
    return path


def generate_latex_per_category(metrics: dict[str, Any], output_dir: Path) -> Path:
    """Produce ``per_category.tex`` — per-category breakdown table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cats = [
        "asset_configuration",
        "diagnostics_reasoning",
        "forecasting_monitoring",
        "maintenance_planning",
    ]
    short_cats = ["Config.", "Diag.", "Forecast.", "Maint."]
    cols = " c" * len(short_cats)

    lines = [
        r"\begin{table}[t]",
        r"  \caption{Per-category pass rates (\%) under the MCP+Skills condition.}",
        r"  \label{tab:per_category}",
        r"  \centering",
        rf"  \begin{{tabular}}{{l{cols} c}}",
        r"    \toprule",
        r"    Agent-Model & " + " & ".join(short_cats) + r" & Overall \\",
        r"    \midrule",
    ]
    for model, data in metrics.items():
        per_cat = data.get("per_category", {})
        short_name = model.rsplit("/", 1)[-1] if "/" in model else model
        vals = []
        for c in cats:
            cat_data = per_cat.get(c, {}).get("mcp_skills", {})
            vals.append(_fmt(cat_data.get("pass_rate", 0)) if cat_data else "--")
        overall = data.get("mcp_skills", {}).get("pass_rate", 0)
        vals.append(_fmt(overall))
        lines.append(f"    {short_name} & " + " & ".join(vals) + r" \\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    path = output_dir / "per_category.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log.info("Wrote %s", path)
    return path


def generate_latex_ablation(metrics: dict[str, Any], output_dir: Path) -> Path:
    """Produce ``ablation.tex`` — skill structure/granularity ablation table.

    Rows correspond to ablation labels found in *metrics* (any condition key
    that is not one of the three base conditions).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    base_conds = {"no_skills", "mcp_only", "mcp_skills"}

    # Collect all ablation labels present across models.
    abl_labels: list[str] = []
    for data in metrics.values():
        for key in data:
            if key not in base_conds and key not in (
                "normalized_gain",
                "per_category",
            ):
                if key not in abl_labels:
                    abl_labels.append(key)
    if not abl_labels:
        # Nothing to write — return empty file.
        path = output_dir / "ablation.tex"
        path.write_text("% No ablation results available.\n", encoding="utf-8")
        return path

    col_headers = ["MCP+Skills (full)"] + [
        lbl.replace("_", r"\_") for lbl in abl_labels
    ]
    cols = " c" * len(col_headers)

    lines = [
        r"\begin{table}[t]",
        r"  \caption{Ablation study: effect of skill building blocks and "
        r"granularity on pass rates (\%).}",
        r"  \label{tab:ablation}",
        r"  \centering",
        rf"  \begin{{tabular}}{{l{cols}}}",
        r"    \toprule",
        r"    Agent-Model & " + " & ".join(col_headers) + r" \\",
        r"    \midrule",
    ]

    for model, data in metrics.items():
        short = model.rsplit("/", 1)[-1] if "/" in model else model
        ms = data.get("mcp_skills", {})
        vals = [_fmt(ms.get("pass_rate", 0)) if ms else "--"]
        for lbl in abl_labels:
            abl_data = data.get(lbl, {})
            vals.append(_fmt(abl_data.get("pass_rate", 0)) if abl_data else "--")
        lines.append(f"    {short} & " + " & ".join(vals) + r" \\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    path = output_dir / "ablation.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log.info("Wrote %s", path)
    return path


def generate_leaderboard(
    metrics: dict[str, Any],
    output_dir: Path,
    formats: list[str] | None = None,
) -> list[Path]:
    """Generate all requested leaderboard outputs.

    Args:
        metrics: Aggregated metrics from :func:`aggregate_results`.
        output_dir: Directory for output files.
        formats: List of formats to produce (``json``, ``latex``).
                 Defaults to both.

    Returns:
        Paths of generated files.
    """
    if formats is None:
        formats = ["json", "latex"]
    paths: list[Path] = []
    if "json" in formats:
        paths.append(generate_json(metrics, output_dir))
    if "latex" in formats:
        paths.append(generate_latex_main(metrics, output_dir))
        paths.append(generate_latex_per_category(metrics, output_dir))
        paths.append(generate_latex_ablation(metrics, output_dir))
    return paths
