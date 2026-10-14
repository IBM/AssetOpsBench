"""Accessible leaderboard charts derived from an :class:`EvalReport`."""

from __future__ import annotations

import hashlib
import json
import re
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any

from .models import EvalReport

DEFAULT_CRITERIA = (
    "task_completion",
    "data_retrieval_accuracy",
    "generalized_result_verification",
)

CRITERION_LABELS = {
    "task_completion": "Task completion",
    "data_retrieval_accuracy": "Data retrieval accuracy",
    "generalized_result_verification": "Result verification",
    "hallucinations": "Hallucination-free",
}

# Official Carbon categorical palette for light themes, in documented order:
# https://carbondesignsystem.com/data-visualization/color-palettes/
CARBON_CATEGORICAL_COLORS = (
    "#6929c4",
    "#1192e8",
    "#005d5d",
    "#9f1853",
    "#fa4d56",
    "#570408",
    "#198038",
    "#002d9c",
    "#ee538b",
    "#b28600",
    "#009d9a",
    "#012749",
    "#8a3800",
    "#a56eff",
)

# Pattern is redundant with color. A prime-sized cycle delays repeated pairs
# when the report contains more models than Carbon's base palette.
_HATCHES = (
    "///",
    "\\\\\\",
    "|||",
    "---",
    "+++",
    "xxx",
    "...",
    "ooo",
    "OOO",
    "***",
    "/\\",
    "|+",
    "x.",
)
_INVERTED_CRITERIA = frozenset({"hallucinations"})
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_MANAGED_NAME = re.compile(r"leaderboard-[A-Za-z0-9._-]+\.(?:svg|png)")
_MANIFEST_FILENAME = ".leaderboard-manifest.json"


class VisualizationDependencyError(RuntimeError):
    """Raised when chart generation is requested without Matplotlib."""


@dataclass(frozen=True)
class CriterionRate:
    """Exact Boolean counts and their derived percentage for one criterion."""

    criterion: str
    successes: int
    applicable: int

    @property
    def percentage(self) -> float | None:
        if self.applicable == 0:
            return None
        return self.successes / self.applicable * 100.0


@dataclass(frozen=True)
class ModelRates:
    """Criterion rates for one model within one runner."""

    model: str
    rates: tuple[CriterionRate, ...]


@dataclass(frozen=True)
class LeaderboardData:
    """Plot-ready, scorer-specific data for one orchestration method."""

    runner: str
    criteria: tuple[str, ...]
    models: tuple[ModelRates, ...]


@dataclass(frozen=True)
class ModelStyle:
    """A report-global categorical color and redundant hatch pattern."""

    color: str
    hatch: str
    hatch_color: str


def criterion_rates(
    report: EvalReport,
    runner: str,
    criteria: Sequence[str] = DEFAULT_CRITERIA,
) -> LeaderboardData:
    """Aggregate strict Boolean LLM-judge details for one runner.

    Missing and non-Boolean criterion values are excluded from the denominator.
    The raw ``hallucinations`` criterion is inverted into a positive outcome.
    """
    criterion_names = tuple(criteria)
    if len(set(criterion_names)) != len(criterion_names):
        raise ValueError("leaderboard criteria must be unique")

    matching = [
        result
        for result in report.results
        if result.runner == runner and result.score.scorer == "llm_judge"
    ]
    model_names = sorted({result.model for result in matching})
    grouped: dict[str, dict[str, list[int]]] = {
        model: {criterion: [0, 0] for criterion in criterion_names}
        for model in model_names
    }

    for result in matching:
        for criterion in criterion_names:
            value = result.score.details.get(criterion)
            if not isinstance(value, bool):
                continue
            counts = grouped[result.model][criterion]
            counts[1] += 1
            positive = not value if criterion in _INVERTED_CRITERIA else value
            counts[0] += int(positive)

    models = tuple(
        ModelRates(
            model=model,
            rates=tuple(
                CriterionRate(
                    criterion=criterion,
                    successes=grouped[model][criterion][0],
                    applicable=grouped[model][criterion][1],
                )
                for criterion in criterion_names
            ),
        )
        for model in model_names
    )
    return LeaderboardData(runner=runner, criteria=criterion_names, models=models)


def model_styles(report: EvalReport) -> dict[str, ModelStyle]:
    """Assign deterministic styles globally across all LLM-judge results."""
    models = sorted(
        {
            result.model
            for result in report.results
            if result.score.scorer == "llm_judge"
        }
    )
    styles: dict[str, ModelStyle] = {}
    for index, model in enumerate(models):
        color = CARBON_CATEGORICAL_COLORS[index % len(CARBON_CATEGORICAL_COLORS)]
        styles[model] = ModelStyle(
            color=color,
            hatch=_HATCHES[index % len(_HATCHES)],
            hatch_color=_hatch_color(color),
        )
    return styles


def require_visualization_dependency() -> None:
    """Fail early, before evaluation may make a paid judge-model call."""
    _load_matplotlib()


def render_leaderboards(
    report: EvalReport,
    output_dir: Path,
    criteria: Sequence[str] = DEFAULT_CRITERIA,
) -> tuple[Path, ...]:
    """Render one SVG and PNG leaderboard per applicable runner."""
    styles = model_styles(report)
    runners = sorted(
        {
            result.runner
            for result in report.results
            if result.score.scorer == "llm_judge"
        }
    )
    leaderboards = [criterion_rates(report, runner, criteria) for runner in runners]
    leaderboards = [data for data in leaderboards if _has_applicable_data(data)]
    if not leaderboards:
        _publish_outputs(Path(output_dir), None, ())
        return ()

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    matplotlib, figures, backend_agg, patches = _load_matplotlib()
    generated_names: list[str] = []
    with TemporaryDirectory(
        dir=output_dir.parent, prefix=".leaderboard-render-"
    ) as temporary:
        staged_dir = Path(temporary)
        with matplotlib.rc_context(_MATPLOTLIB_STYLE):
            for data in leaderboards:
                figure = _render_figure(figures, backend_agg, patches, data, styles)
                try:
                    title = f"AssetOpsBench {data.runner} leaderboard"
                    description = _chart_description(data)
                    base_name = f"leaderboard-{_filesystem_safe(data.runner)}"
                    for suffix in (".svg", ".png"):
                        name = f"{base_name}{suffix}"
                        path = staged_dir / name
                        figure.savefig(
                            path,
                            dpi=180,
                            bbox_inches="tight",
                            facecolor="white",
                            metadata={
                                "Title": title,
                                "Description": description,
                                "Creator": "AssetOpsBench",
                                "Date": report.generated_at,
                            },
                        )
                        if suffix == ".svg":
                            _enhance_svg(path, description)
                        generated_names.append(name)
                finally:
                    figure.clear()
        return _publish_outputs(output_dir, staged_dir, tuple(generated_names))


def _load_matplotlib() -> tuple[ModuleType, ModuleType, ModuleType, ModuleType]:
    try:
        matplotlib = import_module("matplotlib")
        figures = import_module("matplotlib.figure")
        backend_agg = import_module("matplotlib.backends.backend_agg")
        patches = import_module("matplotlib.patches")
    except ModuleNotFoundError as exc:
        raise VisualizationDependencyError(
            "leaderboard charts require the visualization dependency group; "
            "install it with `uv sync --group visualization`"
        ) from exc
    return matplotlib, figures, backend_agg, patches


def _render_figure(
    figures: ModuleType,
    backend_agg: ModuleType,
    patches: ModuleType,
    data: LeaderboardData,
    styles: dict[str, ModelStyle],
) -> Any:
    model_count = len(data.models)
    width = max(10.0, min(16.0, 7.5 + model_count * 0.65))
    legend_labels = [textwrap.fill(model.model, width=42) for model in data.models]
    legend_lines = sum(label.count("\n") + 1 for label in legend_labels)
    height = min(12.0, max(6.5, 5.6 + legend_lines * 0.22))
    figure = figures.Figure(figsize=(width, height), layout="constrained")
    backend_agg.FigureCanvasAgg(figure)
    axis = figure.subplots()
    bar_width = min(0.78 / max(model_count, 1), 0.22)

    legend_handles = []
    for model_index, (model_data, legend_label) in enumerate(
        zip(data.models, legend_labels, strict=True)
    ):
        style = styles[model_data.model]
        offset = (model_index - (model_count - 1) / 2) * bar_width
        positions = [index + offset for index in range(len(data.criteria))]
        heights = [rate.percentage or 0.0 for rate in model_data.rates]
        bars = axis.bar(
            positions,
            heights,
            width=bar_width * 0.94,
            color=style.color,
            edgecolor="#161616",
            linewidth=0.8,
            zorder=3,
        )
        hatch_bars = axis.bar(
            positions,
            heights,
            width=bar_width * 0.94,
            color="none",
            edgecolor=style.hatch_color,
            linewidth=0,
            hatch=style.hatch,
            zorder=4,
        )
        for bar, rate in zip(bars, model_data.rates, strict=True):
            percentage = rate.percentage
            if percentage is None:
                bar.set_visible(False)
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    1.5,
                    "N/A",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=7,
                    color="#525252",
                )
                continue
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                percentage + 1.5,
                f"{percentage:.1f}%",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=7.5,
                color="#161616",
            )
        for hatch_bar, rate in zip(hatch_bars, model_data.rates, strict=True):
            if rate.percentage is None:
                hatch_bar.set_visible(False)
        legend_handles.append(
            patches.Patch(
                facecolor=style.color,
                edgecolor=style.hatch_color,
                linewidth=0.8,
                hatch=style.hatch,
                label=legend_label,
            )
        )

    axis.set_title(textwrap.fill(f"{data.runner} leaderboard", 72), loc="left", pad=18)
    axis.set_ylabel("Success rate (%)")
    axis.set_xticks(
        range(len(data.criteria)),
        [textwrap.fill(_criterion_label(name), 22) for name in data.criteria],
    )
    axis.set_ylim(0, 112)
    axis.set_yticks(range(0, 101, 20), [f"{value}%" for value in range(0, 101, 20)])
    axis.axhline(0, color="#161616", linewidth=1.0)
    axis.yaxis.grid(True, color="#e0e0e0", linewidth=0.8, zorder=0)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="both", length=0)
    figure.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncols=min(2, model_count),
        frameon=False,
        title="Models (color and pattern)",
        fontsize=8.5,
    )
    return figure


def _criterion_label(criterion: str) -> str:
    return CRITERION_LABELS.get(criterion, criterion.replace("_", " ").capitalize())


def _has_applicable_data(data: LeaderboardData) -> bool:
    return any(rate.applicable for model in data.models for rate in model.rates)


def _filesystem_safe(value: str) -> str:
    cleaned = _SAFE_NAME.sub("-", value).strip("-._").lower() or "runner"
    if cleaned == value and len(cleaned) <= 80:
        return cleaned
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned[:67]}-{digest}"


def _chart_description(data: LeaderboardData) -> str:
    model_values = []
    for model in data.models:
        rates = "; ".join(
            f"{_criterion_label(rate.criterion)} "
            + (
                f"{rate.percentage:.1f}% ({rate.successes}/{rate.applicable})"
                if rate.percentage is not None
                else "N/A"
            )
            for rate in model.rates
        )
        model_values.append(f"{model.model}: {rates}")
    return (
        f"Grouped bar chart for {data.runner}. Models are distinguished by color "
        "and pattern. Values are LLM-judge success percentages; missing Boolean "
        f"criteria are shown as N/A. Data: {' | '.join(model_values)}"
    )


def _enhance_svg(path: Path, description: str) -> None:
    """Add screen-reader semantics not emitted by Matplotlib's SVG backend."""
    svg = path.read_text(encoding="utf-8")
    svg = svg.replace(
        "<svg ",
        '<svg role="img" aria-labelledby="chart-title chart-desc" ',
        1,
    )
    svg = svg.replace("<title>", '<title id="chart-title">', 1)
    svg = svg.replace(
        "</title>",
        f'</title>\n <desc id="chart-desc">{escape(description)}</desc>',
        1,
    )
    path.write_text(svg, encoding="utf-8")


def _publish_outputs(
    output_dir: Path,
    staged_dir: Path | None,
    names: tuple[str, ...],
) -> tuple[Path, ...]:
    """Publish staged files atomically and remove only prior managed files."""
    previous = _read_manifest(output_dir)
    staged_manifest: Path | None = None
    if names:
        if staged_dir is None:
            raise ValueError("staged output directory is required")
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            if not _MANAGED_NAME.fullmatch(name):
                raise ValueError(f"unsafe generated chart name: {name!r}")
            (staged_dir / name).replace(output_dir / name)
        staged_manifest = staged_dir / _MANIFEST_FILENAME
        staged_manifest.write_text(
            json.dumps({"version": 1, "files": list(names)}, indent=2) + "\n",
            encoding="utf-8",
        )

    for name in previous.difference(names):
        (output_dir / name).unlink(missing_ok=True)

    manifest = output_dir / _MANIFEST_FILENAME
    if staged_manifest is not None:
        staged_manifest.replace(manifest)
    else:
        manifest.unlink(missing_ok=True)
    return tuple(output_dir / name for name in names)


def _read_manifest(output_dir: Path) -> set[str]:
    manifest = output_dir / _MANIFEST_FILENAME
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return set()
    files = raw.get("files") if isinstance(raw, dict) else None
    if not isinstance(files, list):
        return set()
    return {
        name
        for name in files
        if isinstance(name, str)
        and Path(name).name == name
        and _MANAGED_NAME.fullmatch(name)
    }


def _hatch_color(color: str) -> str:
    luminance = _relative_luminance(color)
    dark_luminance = _relative_luminance("#161616")
    dark_contrast = (luminance + 0.05) / (dark_luminance + 0.05)
    light_contrast = 1.05 / (luminance + 0.05)
    return "#ffffff" if light_contrast > dark_contrast else "#161616"


def _relative_luminance(color: str) -> float:
    channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


_MATPLOTLIB_STYLE = {
    "axes.facecolor": "white",
    "axes.labelcolor": "#161616",
    "axes.titlecolor": "#161616",
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "font.family": "sans-serif",
    "font.size": 10,
    "hatch.linewidth": 0.8,
    "svg.hashsalt": "assetopsbench-leaderboards",
    "svg.fonttype": "none",
    "text.color": "#161616",
    "xtick.color": "#161616",
    "ytick.color": "#161616",
}
