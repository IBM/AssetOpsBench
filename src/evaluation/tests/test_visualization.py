"""Tests for report-derived leaderboard aggregation and rendering."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

from evaluation.models import (
    EvalReport,
    OpsMetrics,
    ScenarioResult,
    ScorerResult,
)
from evaluation.report import build_report
from evaluation.visualization import (
    CARBON_CATEGORICAL_COLORS,
    CriterionRate,
    VisualizationDependencyError,
    criterion_rates,
    model_styles,
    render_leaderboards,
    require_visualization_dependency,
)


def _result(
    *,
    runner: str = "meta-agent",
    model: str = "model-a",
    scorer: str = "llm_judge",
    **details: object,
) -> ScenarioResult:
    return ScenarioResult(
        scenario_id="scenario",
        runner=runner,
        model=model,
        question="q",
        answer="a",
        score=ScorerResult(
            scorer=scorer,
            passed=False,
            details=dict(details),
        ),
        ops=OpsMetrics(),
    )


def _rates_by_model(
    report: EvalReport, runner: str
) -> dict[str, dict[str, CriterionRate]]:
    return {
        model.model: {rate.criterion: rate for rate in model.rates}
        for model in criterion_rates(report, runner).models
    }


def test_criterion_rates_use_only_applicable_boolean_results() -> None:
    report = build_report(
        [
            _result(task_completion=True),
            _result(task_completion=False),
            _result(task_completion=1),
            _result(data_retrieval_accuracy="true"),
            _result(),
        ]
    )

    rates = _rates_by_model(report, "meta-agent")["model-a"]

    assert rates["task_completion"].successes == 1
    assert rates["task_completion"].applicable == 2
    assert rates["task_completion"].percentage == 50.0
    assert rates["data_retrieval_accuracy"].applicable == 0
    assert rates["data_retrieval_accuracy"].percentage is None


def test_models_and_runners_remain_separate() -> None:
    report = build_report(
        [
            _result(model="model-a", task_completion=True),
            _result(model="model-b", task_completion=False),
            _result(runner="plan-execute", model="model-a", task_completion=False),
        ]
    )

    meta_rates = _rates_by_model(report, "meta-agent")
    plan_rates = _rates_by_model(report, "plan-execute")

    assert meta_rates["model-a"]["task_completion"].percentage == 100.0
    assert meta_rates["model-b"]["task_completion"].percentage == 0.0
    assert plan_rates["model-a"]["task_completion"].percentage == 0.0


def test_non_llm_judge_results_do_not_contaminate_rates() -> None:
    report = build_report(
        [
            _result(task_completion=True),
            _result(scorer="static_json", task_completion=False),
        ]
    )

    rate = _rates_by_model(report, "meta-agent")["model-a"]["task_completion"]

    assert (rate.successes, rate.applicable, rate.percentage) == (1, 1, 100.0)


def test_hallucinations_are_exposed_as_a_positive_outcome() -> None:
    report = build_report(
        [
            _result(hallucinations=False),
            _result(hallucinations=True),
            _result(hallucinations=False),
        ]
    )

    rate = criterion_rates(report, "meta-agent", ("hallucinations",)).models[0].rates[0]

    assert rate.successes == 2
    assert rate.applicable == 3
    assert rate.percentage == pytest.approx(66.6667)


def test_model_styles_are_stable_global_and_follow_carbon_order() -> None:
    results = [
        _result(runner="runner-b", model="z-model", task_completion=True),
        _result(runner="runner-a", model="a-model", task_completion=True),
    ]

    forward = model_styles(build_report(results))
    reversed_order = model_styles(build_report(list(reversed(results))))

    assert forward == reversed_order
    assert forward["a-model"].color == CARBON_CATEGORICAL_COLORS[0]
    assert forward["z-model"].color == CARBON_CATEGORICAL_COLORS[1]
    assert all(style.hatch for style in forward.values())


def test_styles_remain_unique_beyond_base_palette() -> None:
    report = build_report(
        [
            _result(model=f"model-{index:02}", task_completion=True)
            for index in range(len(CARBON_CATEGORICAL_COLORS) + 1)
        ]
    )

    styles = model_styles(report)

    assert len({(style.color, style.hatch) for style in styles.values()}) == len(styles)


def test_renderer_creates_svg_and_png_for_each_runner(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    original_backend = matplotlib.get_backend()
    report = build_report(
        [
            _result(runner=runner, model=model, task_completion=True)
            for runner in ("meta-agent", "plan-execute")
            for model in ("model-a", "model-b")
        ]
    )

    paths = render_leaderboards(report, tmp_path / "charts")
    repeated_paths = render_leaderboards(report, tmp_path / "charts-repeated")

    assert [path.name for path in paths] == [
        "leaderboard-meta-agent.svg",
        "leaderboard-meta-agent.png",
        "leaderboard-plan-execute.svg",
        "leaderboard-plan-execute.png",
    ]
    assert all(path.stat().st_size > 0 for path in paths)
    assert [path.read_bytes() for path in paths] == [
        path.read_bytes() for path in repeated_paths
    ]
    svg = paths[0].read_text(encoding="utf-8")
    assert 'role="img"' in svg
    assert '<desc id="chart-desc">' in svg
    assert "model-a: Task completion 100.0% (1/1)" in svg
    assert "<text" in svg
    assert matplotlib.get_backend() == original_backend


def test_renderer_uses_collision_safe_filenames(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    report = build_report(
        [
            _result(runner=runner, task_completion=True)
            for runner in ("alpha.one", "alpha.two", "Alpha", "alpha")
        ]
    )

    paths = render_leaderboards(report, tmp_path / "charts")

    assert len(paths) == 8
    assert len({path.name.casefold() for path in paths}) == 8
    assert all(path.parent == tmp_path / "charts" for path in paths)


def test_renderer_removes_only_stale_managed_outputs(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    output_dir = tmp_path / "charts"
    unrelated = output_dir / "notes.txt"
    output_dir.mkdir()
    unrelated.write_text("keep", encoding="utf-8")
    render_leaderboards(
        build_report(
            [
                _result(runner="runner-a", task_completion=True),
                _result(runner="runner-b", task_completion=True),
            ]
        ),
        output_dir,
    )

    render_leaderboards(
        build_report([_result(runner="runner-a", task_completion=True)]),
        output_dir,
    )

    assert not (output_dir / "leaderboard-runner-b.svg").exists()
    assert not (output_dir / "leaderboard-runner-b.png").exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"

    assert render_leaderboards(build_report([]), output_dir) == ()
    assert not list(output_dir.glob("leaderboard-*.svg"))
    assert not list(output_dir.glob("leaderboard-*.png"))
    assert unrelated.exists()


def test_hatches_choose_contrast_against_each_fill() -> None:
    styles = model_styles(
        build_report(
            [
                _result(model="a-model", task_completion=True),
                _result(model="b-model", task_completion=True),
            ]
        )
    )

    assert styles["a-model"].hatch_color == "#ffffff"
    assert styles["b-model"].hatch_color == "#161616"


def test_missing_visualization_dependency_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_matplotlib(name: str) -> ModuleType:
        raise ModuleNotFoundError("No module named 'matplotlib'", name="matplotlib")

    monkeypatch.setattr("evaluation.visualization.import_module", missing_matplotlib)

    with pytest.raises(
        VisualizationDependencyError, match="uv sync --group visualization"
    ):
        require_visualization_dependency()
