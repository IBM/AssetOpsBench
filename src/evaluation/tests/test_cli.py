"""Tests for the evaluation CLI argument surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation import cli
from evaluation.cli import _build_parser, _resolve_scenario_ids
from evaluation.models import EvalReport
from evaluation.report import build_report
from evaluation.visualization import VisualizationDependencyError


def test_cli_accepts_optional_scenario_selector() -> None:
    args = _build_parser().parse_args(
        [
            "--trajectories",
            "trajectories",
            "--scenarios",
            "scenarios",
            "--scenario-ids",
            "fcc+fmsr_all",
        ]
    )

    assert args.scenario_ids == "fcc+fmsr_all"


def test_cli_charts_are_opt_in() -> None:
    base_args = [
        "--trajectories",
        "trajectories",
        "--scenarios",
        "scenarios",
    ]

    assert _build_parser().parse_args(base_args).charts is False
    assert _build_parser().parse_args([*base_args, "--charts"]).charts is True


@pytest.mark.parametrize("charts", [False, True])
def test_cli_chart_generation_is_opt_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, charts: bool
) -> None:
    calls: list[Path] = []

    class StubEvaluator:
        def __init__(self, **kwargs: object) -> None:
            pass

        def evaluate(self, **kwargs: object) -> EvalReport:
            return build_report([])

    def write_reports_dir(report: EvalReport, path: Path) -> Path:
        return path

    def record_render(report: EvalReport, path: Path) -> tuple[Path, ...]:
        calls.append(path)
        return ()

    def accept_scorer(name: str) -> None:
        pass

    def dependency_available() -> None:
        pass

    monkeypatch.setattr(cli, "Evaluator", StubEvaluator)
    monkeypatch.setattr(cli, "_validate_scorer_default", accept_scorer)
    monkeypatch.setattr(cli, "write_reports_dir", write_reports_dir)
    monkeypatch.setattr(
        "evaluation.visualization.require_visualization_dependency",
        dependency_available,
    )
    monkeypatch.setattr(
        "evaluation.visualization.render_leaderboards",
        record_render,
    )

    args = [
        "--trajectories",
        "trajectories",
        "--scenarios",
        "scenarios",
        "--reports-dir",
        str(tmp_path),
    ]
    if charts:
        args.append("--charts")

    result = cli.main(args)

    assert result == 0
    assert calls == ([tmp_path / "charts"] if charts else [])


def test_cli_fails_before_evaluation_when_chart_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def dependency_error() -> None:
        raise VisualizationDependencyError("install charts")

    monkeypatch.setattr(
        "evaluation.visualization.require_visualization_dependency",
        dependency_error,
    )
    monkeypatch.setattr(
        cli,
        "Evaluator",
        lambda **kwargs: pytest.fail("evaluation should not run"),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "--trajectories",
                "trajectories",
                "--scenarios",
                "scenarios",
                "--charts",
            ]
        )

    assert exc_info.value.code == 2


def test_resolve_scenario_ids_loads_all_yaml_categories() -> None:
    selected = _resolve_scenario_ids("fcc+fmsr_all")

    assert selected is not None
    assert "301" in selected
    assert "327" in selected
    assert "901" in selected
    assert "932" in selected
    assert "401" not in selected


def test_resolve_scenario_ids_loads_lite_yaml_category() -> None:
    assert _resolve_scenario_ids("fcc_lite") == {
        "301",
        "303",
        "305",
        "308",
        "314",
        "316",
        "320",
        "323",
        "325",
        "327",
    }


def test_resolve_scenario_ids_is_optional() -> None:
    assert _resolve_scenario_ids(None) is None
