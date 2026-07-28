"""Tests for the evaluation CLI argument surface."""

from __future__ import annotations

from evaluation.cli import _build_parser, _resolve_scenario_ids


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


def test_resolve_scenario_ids_loads_all_yaml_categories() -> None:
    selected = _resolve_scenario_ids("fcc+fmsr_all")

    assert selected is not None
    assert "301" in selected
    assert "327" in selected
    assert "901" in selected
    assert "932" in selected
    assert "401" not in selected


def test_resolve_scenario_ids_loads_lite_yaml_category() -> None:
    assert _resolve_scenario_ids("fcc_lite") == {"301"}


def test_resolve_scenario_ids_is_optional() -> None:
    assert _resolve_scenario_ids(None) is None
