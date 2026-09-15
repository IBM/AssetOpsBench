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


def test_resolve_scenario_ids_loads_fmea_category() -> None:
    assert _resolve_scenario_ids("fmea_all") == {
        "9001", "9003", "9005", "9007", "9009", "9011",
        "9013", "9015", "9017", "9019", "9021", "9023",
        "9025", "9027", "9029", "9031", "9033",
    }
