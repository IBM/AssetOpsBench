"""Unit tests for the adaptive-escalation experiment harness."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

from agent.plan_execute.escalation import extract_escalation_signals, should_escalate
from agent.plan_execute.models import Plan, PlanStep


def _load_experiment_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "benchmarks"
        / "adaptive_escalation_experiment.py"
    )
    spec = importlib.util.spec_from_file_location("adaptive_escalation_experiment", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


experiment = _load_experiment_module()


def test_live_harness_defines_all_four_comparison_conditions():
    assert experiment.CONDITIONS == ("baseline", "original", "redesigned", "always")


def test_original_and_redesigned_routing_are_technically_distinct():
    plan = Plan(
        steps=[
            PlanStep(1, "Routine read", "wo", "list_workorders", {}, [], "records")
        ],
        raw="",
    )
    signals = extract_escalation_signals("List work orders", plan, [])

    assert experiment._original_policy(signals).action.value == "verify"
    assert should_escalate(signals).action.value == "none"


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("You are reviewing the evidence gathered by an agent", "verification"),
        ("You are summarizing the results", "summarization"),
        ("Generate the JSON arguments for this tool", "argument_resolution"),
        ("Create a plan", "planning"),
    ],
)
def test_classify_prompt(prompt, expected):
    assert experiment.classify_prompt(prompt) == expected


def test_parse_scenario_ids_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicates"):
        experiment.parse_scenario_ids("1,2,1")


def test_parse_conditions_supports_a_gated_single_condition_run():
    assert experiment.parse_conditions("redesigned") == ["redesigned"]


@pytest.mark.parametrize("raw", ["", "redesigned,redesigned", "experimental"])
def test_parse_conditions_rejects_invalid_selections(raw):
    with pytest.raises(ValueError, match="--conditions"):
        experiment.parse_conditions(raw)


def test_recorded_decision_matches_each_experimental_condition():
    plan = Plan(
        steps=[
            PlanStep(1, "Routine read", "wo", "list_workorders", {}, [], "records")
        ],
        raw="",
    )
    signals = extract_escalation_signals("List work orders", plan, [])

    assert experiment._condition_decision("baseline", signals).action.value == "none"
    assert experiment._condition_decision("original", signals).action.value == "verify"
    assert experiment._condition_decision("redesigned", signals).action.value == "none"
    assert experiment._condition_decision("always", signals).action.value == "verify"


def test_external_llm_acknowledgement_is_explicit():
    parser = experiment.build_parser()
    base = ["--scenario-root", "scenarios", "--output-dir", "results"]

    assert not parser.parse_args(base).acknowledge_external_llm
    acknowledged = parser.parse_args(base + ["--acknowledge-external-llm"])
    assert acknowledged.acknowledge_external_llm


@pytest.mark.anyio
async def test_live_run_refuses_external_calls_without_acknowledgement():
    with pytest.raises(ValueError, match="--acknowledge-external-llm"):
        await experiment.run_experiment(
            argparse.Namespace(acknowledge_external_llm=False)
        )


def test_output_labels_match_the_metrics_the_harness_measures(tmp_path):
    record = {
        "scenario_id": "1",
        "condition": "redesigned",
        "passed": True,
        "score": 1.0,
        "input_tokens": 10,
        "output_tokens": 2,
        "total_tokens": 12,
        "llm_calls": 1,
        "estimated_cost_usd": 0.01,
        "duration_ms": 5.0,
        "routed_to_verification": False,
        "routed_to_recovery": True,
        "tool_steps": 2,
        "execution_duration_ms": 3.0,
    }

    summary = experiment.write_outputs([record], tmp_path, ["redesigned"])
    csv_text = (tmp_path / "runs.csv").read_text(encoding="utf-8")

    assert summary["comparison_status"] == "partial_mechanism_check"
    assert "not causal" in summary["interpretation_warning"]
    assert "tool_steps" in csv_text
    assert "execution_duration_ms" in csv_text
    assert "tool_calls" not in csv_text


def test_estimate_cost_handles_full_watsonx_llama_model_id():
    estimate = experiment.estimate_cost(
        "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        1_000_000,
        1_000_000,
    )

    assert estimate == 1.12
