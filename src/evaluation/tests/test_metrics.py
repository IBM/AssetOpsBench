"""Tests for ops metrics extraction and aggregation."""

from __future__ import annotations

from evaluation.metrics import (
    _normalize_model,
    aggregate_ops,
    metrics_from_trajectory,
)
from evaluation.models import (
    OpsMetrics,
    PersistedTrajectory,
    ScenarioResult,
    ScorerResult,
)


def _result(passed: bool = True, ops: OpsMetrics | None = None) -> ScenarioResult:
    return ScenarioResult(
        scenario_id="1",
        scenario_type="iot",
        runner="plan-execute",
        model="watsonx/ibm/granite",
        question="q",
        answer="a",
        score=ScorerResult(scorer="exact_string_match", passed=passed),
        ops=ops or OpsMetrics(),
    )


class TestMetricsFromTrajectory:
    def test_sdk_trajectory_sums_per_turn(self, make_persisted_record):
        rec = PersistedTrajectory.from_raw(make_persisted_record())
        m = metrics_from_trajectory(rec)
        assert m.turn_count == 2
        assert m.tokens_in == 22
        assert m.tokens_out == 12
        assert m.tool_call_count == 1
        assert m.unique_tools == ["sites"]
        assert m.duration_ms == 300.0

    def test_handles_none_trajectory(self, make_persisted_record):
        rec = PersistedTrajectory.from_raw(make_persisted_record(trajectory=None))
        assert metrics_from_trajectory(rec) == OpsMetrics()

    def test_sdk_trajectory_falls_back_to_raw_event_tokens(self, make_persisted_record):
        rec = PersistedTrajectory.from_raw(
            make_persisted_record(
                trajectory={
                    "turns": [
                        {
                            "index": 0,
                            "text": "answer",
                            "tool_calls": [],
                            "input_tokens": 0,
                            "output_tokens": 0,
                        }
                    ],
                    "raw_events": [
                        {
                            "type": "step_finish",
                            "part": {
                                "tokens": {
                                    "input": 13084,
                                    "output": 33,
                                    "reasoning": 61,
                                    "cache": {"read": 128, "write": 2},
                                }
                            },
                        },
                        {
                            "type": "step_finish",
                            "part": {
                                "tokens": {
                                    "input": 209,
                                    "output": 84,
                                    "reasoning": 17,
                                    "cache": {"read": 13184, "write": 0},
                                }
                            },
                        },
                    ],
                }
            )
        )

        m = metrics_from_trajectory(rec)

        assert m.tokens_in == 26607
        assert m.tokens_out == 195

    def test_plan_execute_list_trajectory(self, make_persisted_record):
        rec = PersistedTrajectory.from_raw(
            make_persisted_record(
                trajectory=[
                    {"step_number": 1, "task": "t", "server": "iot", "tool": "sites", "response": "ok"},
                    {"step_number": 2, "task": "t2", "server": "iot", "tool": "assets", "response": "ok"},
                    {"step_number": 3, "task": "t3", "server": "iot", "tool": "sites", "response": "ok"},
                ]
            )
        )
        m = metrics_from_trajectory(rec)
        assert m.turn_count == 3
        assert m.tool_call_count == 3
        assert m.unique_tools == ["assets", "sites"]
        # No run-level tokens_in/tokens_out on the record (pre-fix persisted
        # files, or a runner that never set them): stay at zero, no cost.
        assert m.tokens_in == 0
        assert m.tokens_out == 0
        assert m.est_cost_usd is None

    def test_plan_execute_list_trajectory_reads_run_level_tokens(
        self, make_persisted_record
    ):
        rec = PersistedTrajectory.from_raw(
            make_persisted_record(
                model="gpt-4o",
                trajectory=[
                    {"step_number": 1, "task": "t", "server": "iot", "tool": "sites", "response": "ok"},
                ],
                tokens_in=1000,
                tokens_out=500,
            )
        )
        m = metrics_from_trajectory(rec)
        assert m.tokens_in == 1000
        assert m.tokens_out == 500
        assert m.est_cost_usd == round((1000 * 2.5 + 500 * 10.0) / 1_000_000, 6)


class TestAggregateOps:
    def test_empty(self):
        agg = aggregate_ops([])
        assert agg.tokens_in_total == 0
        assert agg.duration_ms_p50 is None

    def test_sums_and_percentiles(self):
        results = [
            _result(ops=OpsMetrics(tokens_in=10, tokens_out=5, duration_ms=100.0, tool_call_count=1)),
            _result(ops=OpsMetrics(tokens_in=20, tokens_out=10, duration_ms=300.0, tool_call_count=2)),
            _result(ops=OpsMetrics(tokens_in=30, tokens_out=15, duration_ms=500.0, tool_call_count=3)),
        ]
        agg = aggregate_ops(results)
        assert agg.tokens_in_total == 60
        assert agg.tokens_out_total == 30
        assert agg.tool_calls_total == 6
        assert agg.duration_ms_p50 is not None
        assert agg.duration_ms_p95 is not None
        assert agg.duration_ms_p50 <= agg.duration_ms_p95

    def test_cost_only_when_some_present(self):
        results = [
            _result(ops=OpsMetrics(est_cost_usd=0.01)),
            _result(ops=OpsMetrics(est_cost_usd=0.02)),
        ]
        agg = aggregate_ops(results)
        assert agg.est_cost_usd_total == 0.03

    def test_cost_components_use_model_token_rates(self):
        results = [
            ScenarioResult(
                scenario_id="1",
                scenario_type="structured",
                runner="stirrup-agent",
                model="litellm_proxy/azure/gpt-5.6-sol",
                question="q",
                answer="a",
                score=ScorerResult(scorer="static_json", passed=True),
                ops=OpsMetrics(tokens_in=1_000_000, tokens_out=100_000),
            )
        ]

        agg = aggregate_ops(results)

        assert agg.est_input_cost_usd_total == 5.0
        assert agg.est_output_cost_usd_total == 3.0
        assert agg.est_cost_usd_total == 8.0


class TestNormalizeModel:
    def test_strips_provider_prefix(self):
        assert _normalize_model("litellm_proxy/anthropic/claude-opus-4-5") == "claude-opus-4-5"
        assert _normalize_model("watsonx/ibm/granite-13b") == "granite-13b"

    def test_strips_long_numeric_suffix(self):
        assert _normalize_model("claude-opus-4-5-20250101") == "claude-opus-4-5"

    def test_keeps_short_numeric_suffix(self):
        # "4-5" suffix is the model version, not a date — leave it intact.
        assert _normalize_model("claude-opus-4-5") == "claude-opus-4-5"
