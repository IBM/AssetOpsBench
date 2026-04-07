"""Tests for evaluation.runner — aggregate and config logic."""

from __future__ import annotations


from evaluation.runner import EvalConfig, ScenarioResult, aggregate_results


def _result(
    scenario_id: str = "s1",
    model: str = "test-model",
    condition: str = "mcp_only",
    passed: bool = True,
    category: str = "diagnostics_reasoning",
    run: int = 1,
) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=scenario_id,
        model=model,
        condition=condition,
        run=run,
        passed=passed,
        method="exact",
        category=category,
    )


def test_aggregate_pass_rate():
    results = [
        _result(passed=True, condition="mcp_only"),
        _result(passed=False, condition="mcp_only"),
        _result(passed=True, condition="mcp_only"),
    ]
    config = EvalConfig()
    agg = aggregate_results(results, config)
    pr = agg["test-model"]["mcp_only"]["pass_rate"]
    assert abs(pr - 2 / 3) < 1e-9


def test_aggregate_normalized_gain():
    results = [
        # MCP_ONLY: 1/2
        _result(passed=True, condition="mcp_only"),
        _result(passed=False, condition="mcp_only"),
        # MCP_SKILLS: 2/2
        _result(passed=True, condition="mcp_skills"),
        _result(passed=True, condition="mcp_skills"),
    ]
    config = EvalConfig()
    agg = aggregate_results(results, config)
    gain = agg["test-model"]["normalized_gain"]
    # g = (1.0 - 0.5) / (1.0 - 0.5) = 1.0
    assert abs(gain - 1.0) < 1e-9


def test_aggregate_per_category():
    results = [
        _result(category="diagnostics_reasoning", condition="mcp_skills", passed=True),
        _result(category="diagnostics_reasoning", condition="mcp_skills", passed=False),
        _result(category="maintenance_planning", condition="mcp_skills", passed=True),
    ]
    config = EvalConfig()
    agg = aggregate_results(results, config)
    cats = agg["test-model"]["per_category"]
    assert "diagnostics_reasoning" in cats
    assert abs(cats["diagnostics_reasoning"]["mcp_skills"]["pass_rate"] - 0.5) < 1e-9
    assert abs(cats["maintenance_planning"]["mcp_skills"]["pass_rate"] - 1.0) < 1e-9


def test_eval_config_from_yaml(tmp_path):
    cfg_file = tmp_path / "test_config.yaml"
    cfg_file.write_text(
        "evaluation:\n"
        "  models:\n"
        "    - test-model\n"
        "  conditions:\n"
        "    - no_skills\n"
        "    - mcp_only\n"
        "  runs_per_scenario: 3\n"
        "  output_dir: out\n",
        encoding="utf-8",
    )
    config = EvalConfig.from_yaml(cfg_file)
    assert config.models == ["test-model"]
    assert len(config.conditions) == 2
    assert config.runs_per_scenario == 3
