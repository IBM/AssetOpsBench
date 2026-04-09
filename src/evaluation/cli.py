"""CLI entry point for the skills evaluation benchmark.

Usage::

    uv run eval-skills --config configs/eval_skills.yaml
    uv run eval-skills --config configs/eval_skills.yaml --model claude-sonnet-4-6 --condition mcp_skills --runs 1
    uv run eval-skills --leaderboard results/
    uv run eval-skills --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="eval-skills",
        description="Run the AssetOpsBench skills evaluation benchmark.",
    )
    p.add_argument("--config", type=Path, help="Path to YAML config file.")
    p.add_argument("--model", type=str, help="Override: evaluate only this model.")
    p.add_argument(
        "--condition",
        type=str,
        choices=["no_skills", "mcp_only", "mcp_skills"],
        help="Override: evaluate only this condition.",
    )
    p.add_argument("--runs", type=int, help="Override: runs per scenario.")
    p.add_argument(
        "--ablation",
        type=str,
        choices=["procedure_only", "tool_list_only"],
        help="Run a single ablation variant instead of the full ablation list.",
    )
    p.add_argument(
        "--leaderboard",
        type=Path,
        help="Generate leaderboard from existing checkpoint JSON files in this dir.",
    )
    p.add_argument("--verbose", action="store_true", help="Enable INFO logging.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a self-contained demo with a bundled scenario and mock LLM. "
        "No API keys or external dependencies required.",
    )
    return p.parse_args(argv)


def _generate_leaderboard_from_dir(results_dir: Path) -> None:
    """Re-generate leaderboard tables from checkpoint files."""
    from .runner import EvalConfig, ScenarioResult, aggregate_results
    from .leaderboard import generate_leaderboard

    all_results: list[ScenarioResult] = []
    for f in sorted(results_dir.glob("checkpoint_*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        for row in data:
            all_results.append(
                ScenarioResult(
                    scenario_id=row["scenario_id"],
                    model=row["model"],
                    condition=row["condition"],
                    run=row["run"],
                    passed=row["passed"],
                    method=row["method"],
                    category=row["category"],
                )
            )
    if not all_results:
        print(f"No checkpoint files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    config = EvalConfig(output_dir=results_dir)
    metrics = aggregate_results(all_results, config)
    paths = generate_leaderboard(metrics, results_dir)
    for p in paths:
        print(f"Generated: {p}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
        )
    else:
        logging.basicConfig(level=logging.WARNING)

    # Leaderboard-only mode
    if args.leaderboard:
        _generate_leaderboard_from_dir(args.leaderboard)
        return

    # Dry-run mode — self-contained E2E demo, no API keys needed
    if args.dry_run:
        _run_dry_run()
        return

    if not args.config:
        print(
            "Error: --config is required (unless using --leaderboard).", file=sys.stderr
        )
        sys.exit(1)

    from .conditions import EvalCondition
    from .runner import (
        AblationConfig,
        EvalConfig,
        aggregate_results,
        run_full_evaluation,
    )
    from .leaderboard import generate_leaderboard

    config = EvalConfig.from_yaml(args.config)

    # Apply overrides
    if args.model:
        config.models = [args.model]
    if args.condition:
        config.conditions = [EvalCondition(args.condition)]
    if args.runs:
        config.runs_per_scenario = args.runs
    if args.ablation:
        from .ablation import SkillStructure

        config.ablations = [
            AblationConfig(
                label=args.ablation,
                structure=SkillStructure(args.ablation),
            )
        ]

    # Load scenarios
    scenarios = _load_scenarios()

    # Run evaluation
    results = asyncio.run(run_full_evaluation(scenarios, config))

    # Aggregate + leaderboard
    metrics = aggregate_results(results, config)
    paths = generate_leaderboard(metrics, config.output_dir)
    for p in paths:
        print(f"Generated: {p}")

    print(
        f"\nEvaluation complete: {len(results)} results across "
        f"{len(config.models)} model(s) × {len(config.conditions)} condition(s)."
    )


def _run_dry_run() -> None:
    """Self-contained E2E demonstration with a bundled scenario and mock data.

    Shows the full pipeline — skill loading → prompt construction → scoring →
    leaderboard — without requiring LLM API keys, CouchDB, or HuggingFace.
    """
    from pathlib import Path as _Path

    from .ablation import SkillStructure
    from .leaderboard import generate_leaderboard
    from .metrics import compute_normalized_gain, compute_pass_rate
    from .runner import EvalConfig, ScenarioResult, aggregate_results
    from .scorer import score_result
    from .skill_loader import (
        load_skills,
        load_skills_as_text,
        select_skills_for_scenario,
    )

    skills_dir = _Path("src/skills/generated")

    # ── 1. Load skills ─────────────────────────────────────────────────
    skills = load_skills(skills_dir)
    print("=" * 72)
    print("STEP 1 — Load skills")
    print("=" * 72)
    print(f"  Skills directory : {skills_dir}")
    print(f"  Skills loaded    : {len(skills)}")
    for s in skills:
        print(
            f"    • {s.get('name')} (domain={s.get('domain')}, level={s.get('level')})"
        )

    # ── 2. Bundled example scenario ────────────────────────────────────
    print()
    print("=" * 72)
    print("STEP 2 — Example scenario")
    print("=" * 72)

    class _FakeScenario:
        id = "dry-run-1"
        text = (
            "Diagnose the vibration on asset PUMP-101 at site MAIN. "
            "The bearing designation is 6205 and the shaft speed is 1480 RPM. "
            "Provide an ISO 10816 severity assessment."
        )
        type = "vibration"
        category = "diagnostics_reasoning"
        deterministic = True
        expected_result = "Zone B"
        characteristic_form = "ISO 10816 severity zone"
        data = {}
        source = "dry-run"

    scenario = _FakeScenario()
    print(f"  Scenario ID : {scenario.id}")
    print(f"  Type        : {scenario.type}")
    print(f"  Question    : {scenario.text}")
    print(f"  Expected    : {scenario.expected_result}")

    # ── 3. Skill selection ─────────────────────────────────────────────
    print()
    print("=" * 72)
    print("STEP 3 — Skill selection for scenario")
    print("=" * 72)
    selected = select_skills_for_scenario(scenario.text, skills, top_k=2)
    for s in selected:
        print(f"    → {s.get('name')} (keywords matched)")

    # ── 4. Planning prompt construction ────────────────────────────────
    print()
    print("=" * 72)
    print("STEP 4 — Planning prompt (MCP+Skills condition)")
    print("=" * 72)
    full_text = load_skills_as_text(skills_dir)
    proc_text = load_skills_as_text(skills_dir, structure=SkillStructure.PROCEDURE_ONLY)
    tool_text = load_skills_as_text(skills_dir, structure=SkillStructure.TOOL_LIST_ONLY)
    print(f"  Full skills prompt     : {len(full_text):,} chars")
    print(f"  Procedure-only (abl.)  : {len(proc_text):,} chars")
    print(f"  Tool-list-only (abl.)  : {len(tool_text):,} chars")
    print()
    print("  Prompt preview (first 500 chars of full):")
    print("  " + "-" * 50)
    for line in full_text[:500].splitlines():
        print(f"  | {line}")
    print("  " + "-" * 50)

    # ── 5. Simulated scoring (3 conditions) ────────────────────────────
    print()
    print("=" * 72)
    print("STEP 5 — Simulated scoring across 3 conditions")
    print("=" * 72)

    mock_results: list[ScenarioResult] = []
    conditions_sim = [
        ("no_skills", "Zone C", False),  # wrong — no skill guidance
        ("mcp_only", "Zone B", True),  # correct but lucky
        ("mcp_skills", "Zone B", True),  # correct — skill-guided
    ]
    for cond, answer, passed in conditions_sim:
        # Use deterministic scorer
        score = score_result(scenario, answer, trace="", judge_model_id=16)
        result = ScenarioResult(
            scenario_id=scenario.id,
            model="dry-run-model",
            condition=cond,
            run=1,
            passed=score["passed"],
            method=score["method"],
            category="diagnostics_reasoning",
            answer=answer,
        )
        mock_results.append(result)
        print(
            f"  {cond:12s} → answer={answer!r:10s}  "
            f"passed={score['passed']}  method={score['method']}"
        )

    # ── 6. Metrics ─────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("STEP 6 — Metrics")
    print("=" * 72)
    for cond in ("no_skills", "mcp_only", "mcp_skills"):
        vals = [r.passed for r in mock_results if r.condition == cond]
        pr = compute_pass_rate(vals)
        print(f"  {cond:12s}  pass_rate = {pr:.0%}")
    no_skills_vals = [r.passed for r in mock_results if r.condition == "no_skills"]
    mcp_skills_vals = [r.passed for r in mock_results if r.condition == "mcp_skills"]
    gain = compute_normalized_gain(
        compute_pass_rate(mcp_skills_vals), compute_pass_rate(no_skills_vals)
    )
    print(f"  Normalized gain (MCP+Skills vs No-Skills) = {gain:.2f}")

    # ── 7. Leaderboard generation ──────────────────────────────────────
    print()
    print("=" * 72)
    print("STEP 7 — Leaderboard generation")
    print("=" * 72)
    output_dir = _Path("results/dry-run")
    config = EvalConfig(output_dir=output_dir)
    metrics = aggregate_results(mock_results, config)
    paths = generate_leaderboard(metrics, output_dir)
    for p in paths:
        print(f"  Generated: {p}")

    print()
    print("=" * 72)
    print("DRY RUN COMPLETE — full pipeline traced without any API calls.")
    print("=" * 72)


def _load_scenarios() -> list:
    """Load scenarios from HuggingFace dataset."""
    try:
        from tmp.assetopsbench.core.scenarios import Scenario

        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="ibm-research/AssetOpsBench",
            filename="data/scenarios/all_utterance.jsonl",
            repo_type="dataset",
        )
        scenarios = []
        for line in Path(path).read_text(encoding="utf-8").strip().splitlines():
            data = json.loads(line)
            data["id"] = str(data["id"])  # JSONL has int ids; Scenario expects str
            scenarios.append(Scenario(**data))
        return scenarios
    except Exception as exc:
        print(f"Warning: Could not load HuggingFace scenarios: {exc}", file=sys.stderr)
        print(
            "Provide scenarios via --scenarios or fix HuggingFace access.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
