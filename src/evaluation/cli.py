"""CLI entry point for the skills evaluation benchmark.

Usage::

    uv run eval-skills --generate-skills --llm-model watsonx/meta-llama/llama-3-3-70b-instruct
    uv run eval-skills --config configs/eval_skills.yaml
    uv run eval-skills --config configs/eval_skills.yaml --model claude-sonnet-4-6 --condition mcp_skills --runs 1
    uv run eval-skills --refine-skills results/checkpoint_001.json --llm-model claude-sonnet-4-6
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
    p.add_argument(
        "--generate-skills",
        action="store_true",
        help="Auto-generate SKILL.md files for all domains using an LLM. "
        "Requires --llm-model and API credentials in the environment.",
    )
    p.add_argument(
        "--refine-skills",
        type=Path,
        metavar="CHECKPOINT",
        help="Refine skills via GEPA using failed traces from a checkpoint JSON file. "
        "Requires dspy (uv add --group gepa dspy).",
    )
    p.add_argument(
        "--llm-model",
        type=str,
        default="watsonx/meta-llama/llama-3-3-70b-instruct",
        help="LLM model id for skill generation / GEPA refinement (litellm format). "
        "Default: watsonx/meta-llama/llama-3-3-70b-instruct",
    )
    p.add_argument(
        "--skills-dir",
        type=Path,
        default=Path("src/skills/generated"),
        help="Output directory for generated skills. Default: src/skills/generated",
    )
    p.add_argument(
        "--gepa-output-dir",
        type=Path,
        default=Path("src/skills/gepa-refined"),
        help="Output directory for GEPA-refined skills. Default: src/skills/gepa-refined",
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

    # Skill generation mode
    if args.generate_skills:
        _run_generate_skills(args.llm_model, args.skills_dir)
        return

    # GEPA refinement mode
    if args.refine_skills:
        _run_refine_skills(args.refine_skills, args.llm_model, args.skills_dir, args.gepa_output_dir)
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


# ---------------------------------------------------------------------------
# Server descriptions for skill generation
# ---------------------------------------------------------------------------

# Static map of MCP server names to their tool descriptions.  Built from
# the actual ``@mcp.tool`` decorated functions in ``src/servers/``.
_SERVER_DESCRIPTIONS: dict[str, str] = {
    "Vibration": (
        "Tools: get_vibration_data, list_vibration_sensors, compute_fft_spectrum, "
        "compute_envelope_spectrum, assess_vibration_severity, "
        "calculate_bearing_frequencies, list_known_bearings, diagnose_vibration.\n"
        "Domain: vibration analysis for rotating machinery — bearing fault "
        "detection, FFT / envelope spectrum analysis, ISO 10816 severity."
    ),
    "IoT": (
        "Tools: sites, assets, sensors, history.\n"
        "Domain: IoT sensor monitoring — list sites/assets/sensors, retrieve "
        "historical sensor readings."
    ),
    "FMSR": (
        "Tools: get_failure_modes, get_failure_mode_sensor_mapping.\n"
        "Domain: Failure Mode and Symptom Reasoning — FMEA-based root cause "
        "analysis, sensor-to-failure correlations."
    ),
    "TSFM": (
        "Tools: get_ai_tasks, get_tsfm_models, run_tsfm_forecasting, "
        "run_tsfm_finetuning, run_tsad, run_integrated_tsad.\n"
        "Domain: Time Series Foundation Models — anomaly detection, trend "
        "forecasting, predictive maintenance."
    ),
    "WO": (
        "Tools: get_work_orders, get_preventive_work_orders, "
        "get_corrective_work_orders, get_events, get_failure_codes, "
        "get_work_order_distribution, predict_next_work_order, "
        "analyze_alert_to_failure.\n"
        "Domain: Work order management — maintenance history, failure codes, "
        "alert-to-failure analysis, work order prediction."
    ),
    "Utilities": (
        "Tools: json_reader, current_date_time, current_time_english.\n"
        "Domain: general-purpose helpers — read JSON files, get timestamps."
    ),
}


def _run_generate_skills(llm_model: str, skills_dir: Path) -> None:
    """Generate SKILL.md files for all domains using an LLM.

    Calls ``generate_all_skills()`` from ``skill_generator`` and writes
    the results to *skills_dir*.  Updates ``src/skills/manifest.json``.
    """
    from llm.litellm import LiteLLMBackend

    from .skill_generator import generate_all_skills

    print(f"Generating skills with model: {llm_model}")
    print(f"Output directory: {skills_dir}")
    print()

    llm = LiteLLMBackend(llm_model)
    all_paths = generate_all_skills(llm, _SERVER_DESCRIPTIONS, skills_dir)

    total = sum(len(ps) for ps in all_paths.values())
    print()
    print(f"Generated {total} skills across {len(all_paths)} domains:")
    for domain, paths in all_paths.items():
        for p in paths:
            print(f"  {domain}: {p}")
    print()
    print("Manifest updated: src/skills/manifest.json")
    print(
        "Run 'uv run eval-skills --dry-run' to verify, or "
        "'uv run eval-skills --config configs/eval_skills.yaml' for full evaluation."
    )


def _run_refine_skills(
    checkpoint: Path, llm_model: str, skills_dir: Path, output_dir: Path
) -> None:
    """Refine auto-generated skills using GEPA (Guided Exploration with Predictive Auditing).

    Reads failed traces from a checkpoint JSON file, groups them by category,
    and runs DSPy-based optimisation to produce improved SKILL.md files.
    """
    from .gepa_refiner import run_gepa_refinement

    if not checkpoint.exists():
        print(f"Error: checkpoint file not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(checkpoint.read_text(encoding="utf-8"))

    # Group failed traces by category
    failed_by_category: dict[str, list[dict]] = {}
    for row in data:
        if not row.get("passed", True):
            cat = row.get("category", "unknown")
            failed_by_category.setdefault(cat, []).append(
                {
                    "scenario_id": row["scenario_id"],
                    "trace": row.get("trace", ""),
                    "expected_result": row.get("expected_result", ""),
                }
            )

    if not failed_by_category:
        print("No failed traces found in checkpoint -- nothing to refine.")
        return

    print(f"GEPA refinement with model: {llm_model}")
    print(f"Seed skills directory: {skills_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Failed categories: {', '.join(failed_by_category.keys())}")
    print()

    for cat, traces in failed_by_category.items():
        # Find seed skills matching this category
        seed_skills = list(skills_dir.rglob("SKILL.md"))
        if not seed_skills:
            print(f"  Skipping {cat}: no seed skills found in {skills_dir}")
            continue

        print(f"  Refining {len(seed_skills)} skill(s) for category '{cat}' "
              f"using {len(traces)} failed trace(s)...")
        refined = asyncio.run(
            run_gepa_refinement(
                cluster_name=cat,
                seed_skills=seed_skills,
                failed_traces=traces,
                llm_model=llm_model,
                output_dir=output_dir,
            )
        )
        for p in refined:
            print(f"    Refined: {p}")

    print()
    print("GEPA refinement complete.")
    print(
        "Run 'uv run eval-skills --config configs/eval_skills.yaml "
        f"--skills-dir {output_dir}' to evaluate refined skills."
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
    print("STEP 1 - Load skills")
    print("=" * 72)
    print(f"  Skills directory : {skills_dir}")
    print(f"  Skills loaded    : {len(skills)}")
    for s in skills:
        print(
            f"    - {s.get('name')} (domain={s.get('domain')}, level={s.get('level')})"
        )

    # ── 2. Bundled example scenario ────────────────────────────────────
    print()
    print("=" * 72)
    print("STEP 2 - Example scenario")
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
    print("STEP 3 - Skill selection for scenario")
    print("=" * 72)
    selected = select_skills_for_scenario(scenario.text, skills, top_k=2)
    for s in selected:
        print(f"    > {s.get('name')} (keywords matched)")

    # ── 4. Planning prompt construction ────────────────────────────────
    print()
    print("=" * 72)
    print("STEP 4 - Planning prompt (MCP+Skills condition)")
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
    print("STEP 5 - Simulated scoring across 3 conditions")
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
            f"  {cond:12s} > answer={answer!r:10s}  "
            f"passed={score['passed']}  method={score['method']}"
        )

    # ── 6. Metrics ─────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("STEP 6 - Metrics")
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
    print("STEP 7 - Leaderboard generation")
    print("=" * 72)
    output_dir = _Path("results/dry-run")
    config = EvalConfig(output_dir=output_dir)
    metrics = aggregate_results(mock_results, config)
    paths = generate_leaderboard(metrics, output_dir)
    for p in paths:
        print(f"  Generated: {p}")

    print()
    print("=" * 72)
    print("DRY RUN COMPLETE - full pipeline traced without any API calls.")
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
