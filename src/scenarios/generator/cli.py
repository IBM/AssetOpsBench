"""CLI entry for `python -m scenarios.generator`."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

from dotenv import load_dotenv

from agent.cli import _DEFAULT_MODEL

from .agent import ScenarioGeneratorAgent
from .prompt_helpers import default_scenario_output_path, negative_scenario_output_path


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Scenario Generator for AssetOpsBench")
    parser.add_argument("asset_name", help="Asset class name (for example 'Chiller' or 'Wind Turbine')")
    parser.add_argument("--model-id", default=_DEFAULT_MODEL, help="Model ID for LiteLLM")
    parser.add_argument("--num-scenarios", type=int, default=50, help="Total number of scenarios to generate")
    parser.add_argument(
        "--num-negative-scenarios",
        type=int,
        default=2,
        help="Total number of intentionally unanswerable negative scenarios to generate (0 disables the negative track)",
    )
    parser.add_argument("--show-workflow", action="store_true", help="Show intermediate pipeline steps in the console")
    parser.add_argument(
        "--log",
        action="store_true",
        help=(
            "Write prompts and pipeline artifacts under the run directory (generated/scenarios/.../logs/). "
            "Does not change console output; use --show-workflow for step-by-step terminal output."
        ),
    )
    parser.add_argument(
        "--data-in-couchdb",
        action="store_true",
        help="Use grounded open-form generation when matching live CouchDB-backed asset data is available",
    )
    parser.add_argument(
        "--retriever",
        choices=("arxiv", "semantic_scholar"),
        default="arxiv",
        help=(
            "Optional. Academic search backend for Phase 1 evidence retrieval. "
            "Default is arxiv when this flag is omitted."
        ),
    )
    parser.add_argument(
        "--research-digest",
        metavar="PATH",
        default=None,
        help=(
            "Optional. Path to a precomputed research digest (Markdown). When set and the file exists, "
            "skips academic retrieval and digest LLM steps; loads this text for asset profile construction."
        ),
    )

    args = parser.parse_args()
    output_path = default_scenario_output_path(args.asset_name)

    # stderr: `--show-workflow` owns the console (`_print_step`); keep root at
    # WARNING so `_log.info(...)` from agent/grounding/retrieval never mixes in.
    # Without `--show-workflow`, `--data-in-couchdb` raises to INFO so
    # `scenarios.grounding` can emit FMSR/discovery progress on stderr.
    if args.show_workflow:
        level = logging.WARNING
    elif args.data_in_couchdb:
        level = logging.INFO
    else:
        level = logging.WARNING
    # Imported MCP server modules (e.g. servers.fmsr.main) call basicConfig(WARNING)
    # at import time; without force=True, this CLI's basicConfig would be a no-op.
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    if level == logging.INFO:
        for noisy in ("httpx", "httpcore"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    log_dir = None
    if args.log:
        log_dir = str(output_path.parent / "logs")
        os.makedirs(log_dir, exist_ok=True)

    agent = ScenarioGeneratorAgent(
        model_id=args.model_id,
        show_workflow=args.show_workflow,
        log_dir=log_dir,
        retriever=args.retriever,
        research_digest_path=args.research_digest,
    )

    try:
        generation_result = asyncio.run(
            agent.run(
                args.asset_name,
                num_scenarios=args.num_scenarios,
                data_in_couchdb=args.data_in_couchdb,
                num_negative_scenarios=args.num_negative_scenarios,
            )
        )
    except Exception as exc:  # noqa: BLE001
        print(f"\n[FATAL ERROR] {exc}")
        sys.exit(1)

    final_scenarios = generation_result.scenarios
    negative_scenarios = generation_result.negative_scenarios

    if not final_scenarios:
        print("\n[WARNING] No scenarios were successfully generated and validated.")
        sys.exit(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump([scenario.to_dict() for scenario in final_scenarios], handle, indent=2)

    if negative_scenarios:
        negative_output_path = negative_scenario_output_path(output_path)
        with negative_output_path.open("w") as handle:
            json.dump([scenario.to_dict() for scenario in negative_scenarios], handle, indent=2)

    if not args.show_workflow:
        suffix = (
            f" and {len(negative_scenarios)} negative scenarios at {negative_scenario_output_path(output_path)}"
            if negative_scenarios
            else ""
        )
        print(f"Success! Generated {len(final_scenarios)} scenarios at {output_path}{suffix}")
    else:
        print(f"Scenarios saved to {output_path}")
