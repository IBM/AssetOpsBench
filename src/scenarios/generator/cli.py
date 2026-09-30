"""CLI entry for `python -m scenarios.generator`."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv
from pydantic import ValidationError

from ..backends import BACKENDS
from ..config import GeneratorConfig
from ..planning import ScenarioCounts, ScenarioPlan
from .service import generate_scenarios


def build_parser() -> argparse.ArgumentParser:
    defaults = GeneratorConfig.model_fields
    parser = argparse.ArgumentParser(
        description="Scenario Generator for AssetOpsBench",
        epilog=(
            "Lowercase names in angle brackets, such as <asset-class> and <file>, are placeholders to replace with your values. "
            "A value shown after an option is required when that option is used. "
            "Square brackets mark optional arguments; | separates alternatives. Do not type these markers or the angle brackets. "
            "Quote values containing spaces and wrap JSON objects in single quotes."
        ),
    )
    parser.add_argument("asset_name", metavar="<asset-class>", help="Asset class name (for example 'Chiller' or 'Wind Turbine')")
    parser.add_argument(
        "--backend", metavar="<backend>", choices=tuple(BACKENDS), default=defaults["backend"].default,
        help="Generation and repair backend (default: litellm; claude-code/codex use local CLI auth; glm uses ZAI_API_KEY)",
    )
    parser.add_argument(
        "--model-id", metavar="<model>", default=None,
        help="Model ID; defaults: " + "; ".join(
            f"{key}={spec.default_model or 'CLI default'}" for key, spec in BACKENDS.items()
        ),
    )
    counts = parser.add_mutually_exclusive_group()
    counts.add_argument(
        "--scenario-plan", metavar="<json>",
        help="JSON object with positive/negative counts for each focus",
    )
    counts.add_argument(
        "--scenario-counts", metavar="<json>",
        help='JSON positive/negative totals to distribute across focuses (default: {"positive":50,"negative":2})',
    )
    parser.add_argument(
        "--batch-size", metavar="<count>", type=int, default=defaults["batch_size"].default,
        help="Maximum positive scenarios per generation/repair batch (default: 5)",
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
        "--mode", metavar="<mode>", choices=("open", "closed"),
        default=defaults["mode"].default,
        help="open requires matching live asset inventory; closed generates self-contained scenarios (default: closed)",
    )
    parser.add_argument(
        "--retriever",
        metavar="<provider>",
        choices=("arxiv", "semantic_scholar"),
        default=defaults["retriever"].default,
        help=(
            "Academic search provider: arxiv or semantic_scholar (default: arxiv)"
        ),
    )
    research = parser.add_mutually_exclusive_group()
    research.add_argument(
        "--reuse-research", dest="research_file", metavar="<file>",
        help=(
            "Use a saved research brief (Markdown or text), skipping paper retrieval and research "
            "synthesis. Without this option, gather fresh research with --retriever"
        ),
    )
    research.add_argument(
        "--research-digest", dest="research_file", metavar="<file>", help=argparse.SUPPRESS,
    )

    return parser


def main() -> None:
    load_dotenv()
    parser = build_parser()
    values = vars(parser.parse_args())
    try:
        for field, schema in (("scenario_plan", ScenarioPlan), ("scenario_counts", ScenarioCounts)):
            if values[field] is not None:
                values[field] = schema.model_validate_json(values[field])
        config = GeneratorConfig.model_validate(values)
    except ValidationError as exc:
        parser.error(str(exc))

    # stderr: `--show-workflow` owns the console (`_print_step`); keep root at
    # WARNING so `_log.info(...)` from agent/grounding/retrieval never mixes in.
    # Without `--show-workflow`, `--mode open` raises to INFO so
    # `scenarios.grounding` can emit FMSR/discovery progress on stderr.
    if config.show_workflow:
        level = logging.WARNING
    elif config.live_data:
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

    try:
        run = asyncio.run(generate_scenarios(config))
    except Exception as exc:  # noqa: BLE001
        print(f"\n[FATAL ERROR] {exc}")
        sys.exit(1)

    print(f"Generated {len(run.result.scenarios)} scenarios at {run.output_path}")
    if run.negative_output_path:
        print(f"Generated {len(run.result.negative_scenarios)} negative scenarios at {run.negative_output_path}")
    if not run.complete:
        print("[WARNING] The run did not reach the requested counts; partial results and logs are saved.")
        sys.exit(1)
