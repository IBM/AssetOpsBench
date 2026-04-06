"""CLI entry point for the ClaudeAgentRunner.

Usage:
    claude-agent "What sensors are on Chiller 6?"
    claude-agent --model-id claude-opus-4-6 --max-turns 20 "List failure modes for pumps"
    claude-agent --json "What is the current time?"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

_DEFAULT_MODEL = "claude-opus-4-6"
_LITELLM_PREFIX = "litellm_proxy/"
_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_LOG_DATE_FORMAT = "%H:%M:%S"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="claude-agent",
        description="Run a question through the Claude Agent SDK with AssetOpsBench MCP servers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
environment variables:
  ANTHROPIC_API_KEY     Anthropic API key (direct) or LiteLLM key (proxy).
                        Falls back to LITELLM_API_KEY when --model-id starts
                        with "litellm_proxy/".
  ANTHROPIC_BASE_URL    LiteLLM proxy URL (e.g. http://localhost:4000).
                        Falls back to LITELLM_BASE_URL when --model-id starts
                        with "litellm_proxy/".

examples:
  claude-agent "What assets are at site MAIN?"
  claude-agent --model-id claude-opus-4-6 --max-turns 20 "List sensors on Chiller 6"
  claude-agent --model-id litellm_proxy/aws/claude-opus-4-6 "What is the current time?"
  claude-agent --json "What is the current time?"
""",
    )
    parser.add_argument("question", help="The question to answer.")
    parser.add_argument(
        "--model-id",
        default=_DEFAULT_MODEL,
        metavar="MODEL_ID",
        help=f"Claude model ID (default: {_DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=30,
        metavar="N",
        help="Maximum agentic loop turns (default: 30).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output the full result as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show INFO-level logs on stderr.",
    )
    return parser


def _apply_litellm_env(model_id: str) -> None:
    """When model_id has the litellm_proxy/ prefix, populate ANTHROPIC_* env
    vars from their LITELLM_* counterparts so claude-agent-sdk routes through
    the proxy without extra manual configuration."""
    if not model_id.startswith(_LITELLM_PREFIX):
        return
    if not os.environ.get("ANTHROPIC_BASE_URL") and os.environ.get("LITELLM_BASE_URL"):
        os.environ["ANTHROPIC_BASE_URL"] = os.environ["LITELLM_BASE_URL"]
    if not os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("LITELLM_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = os.environ["LITELLM_API_KEY"]


def _setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
    logging.root.handlers.clear()
    logging.root.addHandler(handler)
    logging.root.setLevel(level)


async def _run(args: argparse.Namespace) -> None:
    from agent.claude_agent.runner import ClaudeAgentRunner

    runner = ClaudeAgentRunner(model=args.model_id, max_turns=args.max_turns)
    result = await runner.run(args.question)

    if args.output_json:
        output = {
            "question": result.question,
            "answer": result.answer,
            "history": [
                {
                    "step": r.step_number,
                    "task": r.task,
                    "server": r.server,
                    "response": r.response,
                    "error": r.error,
                    "success": r.success,
                }
                for r in result.history
            ],
        }
        print(json.dumps(output, indent=2))
        return

    print(result.answer)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    args = _build_parser().parse_args()
    _apply_litellm_env(args.model_id)
    _setup_logging(args.verbose)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
