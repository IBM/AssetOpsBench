"""CLI entry point for the ClaudeAgentRunner.

Usage:
    claude-agent "What sensors are on Chiller 6?"
    claude-agent --model-id claude-opus-4-6 --max-turns 20 "List failure modes for pumps"
    claude-agent --show-history "What sensors are on Chiller 6?"
    claude-agent --json "What is the current time?"
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import sys

_DEFAULT_MODEL = "claude-opus-4-6"
_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_LOG_DATE_FORMAT = "%H:%M:%S"
_HR = "─" * 60


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="claude-agent",
        description="Run a question through the Claude Agent SDK with AssetOpsBench MCP servers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
environment variables:
  LITELLM_API_KEY       LiteLLM / Anthropic API key (required)
  LITELLM_BASE_URL      LiteLLM proxy URL (required for litellm_proxy/* models)

examples:
  claude-agent "What assets are at site MAIN?"
  claude-agent --model-id claude-opus-4-6 --max-turns 20 "List sensors on Chiller 6"
  claude-agent --model-id litellm_proxy/aws/claude-opus-4-6 "What is the current time?"
  claude-agent --show-history "What sensors are on Chiller 6?"
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
        "--show-history",
        action="store_true",
        help="Print each turn's text, tool calls, and token usage.",
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


def _setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
    logging.root.handlers.clear()
    logging.root.addHandler(handler)
    logging.root.setLevel(level)


def _print_trace(trajectory) -> None:
    print(f"\n{_HR}")
    print("  Trace")
    print(_HR)
    for turn in trajectory.turns:
        print(f"\n  [Turn {turn.index}]  "
              f"in={turn.input_tokens} out={turn.output_tokens} tokens")
        if turn.text:
            snippet = turn.text[:200] + ("..." if len(turn.text) > 200 else "")
            print(f"    text: {snippet}")
        for tc in turn.tool_calls:
            print(f"    tool: {tc.name}  input: {tc.input}")
    print(f"\n  Total: {trajectory.total_input_tokens} input / "
          f"{trajectory.total_output_tokens} output tokens  "
          f"({len(trajectory.turns)} turns, "
          f"{len(trajectory.all_tool_calls)} tool calls)")


async def _run(args: argparse.Namespace) -> None:
    from agent.claude_agent.runner import ClaudeAgentRunner

    runner = ClaudeAgentRunner(model=args.model_id, max_turns=args.max_turns)
    result = await runner.run(args.question)

    if args.output_json:
        print(json.dumps(dataclasses.asdict(result.history), indent=2))
        return

    if args.show_history:
        _print_trace(result.history)

    print(f"\n{_HR}")
    print("  Answer")
    print(_HR)
    print(result.answer)
    print()


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    args = _build_parser().parse_args()
    _setup_logging(args.verbose)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
