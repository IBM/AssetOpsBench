"""CLI entry point for the DeepAgentRunner.

Usage:
    deep-agent "What sensors are on Chiller 6?"
    deep-agent --model-id litellm_proxy/aws/claude-opus-4-6 "List failure modes for pumps"
    deep-agent --show-trajectory "What sensors are on Chiller 6?"
    deep-agent --json "What is the current time?"
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import sys

_DEFAULT_MODEL = "litellm_proxy/aws/claude-opus-4-6"
_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_LOG_DATE_FORMAT = "%H:%M:%S"
_HR = "─" * 60


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deep-agent",
        description="Run a question through LangChain deep-agents with AssetOpsBench MCP servers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
model-id format:
  litellm_proxy/<model>   LiteLLM proxy (e.g. litellm_proxy/aws/claude-opus-4-6)
  <provider>:<model>      Native provider (e.g. anthropic:claude-sonnet-4-6)

environment variables:
  LITELLM_API_KEY       LiteLLM API key    (required for litellm_proxy/* models)
  LITELLM_BASE_URL      LiteLLM base URL   (required for litellm_proxy/* models)

examples:
  deep-agent "What assets are at site MAIN?"
  deep-agent --model-id litellm_proxy/aws/claude-opus-4-6 "List sensors on Chiller 6"
  deep-agent --show-trajectory "What are the failure modes for a chiller?"
  deep-agent --json "What is the current time?"
""",
    )
    parser.add_argument("question", help="The question to answer.")
    parser.add_argument(
        "--model-id",
        default=_DEFAULT_MODEL,
        metavar="MODEL_ID",
        help=f"Model string; LiteLLM proxy or native provider (default: {_DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=100,
        metavar="N",
        help="Maximum graph recursion steps (default: 100).",
    )
    parser.add_argument(
        "--show-trajectory",
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
            if tc.output is not None:
                out_str = str(tc.output)
                snippet = out_str[:200] + ("..." if len(out_str) > 200 else "")
                print(f"    output: {snippet}")
    print(f"\n  Total: {trajectory.total_input_tokens} input / "
          f"{trajectory.total_output_tokens} output tokens  "
          f"({len(trajectory.turns)} turns, "
          f"{len(trajectory.all_tool_calls)} tool calls)")


async def _run(args: argparse.Namespace) -> None:
    from agent.deep_agent.runner import DeepAgentRunner

    runner = DeepAgentRunner(
        model=args.model_id,
        recursion_limit=args.recursion_limit,
    )
    result = await runner.run(args.question)

    if args.output_json:
        print(json.dumps(dataclasses.asdict(result.trajectory), indent=2, default=str))
        return

    if args.show_trajectory:
        _print_trace(result.trajectory)

    print(f"\n{_HR}")
    print("  Answer")
    print(_HR)
    print(result.answer)
    print()


def main() -> None:
    from dotenv import load_dotenv

    from observability import init_tracing

    load_dotenv()
    args = _build_parser().parse_args()
    _setup_logging(args.verbose)
    init_tracing("deep-agent")
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
