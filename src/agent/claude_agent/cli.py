"""CLI entry point for the ClaudeAgentRunner.

Usage:
    claude-agent "What sensors are on Chiller 6?"
    claude-agent --model-id claude-opus-4-6 --max-turns 20 "List failure modes for pumps"
    claude-agent --show-trajectory "What sensors are on Chiller 6?"
    claude-agent --json "What is the current time?"
"""

from __future__ import annotations

import argparse

from .._cli_common import add_common_args, print_result, run_sdk_cli

_DEFAULT_MODEL = "litellm_proxy/aws/claude-opus-4-6"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="claude-agent",
        description="Run a question through the Claude Agent SDK with AssetOpsBench MCP servers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
environment variables:
  LITELLM_API_KEY       LiteLLM / Anthropic API key (required)
  LITELLM_BASE_URL      LiteLLM proxy URL (required for litellm_proxy/* models)

examples:
  claude-agent "What assets are at site MAIN?"
  claude-agent --model-id claude-opus-4-6 --max-turns 20 "List sensors on Chiller 6"
  claude-agent --model-id litellm_proxy/aws/claude-opus-4-6 "What is the current time?"
  claude-agent --show-trajectory "What sensors are on Chiller 6?"
  claude-agent --json "What is the current time?"
""",
    )
    add_common_args(parser, default_model=_DEFAULT_MODEL)
    parser.add_argument(
        "--max-turns",
        type=int,
        default=30,
        metavar="N",
        help="Maximum agentic loop turns (default: 30).",
    )
    return parser


async def _run(args: argparse.Namespace) -> None:
    from agent.claude_agent.runner import ClaudeAgentRunner

    runner = ClaudeAgentRunner(model=args.model_id, max_turns=args.max_turns)
    result = await runner.run(args.question)
    print_result(result, show_trajectory=args.show_trajectory, output_json=args.output_json)


def main() -> None:
    run_sdk_cli("claude-agent", _build_parser, _run)


if __name__ == "__main__":
    main()
