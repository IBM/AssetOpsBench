"""CLI entry point for the ClaudeCodeRunner.

Usage:
    claude-code-agent --model-id litellm_proxy/aws/claude-opus-4-6 "What sensors are on Chiller 6?"
    claude-code-agent --show-trajectory "List failure modes for pumps"
    claude-code-agent --json "What is the current time?"
"""

from __future__ import annotations

import argparse

from ..._cli_common import add_common_args, print_result, run_sdk_cli

_DEFAULT_MODEL = "litellm_proxy/aws/claude-opus-4-6"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="claude-code-agent",
        description="Run a question through the Claude Code CLI with AssetOpsBench MCP servers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
model-id format (provider prefix selects routing):
  litellm_proxy/<model>    LiteLLM proxy with an Anthropic-compatible route
  tokenrouter/<model>      TokenRouter configured for the Anthropic API

  (Claude Code speaks the Anthropic Messages API, so OpenAI-only providers
   such as OpenRouter are not supported.)

environment variables:
  LITELLM_BASE_URL / LITELLM_API_KEY      (point base URL at the /anthropic route)
  TOKENROUTER_BASE_URL / TOKENROUTER_API_KEY

examples:
  claude-code-agent "What assets are at site MAIN?"
  claude-code-agent --show-trajectory "What are the failure modes for a chiller?"
""",
    )
    add_common_args(parser, default_model=_DEFAULT_MODEL)
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        metavar="S",
        help="Hard wall-clock cap in seconds (default: 900).",
    )
    return parser


async def _run(args: argparse.Namespace) -> None:
    from agent.cli_agent.claude_code.runner import ClaudeCodeRunner

    runner = ClaudeCodeRunner(model=args.model_id, timeout_s=args.timeout)
    result = await runner.run(args.question)
    print_result(result, show_trajectory=args.show_trajectory, output_json=args.output_json)


def main() -> None:
    run_sdk_cli("claude-code-agent", _build_parser, _run)


if __name__ == "__main__":
    main()
