"""CLI entry point for the DeepAgentRunner.

Usage:
    deep-agent "What sensors are on Chiller 6?"
    deep-agent --model-id litellm_proxy/aws/claude-opus-4-6 "List failure modes for pumps"
    deep-agent --show-trajectory "What sensors are on Chiller 6?"
    deep-agent --json "What is the current time?"
"""

from __future__ import annotations

import argparse

from .._cli_common import add_common_args, print_result, run_sdk_cli

_DEFAULT_MODEL = "litellm_proxy/aws/claude-opus-4-6"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deep-agent",
        description="Run a question through LangChain deep-agents with AssetOpsBench MCP servers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
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
    add_common_args(parser, default_model=_DEFAULT_MODEL)
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=100,
        metavar="N",
        help="Maximum graph recursion steps (default: 100).",
    )
    return parser


async def _run(args: argparse.Namespace) -> None:
    from agent.deep_agent.runner import DeepAgentRunner

    runner = DeepAgentRunner(
        model=args.model_id,
        recursion_limit=args.recursion_limit,
    )
    result = await runner.run(args.question)
    print_result(result, show_trajectory=args.show_trajectory, output_json=args.output_json)


def main() -> None:
    run_sdk_cli("deep-agent", _build_parser, _run)


if __name__ == "__main__":
    main()
