"""CLI entry point for the OpenAIAgentRunner.

Usage:
    openai-agent --model-id litellm_proxy/azure/gpt-5.4 "What sensors are on Chiller 6?"
    openai-agent --model-id litellm_proxy/azure/gpt-5.4 --max-turns 20 "List failure modes for pumps"
    openai-agent --model-id litellm_proxy/azure/gpt-5.4 --show-trajectory "What sensors are on Chiller 6?"
    openai-agent --model-id litellm_proxy/azure/gpt-5.4 --json "What is the current time?"
"""

from __future__ import annotations

import argparse

from .._cli_common import add_common_args, print_result, run_sdk_cli

_DEFAULT_MODEL = "litellm_proxy/azure/gpt-5.4"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openai-agent",
        description="Run a question through the OpenAI Agents SDK with AssetOpsBench MCP servers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
model-id format:
  litellm_proxy/<model>   LiteLLM proxy (e.g. litellm_proxy/azure/gpt-5.4)

environment variables:
  LITELLM_API_KEY       LiteLLM API key    (required)
  LITELLM_BASE_URL      LiteLLM base URL   (required)

examples:
  openai-agent "What assets are at site MAIN?"
  openai-agent --model-id litellm_proxy/azure/gpt-5.4 --max-turns 20 "List sensors on Chiller 6"
  openai-agent --show-trajectory "What are the failure modes for a chiller?"
  openai-agent --json "What is the current time?"
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
    from agent.openai_agent.runner import OpenAIAgentRunner

    runner = OpenAIAgentRunner(model=args.model_id, max_turns=args.max_turns)
    result = await runner.run(args.question)
    print_result(result, show_trajectory=args.show_trajectory, output_json=args.output_json)


def main() -> None:
    run_sdk_cli("openai-agent", _build_parser, _run)


if __name__ == "__main__":
    main()
