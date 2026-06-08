"""CLI entry point for the StirrupAgentRunner.

Usage:
    stirrup-agent "What sensors are on Chiller 6?"
    stirrup-agent --model-id litellm_proxy/aws/claude-opus-4-6 "Summarise WO backlog"
    stirrup-agent --no-code "List failure modes for pumps"          # tools-only track
    stirrup-agent --code-backend local --show-trajectory "Generate a work order"
    stirrup-agent --json "What is the current time?"
"""

from __future__ import annotations

import argparse

from .._cli_common import add_common_args, print_result, run_sdk_cli

_DEFAULT_MODEL = "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stirrup-agent",
        description="Run a question through a Stirrup agent with AssetOpsBench MCP servers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
model-id format:
  litellm_proxy/<model>   LiteLLM proxy (e.g. litellm_proxy/aws/claude-opus-4-6)
  <provider>/<model>      Native via Stirrup's LiteLLMClient. watsonx works
                          directly here, e.g.
                          watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8

tracks:
  --code-enabled (default)  Add a sandboxed code-execution tool (code track).
  --no-code                 Tools-only: directly comparable to the other runners.

environment variables:
  LITELLM_API_KEY       LiteLLM API key    (required for litellm_proxy/* models)
  LITELLM_BASE_URL      LiteLLM base URL   (required for litellm_proxy/* models)
  WATSONX_APIKEY/...    Standard LiteLLM watsonx vars (for watsonx/* models)
  STIRRUP_CODE_IMAGE    Docker image for the code track (default python:3.12-slim)

examples:
  stirrup-agent "What assets are at site MAIN?"
  stirrup-agent --no-code --show-trajectory "List sensors on Chiller 6"
  stirrup-agent --json "Cross-reference open work orders against forecasted anomalies"
""",
    )
    add_common_args(parser, default_model=_DEFAULT_MODEL)
    code_group = parser.add_mutually_exclusive_group()
    code_group.add_argument(
        "--code-enabled",
        dest="code_enabled",
        action="store_true",
        default=True,
        help="Enable a sandboxed code-execution tool so the agent can run code (default).",
    )
    code_group.add_argument(
        "--no-code",
        dest="code_enabled",
        action="store_false",
        help="Tools-only run; disable code execution for 1:1 comparison.",
    )
    parser.add_argument(
        "--code-backend",
        choices=["docker", "local", "e2b"],
        default="docker",
        help="Code-execution sandbox backend (default: docker).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=30,
        metavar="N",
        help="Maximum agent turns (default: 30).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16_384,
        metavar="N",
        help="Max output tokens per model call; must stay under the provider "
             "limit (watsonx caps new tokens at 100k). Default: 16384.",
    )
    return parser


async def _run(args: argparse.Namespace) -> None:
    from agent.stirrup_agent.runner import StirrupAgentRunner

    runner = StirrupAgentRunner(
        model=args.model_id,
        code_enabled=args.code_enabled,
        code_backend=args.code_backend,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
    )
    result = await runner.run(args.question)
    print_result(result, show_trajectory=args.show_trajectory, output_json=args.output_json)


def main() -> None:
    run_sdk_cli("stirrup-agent", _build_parser, _run)


if __name__ == "__main__":
    main()