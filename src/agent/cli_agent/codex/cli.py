"""CLI entry point for the CodexCliRunner.

Usage:
    codex-agent --model-id tokenrouter/gpt-5 "What sensors are on Chiller 6?"
    codex-agent --model-id litellm_proxy/azure/gpt-5.4 --show-trajectory "List failure modes for pumps"
    codex-agent --model-id openrouter/openai/gpt-5.4 --json "What is the current time?"
"""

from __future__ import annotations

import argparse

from ..._cli_common import add_common_args, print_result, run_sdk_cli

_DEFAULT_MODEL = "litellm_proxy/azure/gpt-5.4"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codex-agent",
        description="Run a question through the Codex CLI with AssetOpsBench MCP servers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
model-id format (provider prefix selects routing):
  litellm_proxy/<model>          LiteLLM proxy (e.g. litellm_proxy/azure/gpt-5.4)
  openrouter/<vendor>/<model>    OpenRouter   (e.g. openrouter/openai/gpt-5.4)
  tokenrouter/<model>            TokenRouter  (e.g. tokenrouter/gpt-5)

environment variables (per provider):
  LITELLM_BASE_URL / LITELLM_API_KEY
  OPENROUTER_API_KEY            (base URL defaults to https://openrouter.ai/api/v1)
  TOKENROUTER_BASE_URL / TOKENROUTER_API_KEY

examples:
  codex-agent "What assets are at site MAIN?"
  codex-agent --model-id tokenrouter/gpt-5 --show-trajectory "List sensors on Chiller 6"
  codex-agent --json "What is the current time?"
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
    from agent.cli_agent.codex.runner import CodexCliRunner

    runner = CodexCliRunner(model=args.model_id, timeout_s=args.timeout)
    result = await runner.run(args.question)
    print_result(result, show_trajectory=args.show_trajectory, output_json=args.output_json)


def main() -> None:
    run_sdk_cli("codex-agent", _build_parser, _run)


if __name__ == "__main__":
    main()
