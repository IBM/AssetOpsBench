"""CLI entry point for the GeminiCliRunner.

Usage:
    gemini-agent --model-id gemini-2.5-pro "What sensors are on Chiller 6?"
    gemini-agent --model-id litellm_proxy/gemini-2.5-pro --show-trajectory "List failure modes for pumps"
    gemini-agent --json "What is the current time?"
"""

from __future__ import annotations

import argparse

from ..._cli_common import add_common_args, print_result, run_sdk_cli

_DEFAULT_MODEL = "gemini-2.5-pro"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gemini-agent",
        description="Run a question through the Gemini CLI with AssetOpsBench MCP servers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
model-id format (provider prefix selects routing):
  <model>                  direct Google auth (e.g. gemini-2.5-pro, via GEMINI_API_KEY)
  litellm_proxy/<model>    LiteLLM proxy with the /gemini passthrough
  tokenrouter/<model>      TokenRouter configured for the Gemini API

  (Gemini CLI speaks the Gemini API, so OpenAI-only providers such as
   OpenRouter are not supported.)

environment variables:
  GEMINI_API_KEY                          (direct)
  LITELLM_BASE_URL / LITELLM_API_KEY      (proxy)
  TOKENROUTER_BASE_URL / TOKENROUTER_API_KEY

examples:
  gemini-agent "What assets are at site MAIN?"
  gemini-agent --model-id gemini-2.5-pro --show-trajectory "List sensors on Chiller 6"
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
    from agent.cli_agent.gemini.runner import GeminiCliRunner

    runner = GeminiCliRunner(model=args.model_id, timeout_s=args.timeout)
    result = await runner.run(args.question)
    print_result(result, show_trajectory=args.show_trajectory, output_json=args.output_json)


def main() -> None:
    run_sdk_cli("gemini-agent", _build_parser, _run)


if __name__ == "__main__":
    main()
