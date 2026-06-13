"""CLI entry point for the direct LLM baseline runner.

Usage:
    direct-llm-agent "Return only JSON: {\"test\": 1}"
    direct-llm-agent --model-id litellm_proxy/Azure/gpt-5-mini-2025-08-07 "..."
"""

from __future__ import annotations

import argparse
import sys

from agent._cli_common import add_common_args, print_result, run_sdk_cli

_DEFAULT_MODEL = "litellm_proxy/Azure/gpt-5-mini-2025-08-07"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="direct-llm-agent",
        description="Run a question through a direct model-only LLM baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
model-id format:
  litellm_proxy/<model>    LiteLLM proxy model
                           e.g. litellm_proxy/Azure/gpt-5-mini-2025-08-07

environment variables:
  LITELLM_API_KEY          LiteLLM API key
  LITELLM_BASE_URL         LiteLLM base URL

examples:
  direct-llm-agent 'Return only JSON: {"test": 1}'
  direct-llm-agent --model-id litellm_proxy/aws/claude-opus-4-8 'Return only one integer.'
""",
    )
    add_common_args(parser, default_model=_DEFAULT_MODEL)
    return parser


def _build_llm(model_id: str):
    try:
        from llm.litellm import LiteLLMBackend
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        return LiteLLMBackend(model_id=model_id)
    except KeyError as exc:
        print(f"error: missing environment variable {exc}", file=sys.stderr)
        sys.exit(1)


async def _run(args: argparse.Namespace) -> None:
    from agent.direct_llm_agent.runner import DirectLLMAgentRunner

    llm = _build_llm(args.model_id)
    runner = DirectLLMAgentRunner(llm=llm)
    result = await runner.run(args.question)

    print_result(
        result,
        show_trajectory=args.show_trajectory,
        output_json=args.output_json,
    )


def main() -> None:
    run_sdk_cli("direct-llm-agent", _build_parser, _run)


if __name__ == "__main__":
    main()