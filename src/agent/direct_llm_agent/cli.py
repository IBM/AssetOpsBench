"""CLI entry point for the direct LLM baseline runner.

Usage:
    direct-llm-agent "Return only JSON: {\"test\": 1}"
    direct-llm-agent --model-id litellm_proxy/Azure/gpt-5-mini-2025-08-07 "..."
"""

from __future__ import annotations

import argparse
import sys

from agent._cli_common import add_common_args, print_result, run_sdk_cli

_DEFAULT_MODEL = "tokenrouter/MiniMax-M3"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="direct-llm-agent",
        description="Run a question through a direct model-only LLM baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
model-id format:
  tokenrouter/<model>      TokenRouter (OpenAI-compatible) model  [default]
                           e.g. tokenrouter/MiniMax-M3
  litellm_proxy/<model>    LiteLLM proxy model
                           e.g. litellm_proxy/Azure/gpt-5-mini-2025-08-07

environment variables:
  TOKENROUTER_API_KEY      TokenRouter API key  (for tokenrouter/* models)
  TOKENROUTER_BASE_URL     TokenRouter base URL (e.g. https://api.tokenrouter.com/v1)
  LITELLM_API_KEY          LiteLLM API key      (for litellm_proxy/* models)
  LITELLM_BASE_URL         LiteLLM base URL

examples:
  direct-llm-agent 'Return only JSON: {"test": 1}'
  direct-llm-agent --model-id tokenrouter/MiniMax-M3 'Return only one integer.'
""",
    )
    add_common_args(parser, default_model=_DEFAULT_MODEL)
    return parser


def _build_llm(model_id: str):
    try:
        from llm import make_backend
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        return make_backend(model_id)
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
