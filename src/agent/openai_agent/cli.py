"""CLI entry point for the OpenAIAgentRunner.

Usage:
    openai-agent --model-id litellm_proxy/azure/gpt-5.4 "What sensors are on Chiller 6?"
    openai-agent --model-id litellm_proxy/azure/gpt-5.4 --max-turns 20 "List failure modes for pumps"
    openai-agent --model-id litellm_proxy/azure/gpt-5.4 --show-trajectory "What sensors are on Chiller 6?"
    openai-agent --model-id litellm_proxy/azure/gpt-5.4 --json "What is the current time?"
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
  tokenrouter/<model>     TokenRouter (e.g. tokenrouter/openai/gpt-5.6-sol)

API routing:
  Responses API:
    tokenrouter/openai/gpt-5.*
    tokenrouter/MiniMax-M3
    tokenrouter/google/gemini-3.6-flash
  Chat Completions API:
    all other model IDs

reasoning summaries:
  tokenrouter/openai/gpt-5.* models request safe reasoning summaries by
  default. Raw internal chain-of-thought is never exposed. Use
  --reasoning-summary none to disable.

permissions:
  All AssetOpsBench MCP tools are enabled. Local files, Bash, edits, and web
  access are denied unless their --allow-* flags are passed. Files, Bash, and
  edits require --workspace-dir.

environment variables:
  LITELLM_API_KEY       LiteLLM API key    (required)
  LITELLM_BASE_URL      LiteLLM base URL   (required)
  TOKENROUTER_API_KEY   TokenRouter API key  (for tokenrouter/* models)
  TOKENROUTER_BASE_URL  TokenRouter base URL (e.g. https://api.tokenrouter.com/v1)

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
    parser.add_argument(
        "--reasoning-summary",
        choices=("auto", "concise", "detailed", "none"),
        default="auto",
        help=(
            "Reasoning-summary detail for Responses models (default: auto). "
            "Ignored for Chat Completions; use none to disable."
        ),
    )
    parser.add_argument(
        "--allow-files",
        action="store_true",
        help=(
            "Allow workspace file listing, reading, and search. "
            "Requires --workspace-dir."
        ),
    )
    parser.add_argument(
        "--allow-bash",
        action="store_true",
        help=(
            "Allow Bash commands and workspace edits. Requires --workspace-dir; "
            "this is not an OS-level sandbox."
        ),
    )
    parser.add_argument(
        "--allow-edit",
        action="store_true",
        help=(
            "Allow workspace file writes, replacements, and deletes. "
            "Requires --workspace-dir."
        ),
    )
    parser.add_argument(
        "--allow-web",
        action="store_true",
        help="Allow public web search and fetch tools.",
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Dedicated workspace required by --allow-files/--allow-bash/--allow-edit.",
    )
    return parser


async def _run(args: argparse.Namespace) -> None:
    from agent.openai_agent.runner import OpenAIAgentRunner

    async with OpenAIAgentRunner(
        model=args.model_id,
        max_turns=args.max_turns,
        allow_files=args.allow_files,
        allow_bash=args.allow_bash,
        allow_edit=args.allow_edit,
        allow_web=args.allow_web,
        workspace_dir=args.workspace_dir,
        reasoning_summary=(
            None if args.reasoning_summary == "none" else args.reasoning_summary
        ),
    ) as runner:
        result = await runner.run(args.question)
    print_result(
        result, show_trajectory=args.show_trajectory, output_json=args.output_json
    )


def main() -> None:
    run_sdk_cli("openai-agent", _build_parser, _run)


if __name__ == "__main__":
    main()
