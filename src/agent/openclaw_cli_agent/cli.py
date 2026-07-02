"""CLI entry point for the OpenClawCliAgentRunner.

Usage:
    openclaw-cli-agent "What sensors are on Chiller 6?"
    openclaw-cli-agent --model-id tokenrouter/MiniMax-M3 "What is the current time?"
    openclaw-cli-agent --workspace-dir /tmp/openclaw-run --allow-files --allow-bash "Analyze work orders"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .._cli_common import add_common_args, print_result, run_sdk_cli

_DEFAULT_MODEL = "tokenrouter/MiniMax-M3"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openclaw-cli-agent",
        description="Run a question through OpenClaw CLI with AssetOpsBench MCP servers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
model-id examples:
  tokenrouter/MiniMax-M3       TokenRouter OpenAI-compatible endpoint
  litellm_proxy/<model>        Use LITELLM_BASE_URL / LITELLM_API_KEY
  openai/gpt-5.1              Direct OpenAI provider in OpenClaw
  anthropic/claude-sonnet-4-5 Direct Anthropic provider in OpenClaw

environment variables:
  TOKENROUTER_API_KEY     TokenRouter key for tokenrouter/* models
  TOKENROUTER_BASE_URL    TokenRouter OpenAI-compatible base URL
  LITELLM_API_KEY         LiteLLM router key for litellm_proxy/* models
  LITELLM_BASE_URL        LiteLLM OpenAI-compatible base URL
  OPENAI_API_KEY          Direct OpenAI key for openai/* models
  ANTHROPIC_API_KEY       Direct Anthropic key for anthropic/* models

examples:
  openclaw-cli-agent "What assets are at site MAIN?"
  openclaw-cli-agent --model-id tokenrouter/MiniMax-M3 "What is the current time?"
  openclaw-cli-agent --workspace-dir /tmp/openclaw-run --allow-files --allow-bash "Analyze work orders"
  openclaw-cli-agent --show-trajectory "What sensors are on Chiller 6?"
""",
    )
    add_common_args(parser, default_model=_DEFAULT_MODEL)
    parser.add_argument(
        "--agent-name",
        default="main",
        help="OpenClaw agent id to target from generated config (default: main).",
    )
    parser.add_argument(
        "--openclaw-bin",
        default="openclaw",
        help="OpenClaw executable path (default: openclaw).",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=900,
        help="Wall-clock timeout for `openclaw agent` in seconds (default: 900).",
    )
    parser.add_argument(
        "--thinking",
        default="off",
        help="OpenClaw thinking level, e.g. off, low, medium, high, xhigh.",
    )
    parser.add_argument(
        "--allow-files",
        action="store_true",
        help="Allow OpenClaw file inspection tools inside --workspace-dir.",
    )
    parser.add_argument(
        "--allow-bash",
        action="store_true",
        help="Allow OpenClaw shell/exec tools. Disabled by default for benchmark runs.",
    )
    parser.add_argument(
        "--allow-edit",
        action="store_true",
        help="Allow OpenClaw file edits inside --workspace-dir.",
    )
    parser.add_argument(
        "--allow-web",
        action="store_true",
        help="Allow OpenClaw web search/fetch. Disabled by default.",
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Dedicated OpenClaw run workspace. Required when enabling files, edits, or bash.",
    )
    parser.add_argument(
        "--home-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Dedicated OpenClaw HOME root. Defaults to an isolated temp dir or workspace child.",
    )
    return parser


async def _run(args: argparse.Namespace) -> None:
    from agent.openclaw_cli_agent.runner import OpenClawCliAgentRunner

    runner = OpenClawCliAgentRunner(
        model=args.model_id,
        agent_name=args.agent_name,
        openclaw_bin=args.openclaw_bin,
        timeout_s=args.timeout_s,
        thinking=args.thinking,
        allow_files=args.allow_files,
        allow_bash=args.allow_bash,
        allow_edit=args.allow_edit,
        allow_web=args.allow_web,
        workspace_dir=args.workspace_dir,
        home_dir=args.home_dir,
    )
    result = await runner.run(args.question)
    print_result(
        result, show_trajectory=args.show_trajectory, output_json=args.output_json
    )


def main() -> None:
    run_sdk_cli("openclaw-cli-agent", _build_parser, _run)


if __name__ == "__main__":
    main()
