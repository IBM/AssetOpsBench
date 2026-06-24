"""CLI entry point for the OpenCodeAgentRunner.

Usage:
    opencode-agent "What sensors are on Chiller 6?"
    opencode-agent --model-id opencode/gpt-5.1-codex --max-steps 20 "List failure modes"
    opencode-agent --model-id tokenrouter/MiniMax-M3 "What is the current time?"
"""

from __future__ import annotations

import argparse

from .._cli_common import add_common_args, print_result, run_sdk_cli

_DEFAULT_MODEL = "opencode/gpt-5.1-codex"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opencode-agent",
        description="Run a question through OpenCode CLI with AssetOpsBench MCP servers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
model-id examples:
  opencode/gpt-5.1-codex       OpenCode Zen model
  openai/gpt-5.1               Direct OpenAI provider in OpenCode
  anthropic/claude-sonnet-4-5  Direct Anthropic provider in OpenCode
  litellm_proxy/<model>        Use LITELLM_BASE_URL / LITELLM_API_KEY
  tokenrouter/<model>          Use TOKENROUTER_BASE_URL / TOKENROUTER_API_KEY

environment variables:
  OPENAI_API_KEY          Direct OpenAI key, or set through `opencode /connect`
  ANTHROPIC_API_KEY       Direct Anthropic key, or set through `opencode /connect`
  LITELLM_API_KEY         LiteLLM router key for litellm_proxy/* models
  LITELLM_BASE_URL        LiteLLM OpenAI-compatible base URL
  TOKENROUTER_API_KEY     TokenRouter key for tokenrouter/* models
  TOKENROUTER_BASE_URL    TokenRouter OpenAI-compatible base URL

examples:
  opencode-agent "What assets are at site MAIN?"
  opencode-agent --model-id tokenrouter/MiniMax-M3 "List sensors on Chiller 6"
  opencode-agent --attach http://localhost:4096 "What is the current time?"
  opencode-agent --show-trajectory "What sensors are on Chiller 6?"
""",
    )
    add_common_args(parser, default_model=_DEFAULT_MODEL)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        metavar="N",
        help="Maximum OpenCode agentic iterations (default: 30).",
    )
    parser.add_argument(
        "--agent-name",
        default="assetops",
        help="OpenCode agent name to create/select from inline config.",
    )
    parser.add_argument(
        "--opencode-bin",
        default="opencode",
        help="OpenCode executable path (default: opencode).",
    )
    parser.add_argument(
        "--attach",
        default=None,
        metavar="URL",
        help="Attach to a running `opencode serve` instance, e.g. http://localhost:4096.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=900,
        help="Wall-clock timeout for `opencode run` in seconds (default: 900).",
    )
    parser.add_argument(
        "--allow-bash",
        action="store_true",
        help="Allow OpenCode's bash tool. Disabled by default for benchmark runs.",
    )
    parser.add_argument(
        "--allow-edit",
        action="store_true",
        help="Allow OpenCode file edits. Disabled by default for benchmark runs.",
    )
    parser.add_argument(
        "--allow-web",
        action="store_true",
        help="Allow OpenCode web fetch/search. Disabled by default for benchmark runs.",
    )
    parser.add_argument(
        "--ask-permissions",
        action="store_true",
        help="Do not auto-approve allowed permissions. Usually unsuitable for batch runs.",
    )
    return parser


async def _run(args: argparse.Namespace) -> None:
    from agent.opencode_agent.runner import OpenCodeAgentRunner

    runner = OpenCodeAgentRunner(
        model=args.model_id,
        max_steps=args.max_steps,
        agent_name=args.agent_name,
        opencode_bin=args.opencode_bin,
        attach=args.attach,
        timeout_s=args.timeout_s,
        allow_bash=args.allow_bash,
        allow_edit=args.allow_edit,
        allow_web=args.allow_web,
        dangerously_skip_permissions=not args.ask_permissions,
    )
    result = await runner.run(args.question)
    print_result(
        result, show_trajectory=args.show_trajectory, output_json=args.output_json
    )


def main() -> None:
    run_sdk_cli("opencode-agent", _build_parser, _run)


if __name__ == "__main__":
    main()
