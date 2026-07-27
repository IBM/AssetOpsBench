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


def _parse_mcp_tool_permission(value: str) -> tuple[str, str]:
    """Parse ``SERVER/TOOL`` for the repeatable MCP allowlist flag."""
    server_name, separator, tool_name = value.partition("/")
    if not separator or not server_name or not tool_name:
        raise argparse.ArgumentTypeError("expected SERVER/TOOL, for example iot/sites")
    return server_name, tool_name


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
  tokenrouter/openai/gpt-5.*   Responses API
  all other model IDs          Chat Completions API

permissions:
  Only configured AssetOpsBench MCP tools are exposed. Shell, file, edit, web,
  and other hosted tools are not registered. Repeat --allow-mcp-tool SERVER/TOOL
  to expose only selected MCP tools; once used, unlisted servers expose no tools.

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
        "--allow-mcp-tool",
        action="append",
        default=None,
        type=_parse_mcp_tool_permission,
        metavar="SERVER/TOOL",
        help=(
            "Restrict MCP access to this server/tool pair. Repeat as needed; "
            "using the flag enables a fail-closed allowlist."
        ),
    )
    return parser


async def _run(args: argparse.Namespace) -> None:
    from agent.openai_agent.runner import OpenAIAgentRunner

    mcp_tool_allowlist: dict[str, set[str]] | None = None
    if args.allow_mcp_tool:
        mcp_tool_allowlist = {}
        for server_name, tool_name in args.allow_mcp_tool:
            mcp_tool_allowlist.setdefault(server_name, set()).add(tool_name)

    async with OpenAIAgentRunner(
        model=args.model_id,
        max_turns=args.max_turns,
        mcp_tool_allowlist=mcp_tool_allowlist,
    ) as runner:
        result = await runner.run(args.question)
    print_result(
        result, show_trajectory=args.show_trajectory, output_json=args.output_json
    )


def main() -> None:
    run_sdk_cli("openai-agent", _build_parser, _run)


if __name__ == "__main__":
    main()
