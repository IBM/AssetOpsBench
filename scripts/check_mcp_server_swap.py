#!/usr/bin/env python
"""Compare the tool surface of a swapped MCP server against the local one.

Swapping a domain server for someone else's hosted service is not a drop-in:
tool names, schemas and coverage all differ, and a scenario calling a tool the
replacement does not expose fails at run time rather than at startup. This
script connects to both and prints the difference, so a sweep starts with the
gap known instead of discovering it scenario by scenario.

    ASSETOPS_MCP_URL_TSFM=https://api.tsfm.ai/mcp \\
    ASSETOPS_MCP_TOKEN_TSFM=$TSFM_API_KEY \\
        uv run python scripts/check_mcp_server_swap.py --server tsfm

It calls no model and runs no scenario. It connects, lists tools, and exits
non-zero when the replacement is unreachable, so it is safe as a preflight gate
in front of a long run.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agent.mcp_servers import (  # noqa: E402
    DEFAULT_SERVER_PATHS,
    apply_env_overrides,
    is_remote,
)


async def _local_tools(entry_point: str) -> list[str]:
    """Tool names exposed by a local stdio server."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command="uv",
        args=["run", "--directory", str(REPO_ROOT), entry_point],
        cwd=str(REPO_ROOT),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return sorted(t.name for t in (await session.list_tools()).tools)


async def _remote_tools(spec) -> list[str]:
    """Tool names exposed by a Streamable HTTP server."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(
        url=spec.url,
        headers=spec.headers(),
        timeout=spec.timeout_s,
        sse_read_timeout=spec.sse_read_timeout_s,
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return sorted(t.name for t in (await session.list_tools()).tools)


def _root_causes(exc: BaseException) -> list[str]:
    """Flatten an exception, following ExceptionGroup and __cause__ chains.

    anyio wraps connection failures in a TaskGroup ExceptionGroup, whose repr
    says only "1 sub-exception". A preflight gate has to name the actual cause,
    because a bad token and an unreachable host need different fixes.
    """
    out: list[str] = []
    seen: set[int] = set()

    def walk(e: BaseException | None) -> None:
        if e is None or id(e) in seen:
            return
        seen.add(id(e))
        subs = getattr(e, "exceptions", None)
        if subs:
            for sub in subs:
                walk(sub)
            return
        out.append(f"{type(e).__name__}: {e}")
        walk(e.__cause__)
        walk(e.__context__)

    walk(exc)
    return out or [f"{type(exc).__name__}: {exc}"]


def _print_block(title: str, names: list[str]) -> None:
    print(f"\n{title} ({len(names)})")
    for name in names:
        print(f"  {name}")


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare a swapped MCP server's tools against the local one."
    )
    parser.add_argument(
        "--server",
        default="tsfm",
        help="Which server slot to inspect (default: tsfm).",
    )
    parser.add_argument(
        "--show-tools",
        action="store_true",
        help="Print every tool name, not just the counts and the gap.",
    )
    args = parser.parse_args()

    name = args.server
    if name not in DEFAULT_SERVER_PATHS:
        print(f"error: unknown server {name!r}", file=sys.stderr)
        return 2

    resolved = apply_env_overrides(dict(DEFAULT_SERVER_PATHS))
    spec = resolved[name]

    if not is_remote(spec):
        print(
            f"{name}: no override set. Export "
            f"ASSETOPS_MCP_URL_{name.upper()} to point this slot at a remote "
            "MCP service, and ASSETOPS_MCP_TOKEN_"
            f"{name.upper()} if it needs a bearer token."
        )
        return 1

    print(f"server slot : {name}")
    print(f"replacement : {spec.redacted()}")

    try:
        remote = await _remote_tools(spec)
    except Exception as exc:  # noqa: BLE001 - any failure is a failed preflight
        print("\nFAIL  could not reach the replacement:")
        for cause in _root_causes(exc):
            print(f"  {cause}")
        print(
            "\nA 401 or 403 means the token is wrong or missing; a DNS or "
            "connection error means the URL or your network is. Do not start a "
            "sweep until this connects."
        )
        return 1

    local_entry = DEFAULT_SERVER_PATHS[name]
    try:
        local = await _local_tools(str(local_entry))
    except Exception as exc:  # noqa: BLE001
        print(f"\nwarning: local {name} server did not start ({exc}).")
        print("Reporting the replacement's surface only.")
        local = []

    print(f"\nlocal tools       : {len(local)}")
    print(f"replacement tools : {len(remote)}")

    if local:
        missing = [t for t in local if t not in remote]
        extra = [t for t in remote if t not in local]
        shared = [t for t in local if t in remote]
        print(f"same name in both : {len(shared)}")
        print(f"only local        : {len(missing)}")
        print(f"only replacement  : {len(extra)}")
        if missing:
            print(
                f"\nScenarios calling any of these {len(missing)} tools will fail "
                "under the swap. That is the measurement; know the list first."
            )
        if args.show_tools:
            _print_block("shared", shared)
            _print_block("only local (will fail under the swap)", missing)
            _print_block("only replacement (unreachable by our scenarios)", extra)
    elif args.show_tools:
        _print_block("replacement tools", remote)

    print("\nPreflight OK: the replacement is reachable and lists its tools.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
