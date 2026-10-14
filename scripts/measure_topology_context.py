#!/usr/bin/env python
"""Measure the root agent's per-turn tool-surface cost under each topology.

This is the static half of the context-bloat measurement, and it costs nothing:
it connects to the MCP servers, reads the real JSON Schemas, and computes what
each topology pins into the root context on every turn. No model is called, so
the numbers are exact and reproducible rather than sampled from a sweep.

    uv run python scripts/measure_topology_context.py
    uv run python scripts/measure_topology_context.py --json topology_cost.json

Reported per configuration:

``tools``   entries in the root's tool list
``tokens``  tokens of name + description + JSON Schema, re-sent every turn

What it does NOT measure is the dynamic half: schemas the gateway discloses on
demand, tokens re-paid inside sub-agent delegations, and the discovery turns the
gateway spends before it can act. Those need a real sweep, and the topology that
looks cheapest here is not necessarily the one that finishes a scenario for
fewer total tokens. Report both halves together.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent.runner import DEFAULT_SERVER_PATHS  # noqa: E402
from agent.stirrup_agent.gateway import (  # noqa: E402
    MCPGatewayToolProvider,
    ToolIndex,
)
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _counter():
    """Return a token counter: tiktoken when available, else chars/4."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(enc.encode(text)), "tiktoken/cl100k_base"
    except Exception:  # noqa: BLE001 - the estimate is fine, just say which one
        return lambda text: max(1, len(text) // 4), "estimate (chars/4)"


def _tool_cost(tool, count) -> int:
    """Tokens for one tool as it appears on the wire."""
    try:
        schema = json.dumps(tool.parameters.model_json_schema(), ensure_ascii=False)
    except Exception:  # noqa: BLE001
        schema = "{}"
    return count(f"{tool.name}\n{tool.description or ''}\n{schema}")


class _Static:
    """Stand in for a connected provider when building gateway tools offline."""

    def __init__(self, tools):
        self._tools = tools

    async def __aenter__(self):
        return self._tools

    async def __aexit__(self, *exc):
        return None


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None, help="Write results here.")
    args = parser.parse_args()

    from stirrup.tools.mcp import MCPConfig, MCPToolProvider

    servers = {
        name: {
            "command": "uv",
            "args": ["run", "--directory", str(_REPO_ROOT), str(spec)],
            "cwd": str(_REPO_ROOT),
        }
        for name, spec in DEFAULT_SERVER_PATHS.items()
    }
    config = MCPConfig.model_validate({"mcpServers": servers})

    count, counter_name = _counter()
    provider = MCPToolProvider(config=config)
    tools = await provider.__aenter__()
    try:
        by_server: dict[str, list] = {}
        for tool in tools:
            by_server.setdefault(tool.name.split("__", 1)[0], []).append(tool)

        print(f"Connected: {len(tools)} tools across {len(by_server)} servers")
        print(f"Token counter: {counter_name}\n")

        print(f"{'server':<12}{'tools':>7}{'tokens':>10}")
        print("-" * 29)
        flat_total = 0
        for server in sorted(by_server):
            cost = sum(_tool_cost(t, count) for t in by_server[server])
            flat_total += cost
            print(f"{server:<12}{len(by_server[server]):>7}{cost:>10,}")
        print("-" * 29)
        print(f"{'flat':<12}{len(tools):>7}{flat_total:>10,}\n")

        results = {
            "counter": counter_name,
            "servers": {
                s: {"tools": len(v), "tokens": sum(_tool_cost(t, count) for t in v)}
                for s, v in sorted(by_server.items())
            },
            "topologies": {"flat": {"tools": len(tools), "tokens": flat_total}},
        }

        # Gateway: three routing tools, index mode additionally pinning the
        # one-line catalogue into describe_tools' description.
        for mode in ("index", "search"):
            gateway = MCPGatewayToolProvider(_Static(tools), mode=mode)
            gw_tools = await gateway.__aenter__()
            cost = sum(_tool_cost(t, count) for t in gw_tools)
            results["topologies"][f"gateway_{mode}"] = {
                "tools": len(gw_tools),
                "tokens": cost,
            }

        print(f"{'topology':<20}{'tools':>7}{'tokens':>10}{'vs flat':>10}")
        print("-" * 47)
        for name in ("flat", "gateway_index", "gateway_search"):
            entry = results["topologies"][name]
            share = entry["tokens"] / flat_total if flat_total else 0
            print(
                f"{name:<20}{entry['tools']:>7}{entry['tokens']:>10,}"
                f"{1 - share:>9.0%}"
            )
        print(
            "\nPer-turn static cost only. Deferred schemas, delegation overhead\n"
            "and discovery turns are dynamic and need a sweep to measure."
        )

        if args.json:
            args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"\nWrote {args.json}")
        return 0
    finally:
        await provider.__aexit__(None, None, None)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
