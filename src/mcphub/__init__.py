"""
mcphub — a ToolUniverse-style client over the AssetOpsBench MCP servers.

Same interface as ToolUniverse (load_tools -> run), implemented directly over
the six stdio FastMCP servers (iot, utilities, fmsr, wo, tsfm, vibration).
Depends only on the ``mcp`` client the project already requires — there is no
dependency on any external ``tooluniverse`` package.

    from mcphub import ToolUniverse

    tu = ToolUniverse()                                   # 1. init
    tu.load_tools()                                       # 2. load (connect + discover)
    tu.run({                                              # 3. run
        "name": "iot.sensors",
        "arguments": {"site_name": "MAIN", "asset_id": "Chiller 6"},
    })

Tools are namespaced ``<server>.<tool>``; a bare tool name also works when it is
unambiguous across loaded servers. Workflows are registered and callable through
the same ``run`` entrypoint (see ``mcphub.workflows``).
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import threading
from contextlib import AsyncExitStack
from typing import Any, Callable, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

__all__ = ["ToolUniverse", "ToolClient", "DEFAULT_SERVERS"]

_log = logging.getLogger("mcphub")

# How to launch each server as a stdio process. `uv run` resolves the project
# venv + .env; pass your own table if you run inside an already-active venv.
_SERVER_NAMES = ["iot", "utilities", "fmsr", "wo", "tsfm", "vibration"]
DEFAULT_SERVERS = {n: ["uv", "run", f"{n}-mcp-server"] for n in _SERVER_NAMES}


def _unwrap(result) -> Any:
    """MCP CallToolResult -> plain Python value."""
    structured = getattr(result, "structuredContent", None)
    if structured:
        return structured
    out = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text is None:
            out.append(block)
            continue
        try:
            out.append(json.loads(text))
        except (ValueError, TypeError):
            out.append(text)
    if not out:
        return None
    return out[0] if len(out) == 1 else out


class _Worker:
    """A single asyncio task (on a daemon thread) owns every MCP session.

    MCP's stdio sessions use anyio cancel scopes that must be entered and exited
    in the same task, so all session work is funnelled through one worker task;
    sync callers post commands over a queue and block on a thread-safe future.
    """

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        self._queue: asyncio.Queue = self._call(asyncio.Queue)
        asyncio.run_coroutine_threadsafe(self._run(), self._loop)

    def _call(self, fn):
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._loop.call_soon_threadsafe(
            lambda: fut.set_result(fn()) if not fut.done() else None
        )
        return fut.result()

    def _submit(self, action: str, payload):
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._loop.call_soon_threadsafe(
            self._queue.put_nowait, (action, payload, fut)
        )
        return fut.result()

    async def _run(self):
        stack = AsyncExitStack()
        sessions: Dict[str, ClientSession] = {}
        stop_fut = None
        try:
            while True:
                action, payload, fut = await self._queue.get()
                if action == "stop":
                    stop_fut = fut
                    break
                try:
                    if action == "open":
                        name, cmd, env = payload
                        params = StdioServerParameters(
                            command=cmd[0], args=cmd[1:], env=env
                        )
                        read, write = await stack.enter_async_context(
                            stdio_client(params)
                        )
                        session = await stack.enter_async_context(
                            ClientSession(read, write)
                        )
                        await session.initialize()
                        sessions[name] = session
                        _set(fut, None)
                    elif action == "list":
                        _set(fut, await sessions[payload].list_tools())
                    elif action == "call":
                        name, tool, args = payload
                        _set(fut, await sessions[name].call_tool(tool, args))
                except Exception as exc:
                    _set(fut, exc, error=True)
        finally:
            await stack.aclose()
            if stop_fut is not None:
                _set(stop_fut, None)

    def open(self, name, cmd, env):
        self._submit("open", (name, cmd, env))

    def list(self, name):
        return self._submit("list", name)

    def call(self, name, tool, args):
        return self._submit("call", (name, tool, args))

    def stop(self):
        try:
            self._submit("stop", None)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)


def _set(fut, value, error=False):
    if fut.done():
        return
    fut.set_exception(value) if error else fut.set_result(value)


class ToolUniverse:
    """ToolUniverse-style facade over the AssetOpsBench MCP servers."""

    def __init__(self, servers: Optional[Dict[str, List[str]]] = None,
                 env: Optional[dict] = None):
        self.servers = servers or DEFAULT_SERVERS
        self.env = {**os.environ, **(env or {})}
        self._worker = _Worker()
        self._connected: set = set()
        self.all_tools: Dict[str, dict] = {}       # qualified name -> spec
        self._alias: Dict[str, str] = {}            # bare name -> qualified
        self._workflows: Dict[str, Callable] = {}
        self._register_default_workflows()

    # -- context manager -- #
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._worker.stop()

    # -- step 2: load ------------------------------------------------------ #
    def load_tools(self, servers: Optional[List[str]] = None) -> int:
        """Connect to servers and discover their tools. Returns tool count."""
        names = list(servers) if servers else list(self.servers)
        for server in names:
            self._connect(server)
            for t in self._worker.list(server).tools:
                qualified = f"{server}.{t.name}"
                self.all_tools[qualified] = {
                    "name": qualified,
                    "server": server,
                    "tool": t.name,
                    "description": t.description or "",
                    "parameters": getattr(t, "inputSchema", {}) or {},
                }
                # Register a bare alias when it is unambiguous.
                if t.name not in self._alias and t.name not in self.all_tools:
                    self._alias[t.name] = qualified
                elif self._alias.get(t.name) not in (None, qualified):
                    _log.warning("Ambiguous tool name '%s'; use '%s'.",
                                 t.name, qualified)
                    self._alias.pop(t.name, None)
        return len(self.all_tools)

    # -- step 3: run ------------------------------------------------------- #
    def run(self, query, arguments: Optional[dict] = None) -> Any:
        """Execute a tool or workflow.

        ToolUniverse form:  run({"name": "iot.sensors", "arguments": {...}})
        Shorthand:          run("iot.sensors", {...})
        """
        if isinstance(query, dict):
            name = query["name"]
            args = query.get("arguments", {}) or {}
        else:
            name = query
            args = arguments or {}

        if name in self._workflows:
            return self._workflows[name](self, **args)

        server, tool = self._resolve(name)
        result = self._worker.call(server, tool, args)
        if getattr(result, "isError", False):
            raise RuntimeError(f"Tool '{name}' failed: {_unwrap(result)}")
        return _unwrap(result)

    # -- discovery --------------------------------------------------------- #
    def find_tools(self, description: str, limit: int = 5) -> List[dict]:
        """Keyword search over loaded tool names + descriptions."""
        terms = [w for w in description.lower().split() if w]

        def score(spec):
            hay = f"{spec['name']} {spec['description']}".lower()
            return sum(hay.count(t) for t in terms)

        ranked = sorted(self.all_tools.values(), key=score, reverse=True)
        return [s for s in ranked if score(s) > 0][:limit]

    def list_tools(self, server: Optional[str] = None) -> List[str]:
        """List loaded tool names, optionally for one server (lazy-connects)."""
        if server is not None:
            self._connect(server)
            return [t.name for t in self._worker.list(server).tools]
        return sorted(self.all_tools) + sorted(self._workflows)

    def tool_specification(self, name: str) -> dict:
        return self.all_tools[self._qualify(name)]

    # -- workflows --------------------------------------------------------- #
    def register_workflow(self, name: str, fn: Callable):
        """Register a workflow callable fn(tu, **arguments) under `name`."""
        self._workflows[name] = fn

    def _register_default_workflows(self):
        from . import workflows
        for wf_name in getattr(workflows, "REGISTERED", []):
            self._workflows[wf_name] = getattr(workflows, wf_name)

    # -- internals --------------------------------------------------------- #
    def _connect(self, server: str):
        if server not in self._connected:
            if server not in self.servers:
                raise KeyError(
                    f"Unknown server '{server}'. Known: {list(self.servers)}"
                )
            self._worker.open(server, self.servers[server], self.env)
            self._connected.add(server)

    def _qualify(self, name: str) -> str:
        if name in self.all_tools:
            return name
        if name in self._alias:
            return self._alias[name]
        if "." in name:
            return name
        raise KeyError(f"Tool '{name}' not loaded. Call load_tools() first.")

    def _resolve(self, name: str):
        qualified = self._qualify(name)
        server, _, tool = qualified.partition(".")
        if not tool:
            raise ValueError(f"name must be '<server>.<tool>', got '{name}'")
        self._connect(server)
        return server, tool


# Backwards-compatible alias.
ToolClient = ToolUniverse