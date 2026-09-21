#!/usr/bin/env python
"""Assert the Stirrup APIs the sub-agent topology depends on.

The topology was implemented against Stirrup main at 247f24d5, which the
changelog described as "0.2.0 (unreleased)". The released 0.2.0 may not be that
commit exactly, so this checks the handful of behaviours the design actually
rests on, rather than trusting the version string.

Run it once after installing or upgrading Stirrup:

    uv run python scripts/check_stirrup_subagent_contract.py

It touches no network, no model, no MCP server, and no Docker: it reads the
installed package's own source and signatures.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import io
import json
import sys


from importlib.metadata import PackageNotFoundError, version as pkg_version


def _fail(message: str) -> None:
    print(f"FAIL  {message}")


def _ok(message: str) -> None:
    print(f"ok    {message}")


_SENTINEL = "STIRRUP_CONTRACT_PROBE_EARLY_TURN"


class _ProbeResult:
    """What one scripted sub-agent run revealed about the result shape."""

    def __init__(self) -> None:
        self.history_in_metadata = False
        self.history_in_content = False
        self.finish_params_in_content = False
        self.error: str | None = None


def _probe_sub_agent_result() -> "_ProbeResult":
    """Run a scripted sub-agent and inspect where its history ended up.

    Touches no network, no model, no MCP server and no Docker: the client is a
    stub that emits two canned turns. The first carries a sentinel that must
    never reach the parent's context; the second finishes.
    """
    out = _ProbeResult()
    try:
        from stirrup import Agent
        from stirrup.core.agent import SubAgentParams
        from stirrup.core.models import AssistantMessage, TextBlock, ToolCall
        from stirrup.tools.finish import DEFAULT_FINISH_TOOL_NAME, SIMPLE_FINISH_TOOL
        from stirrup.utils.logging import AgentLogger

        class _ScriptedClient:
            max_tokens = 4096
            context_window_tokens = 40_000
            model_slug = "fake/contract-probe"

            def __init__(self) -> None:
                self.calls = 0

            async def generate(self, messages, tools):
                self.calls += 1
                if self.calls == 1:
                    # A turn the parent must never see verbatim.
                    return AssistantMessage(blocks=[TextBlock(text=_SENTINEL)])
                return AssistantMessage(
                    blocks=[
                        TextBlock(text="final visible answer"),
                        ToolCall(
                            name=DEFAULT_FINISH_TOOL_NAME,
                            arguments=json.dumps(
                                {"reason": "probe complete", "paths": []}
                            ),
                            tool_call_id="finish-probe",
                        ),
                    ],
                )

        async def _run():
            agent = Agent(
                client=_ScriptedClient(),
                name="contract-probe",
                tools=[],
                finish_tool=SIMPLE_FINISH_TOOL,
                max_turns=4,
                logger=AgentLogger(show_spinner=False),
            )
            tool = agent.to_tool()
            return await tool.executor(SubAgentParams(task="probe the result shape"))

        # The agent logger renders panels to the console; this script's output
        # is the signal, so swallow them.
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            result = asyncio.run(_run())

        history = getattr(result.metadata, "message_history", None)
        content = str(result.content)
        out.history_in_metadata = bool(history) and _SENTINEL in repr(history)
        out.history_in_content = _SENTINEL in content
        out.finish_params_in_content = "probe complete" in content
    except Exception as exc:  # noqa: BLE001 - any failure means "unverified"
        out.error = f"{type(exc).__name__}: {exc}"
    return out


def main() -> int:
    failures = 0

    import stirrup
    from stirrup import Agent
    from stirrup.core.agent import SubAgentParams
    from stirrup.core.models import SubAgentMetadata
    from stirrup.tools.mcp import MCPToolProvider

    # `stirrup` ships no __version__ dunder, so read the installed
    # distribution's metadata. A sweep log that cannot name the framework
    # version it ran against is not reproducible.
    try:
        version = pkg_version("stirrup")
    except PackageNotFoundError:  # pragma: no cover - source checkout
        version = getattr(stirrup, "__version__", "unknown")
    print(f"stirrup {version}\n")

    sig = inspect.signature(Agent.__init__)

    # 1. The client carries the working-context budget explicitly. The runner
    #    dropped its `max_tokens` adapter because of this.
    from stirrup.clients.litellm_client import LiteLLMClient

    if "context_window_tokens" in inspect.signature(LiteLLMClient.__init__).parameters:
        _ok("LLM clients take context_window_tokens")
    else:
        _fail(
            "LiteLLMClient has no context_window_tokens parameter; runner.py's "
            "_build_client will not construct. Reinstate a client adapter that "
            "reports the working-context budget."
        )
        failures += 1

    # 2. share_parent_exec_env exists and is NOT what we want. subagents.py sets
    #    it False deliberately; if the parameter vanished, the comment explaining
    #    why is stale and someone will re-derive the trap.
    if "share_parent_exec_env" in sig.parameters:
        default = sig.parameters["share_parent_exec_env"].default
        if default is False:
            _ok("Agent(share_parent_exec_env=...) present, defaults False (we also pass it explicitly)")
        else:
            _ok(
                "Agent(share_parent_exec_env=...) present (we pass False on purpose; "
                f"upstream default is now {default!r})"
            )
    else:
        _fail(
            "Agent no longer takes share_parent_exec_env; re-read how a sub-agent "
            "acquires an exec env before trusting subagents.py's invariant 1."
        )
        failures += 1

    # 3+4. The whole context saving rests on this: a delegated run must hand
    #      its message history back as tool *metadata* while the tool *content*
    #      carries only the last assistant text plus a model_dump() of the
    #      finish params. Probing this behaviourally rather than by grepping
    #      to_tool's source means an upstream rename cannot raise a false
    #      alarm, and moving the logic into a helper cannot raise a false pass.
    probe = _probe_sub_agent_result()
    if probe.error is not None:
        _fail(
            f"Could not execute a sub-agent probe: {probe.error}. The contract "
            "below is unverified; do not trust a sweep until this runs."
        )
        failures += 1
    else:
        if probe.history_in_metadata and not probe.history_in_content:
            _ok("to_tool returns message_history via metadata, not in tool content")
        else:
            _fail(
                "to_tool no longer separates history from tool content "
                f"(in metadata={probe.history_in_metadata}, "
                f"leaked into content={probe.history_in_content}). If the full "
                "history now reaches the parent's context, --topology subagent "
                "saves nothing and the design needs revisiting."
            )
            failures += 1

        if probe.finish_params_in_content:
            _ok("finish params reach the parent via model_dump()")
        else:
            _fail(
                "A finished sub-agent's result no longer carries its finish "
                "params. Domain sub-agents would lose their artifact handles "
                "and entity ids."
            )
            failures += 1

    # 5. Per-server scoping is how one sub-agent gets exactly one MCP server.
    if "server_names" in inspect.signature(MCPToolProvider.__init__).parameters:
        _ok("MCPToolProvider takes server_names")
    else:
        _fail(
            "MCPToolProvider has no server_names parameter; workspace_bridge.py "
            "cannot scope a provider to one server."
        )
        failures += 1

    # 6. Field names the recorder and trajectory splicing read.
    for field in ("message_history", "run_metadata"):
        if field in SubAgentMetadata.model_fields:
            _ok(f"SubAgentMetadata.{field}")
        else:
            _fail(f"SubAgentMetadata has no {field}; SubAgentHistoryRecorder breaks.")
            failures += 1

    if "task" in SubAgentParams.model_fields:
        _ok("SubAgentParams.task")
    else:
        _fail("SubAgentParams has no task field.")
        failures += 1

    print()
    if failures:
        print(f"{failures} contract check(s) failed. Do not trust a sweep until resolved.")
        return 1
    print("All sub-agent contract checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
