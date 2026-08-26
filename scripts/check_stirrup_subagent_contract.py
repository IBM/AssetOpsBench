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

import inspect
import sys


def _fail(message: str) -> None:
    print(f"FAIL  {message}")


def _ok(message: str) -> None:
    print(f"ok    {message}")


def main() -> int:
    failures = 0

    import stirrup
    from stirrup import Agent
    from stirrup.core.agent import SubAgentParams
    from stirrup.core.models import SubAgentMetadata
    from stirrup.tools.mcp import MCPToolProvider

    version = getattr(stirrup, "__version__", "unknown")
    print(f"stirrup {version}\n")

    # 1. The client carries the working-context budget explicitly. The runner
    #    dropped its `max_tokens` adapter because of this.
    sig = inspect.signature(Agent.__init__)
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
        _ok("Agent(share_parent_exec_env=...) present (we pass False on purpose)")
    else:
        _fail(
            "Agent no longer takes share_parent_exec_env; re-read how a sub-agent "
            "acquires an exec env before trusting subagents.py's invariant 1."
        )
        failures += 1

    # 3. The whole context saving rests on this: to_tool must return the
    #    sub-agent's message history as tool *metadata*, not as tool content.
    src = inspect.getsource(Agent.to_tool)
    history_in_metadata = "SubAgentMetadata(" in src and "metadata=sub_metadata" in src
    content_is_composed = "result_content" in src and "<sub_agent_result>" in src
    if history_in_metadata and content_is_composed:
        _ok("to_tool returns message_history via metadata, not in tool content")
    else:
        _fail(
            "to_tool no longer separates history from tool content. If the full "
            "history now reaches the parent's context, --topology subagent saves "
            "nothing and the design needs revisiting."
        )
        failures += 1

    # 4. Only the last assistant text and a model_dump() of finish params cross
    #    back, which is why DomainFinishParams is typed.
    if "finish_params.model_dump()" in src:
        _ok("finish params reach the parent via model_dump()")
    else:
        _fail(
            "to_tool no longer dumps finish params into the result. Domain "
            "sub-agents would lose their artifact handles and entity ids."
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
