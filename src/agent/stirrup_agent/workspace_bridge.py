"""Persist large MCP results in the active Stirrup code workspace.

The LLM should not have to copy a large MCP response into ``code_exec`` just to
analyze it. This provider wraps Stirrup's MCP tools so oversized text results
are written to the existing code-execution environment and replaced in the
conversation with a compact, cacheable artifact handle.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

from pydantic import BaseModel
from stirrup.core.models import Tool, ToolResult, ToolUseCountMetadata
from stirrup.tools.code_backends.base import CodeExecToolProvider
from stirrup.tools.mcp import MCPConfig, MCPToolProvider

_log = logging.getLogger(__name__)

# Above this size a result becomes a workspace handle instead of conversation
# text. 100 KiB was far too generous: a typical paginated MCP page lands near
# 45 KiB, so it rode inline and a 40k-token sub-agent filled its context in
# three calls, summarized away the rows it had just fetched, re-fetched, and
# died at its turn cap. The threshold must sit below one page, not above it.
DEFAULT_PERSIST_THRESHOLD_BYTES = int(
    os.environ.get("STIRRUP_MCP_SPILL_BYTES", 16 * 1024)
)
# A sub-agent has a smaller budget than the root and no code_exec, so it should
# hold even less raw text. Defaults to half the root's threshold.
SUBAGENT_PERSIST_THRESHOLD_BYTES = int(
    os.environ.get(
        "STIRRUP_SUBAGENT_MCP_SPILL_BYTES", DEFAULT_PERSIST_THRESHOLD_BYTES // 2
    )
)
# Stirrup builds its stdio ClientSession without read_timeout_seconds and awaits
# call_tool bare, so a server that never answers hangs the run forever.
MCP_TOOL_TIMEOUT_S = float(os.environ.get("STIRRUP_MCP_TOOL_TIMEOUT", 300))

# What to tell the recipient of a handle. Only the root holds code_exec; a
# sub-agent that is told to "process it with code_exec" is being pointed at a
# tool it does not have, and answers by paginating the data into its context
# instead. It should pass the handle up untouched.
_CODE_EXEC_INSTRUCTION = (
    "Treat workspace_file as the complete, read-only MCP snapshot. "
    "Process it in place with code_exec and print only required fields or "
    "aggregates; never dump the entire file. Do not repeat this MCP read "
    "unless the underlying domain state has changed."
)
_HANDOFF_INSTRUCTION = (
    "Treat workspace_file as the complete, read-only MCP snapshot. You have no "
    "code_exec: do NOT try to read, page through, or re-fetch this data. Copy "
    "this handle verbatim into your finish `artifacts` field and describe in "
    "`answer` what it contains; the root agent analyses it. Do not repeat this "
    "MCP read unless the underlying domain state has changed."
)
_ARTIFACT_DIRECTORY = "mcp_results"
_MUTATING_TOOLS = {
    "fmsr__add_failure_modes",
    "tsfm__deprecate_feature",
    "tsfm__deprecate_model",
    "tsfm__new_feature_version",
    "tsfm__new_model_version",
    "tsfm__register_feature",
    "tsfm__register_finetuned",
    "tsfm__register_model",
    "tsfm__run_plan",
    "tsfm__run_recipe",
    "tsfm__run_tabular_recipe",
    "tsfm__update_feature",
    "tsfm__update_model",
    "wo__generate_work_order",
    "wo__update_workorder",
    "wo__approve_workorder",
    "wo__assign_technician",
    "wo__close_workorder",
    "wo__cancel_workorder",
}


@dataclass(frozen=True)
class MCPResultArtifact:
    """A durable workspace snapshot of one MCP tool result."""

    workspace_file: str
    tool: str
    arguments: dict[str, Any]
    bytes: int
    sha256: str

    def tool_content(
        self, *, cached: bool = False, reader_has_code_exec: bool = True
    ) -> str:
        payload = {
            **asdict(self),
            "artifact_type": "mcp_result",
            "cached": cached,
            "instruction": (
                _CODE_EXEC_INSTRUCTION
                if reader_has_code_exec
                else _HANDOFF_INSTRUCTION
            ),
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _normalized_arguments(params: BaseModel) -> dict[str, Any]:
    return params.model_dump(mode="json")


def _query_key(tool_name: str, arguments: dict[str, Any]) -> str:
    normalized = json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(f"{tool_name}:{normalized}".encode()).hexdigest()


def _artifact_extension(content: str) -> str:
    try:
        json.loads(content)
    except json.JSONDecodeError:
        return "txt"
    return "json"


def _artifact_path(
    tool_name: str,
    query_key: str,
    content_hash: str,
    extension: str,
) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", tool_name)
    return (
        f"{_ARTIFACT_DIRECTORY}/{safe_name}_{query_key[:12]}_"
        f"{content_hash[:12]}.{extension}"
    )


class WorkspaceBridgedMCPToolProvider(MCPToolProvider):
    """MCP provider that spills oversized text results into ``code_exec``."""

    def __init__(
        self,
        config: MCPConfig,
        *,
        exec_env: CodeExecToolProvider,
        server_names: list[str] | None = None,
        persist_threshold_bytes: int = DEFAULT_PERSIST_THRESHOLD_BYTES,
        reader_has_code_exec: bool = True,
    ) -> None:
        """Bridge oversized MCP results into ``exec_env``.

        ``server_names`` restricts this provider to a subset of the servers in
        ``config``; ``None`` connects to all of them. That subset is how a
        domain sub-agent gets exactly one server without needing its own
        config object.

        ``exec_env`` is a constructor argument, never a tool. A sub-agent given
        this provider can therefore spill a large result into the root's
        workspace while having no ability to execute code itself.
        """
        if persist_threshold_bytes <= 0:
            raise ValueError("persist_threshold_bytes must be positive")
        super().__init__(config=config, server_names=server_names)
        self._exec_env = exec_env
        self._persist_threshold_bytes = persist_threshold_bytes
        self._reader_has_code_exec = reader_has_code_exec
        self._artifacts: dict[str, MCPResultArtifact] = {}

    async def _artifact_is_intact(self, artifact: MCPResultArtifact) -> bool:
        try:
            payload = await self._exec_env.read_file_bytes(artifact.workspace_file)
        except Exception:
            _log.debug(
                "Unable to validate cached MCP artifact %s",
                artifact.workspace_file,
                exc_info=True,
            )
            return False
        return hashlib.sha256(payload).hexdigest() == artifact.sha256

    async def __aenter__(self) -> list[Tool[Any, ToolUseCountMetadata]]:
        tools = await super().__aenter__()
        return [self._wrap_tool(tool) for tool in tools]

    def _wrap_tool(
        self, tool: Tool[Any, ToolUseCountMetadata]
    ) -> Tool[Any, ToolUseCountMetadata]:
        original_executor = tool.executor

        async def executor(
            params: BaseModel,
        ) -> ToolResult[ToolUseCountMetadata]:
            arguments = _normalized_arguments(params)
            query_key = _query_key(tool.name, arguments)
            cacheable = tool.name not in _MUTATING_TOOLS
            artifact = self._artifacts.get(query_key) if cacheable else None

            if artifact is not None and await self._artifact_is_intact(artifact):
                return ToolResult(
                    content=artifact.tool_content(
                        cached=True,
                        reader_has_code_exec=self._reader_has_code_exec,
                    ),
                    metadata=ToolUseCountMetadata(),
                )

            try:
                result = await asyncio.wait_for(
                    original_executor(params), timeout=MCP_TOOL_TIMEOUT_S
                )
            except (asyncio.TimeoutError, TimeoutError):
                _log.warning(
                    "MCP tool %s timed out after %.0fs", tool.name, MCP_TOOL_TIMEOUT_S
                )
                return ToolResult(
                    content=(
                        f"{tool.name} did not respond within "
                        f"{MCP_TOOL_TIMEOUT_S:.0f}s and was cancelled. The server may "
                        "be unavailable. Try a narrower query or a different tool."
                    ),
                    success=False,
                    metadata=ToolUseCountMetadata(),
                )
            if result.success and tool.name in _MUTATING_TOOLS:
                self._artifacts.clear()

            if not result.success or not isinstance(result.content, str):
                return result

            payload = result.content.encode("utf-8")
            if len(payload) <= self._persist_threshold_bytes:
                return result

            content_hash = hashlib.sha256(payload).hexdigest()
            path = _artifact_path(
                tool.name,
                query_key,
                content_hash,
                _artifact_extension(result.content),
            )
            try:
                await self._exec_env.write_file_bytes(path, payload)
            except Exception:
                _log.warning(
                    "Failed to persist large MCP result from %s; returning it inline",
                    tool.name,
                    exc_info=True,
                )
                return result

            artifact = MCPResultArtifact(
                workspace_file=path,
                tool=tool.name,
                arguments=arguments,
                bytes=len(payload),
                sha256=content_hash,
            )
            if cacheable:
                self._artifacts[query_key] = artifact
            return ToolResult(
                content=artifact.tool_content(
                    reader_has_code_exec=self._reader_has_code_exec
                ),
                success=result.success,
                metadata=result.metadata,
            )

        return Tool(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            executor=executor,
        )