"""Persist large MCP results in the active Stirrup code workspace.

The LLM should not have to copy a large MCP response into ``code_exec`` just to
analyze it. This provider wraps Stirrup's MCP tools so oversized text results
are written to the existing code-execution environment and replaced in the
conversation with a compact, cacheable artifact handle.

It also copies the files MCP servers point at. A server such as TSFM writes its full
output to a results file on this host and returns a ``results_file`` pointer; agent code
runs in a separate sandbox that cannot open that path, so the bridge copies the file into
the workspace and adds the copy's path to the response.
"""

from __future__ import annotations

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

DEFAULT_PERSIST_THRESHOLD_BYTES = 100 * 1024
_ARTIFACT_DIRECTORY = "mcp_results"
# where MCP servers write the results files their pointers name (TSFM: TSFM_WORKDIR)
_DEFAULT_RESULT_FILE_ROOTS = (os.environ.get("TSFM_WORKDIR", "/tmp/tsfm_work"),)
_MAX_RESULT_FILE_BYTES = 50 * 1024 * 1024
_MAX_RESULT_FILES_PER_CALL = 20
_RESULT_FILES_NOTE = (
    "results_workspace_files maps each results_file pointer to a copy in the code "
    "workspace. Open that relative path in code_exec for the full payload; the pointer "
    "itself is a host path the sandbox cannot read."
)
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

    def tool_content(self, *, cached: bool = False) -> str:
        payload = {
            **asdict(self),
            "artifact_type": "mcp_result",
            "cached": cached,
            "instruction": (
                "Treat workspace_file as the complete, read-only MCP snapshot. "
                "Process it in place with code_exec and print only required "
                "fields or aggregates; never dump the entire file. Do not repeat "
                "this MCP read unless the underlying domain state has changed."
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


def _result_file_pointers(obj: Any, depth: int = 0):
    """Every ``results_file`` string in a tool response, including nested plan steps."""
    if depth > 4:
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "results_file" and isinstance(value, str):
                yield value
            else:
                yield from _result_file_pointers(value, depth + 1)
    elif isinstance(obj, list):
        for value in obj:
            yield from _result_file_pointers(value, depth + 1)


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
        persist_threshold_bytes: int = DEFAULT_PERSIST_THRESHOLD_BYTES,
        result_file_roots: tuple[str, ...] | None = None,
    ) -> None:
        if persist_threshold_bytes <= 0:
            raise ValueError("persist_threshold_bytes must be positive")
        super().__init__(config=config)
        self._exec_env = exec_env
        self._persist_threshold_bytes = persist_threshold_bytes
        self._artifacts: dict[str, MCPResultArtifact] = {}
        roots = (
            _DEFAULT_RESULT_FILE_ROOTS if result_file_roots is None else result_file_roots
        )
        self._result_file_roots = tuple(os.path.realpath(str(r)) for r in roots if r)

    async def _copy_result_files(self, content: str) -> str:
        """Copy each results file a response points at into the code workspace.

        Only files under the result-file roots are copied (realpath, so a symlink cannot
        reach outside them), each up to 50 MiB and at most 20 per call. Returns the
        response with ``results_workspace_files`` added, or unchanged when nothing was
        copied. A failed copy is logged and skipped; the response still goes through.
        """
        try:
            obj = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return content
        if not isinstance(obj, dict):
            return content
        copied: dict[str, str] = {}
        for ref in dict.fromkeys(_result_file_pointers(obj)):
            if len(copied) >= _MAX_RESULT_FILES_PER_CALL:
                break
            path = os.path.realpath(ref[7:] if ref.startswith("file://") else ref)
            if not any(path.startswith(root + os.sep) for root in self._result_file_roots):
                continue
            try:
                if os.path.getsize(path) > _MAX_RESULT_FILE_BYTES:
                    continue
                with open(path, "rb") as fh:
                    data = fh.read()
                dest = f"{_ARTIFACT_DIRECTORY}/{os.path.basename(path)}"
                await self._exec_env.write_file_bytes(dest, data)
                copied[ref] = dest
            except Exception:
                _log.warning("Failed to copy results file %s", ref, exc_info=True)
        if not copied:
            return content
        obj["results_workspace_files"] = copied
        obj["results_workspace_note"] = _RESULT_FILES_NOTE
        return json.dumps(obj, ensure_ascii=False)

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
                    content=artifact.tool_content(cached=True),
                    metadata=ToolUseCountMetadata(),
                )

            result = await original_executor(params)
            if result.success and tool.name in _MUTATING_TOOLS:
                self._artifacts.clear()

            if not result.success or not isinstance(result.content, str):
                return result

            content = await self._copy_result_files(result.content)
            if content is not result.content:
                result = ToolResult(
                    content=content,
                    success=result.success,
                    metadata=result.metadata,
                )

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
                content=artifact.tool_content(),
                success=result.success,
                metadata=result.metadata,
            )

        return Tool(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            executor=executor,
        )
