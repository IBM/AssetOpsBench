"""Tests for automatic persistence of large MCP tool results."""

from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import BaseModel
from stirrup.core.models import Tool, ToolResult, ToolUseCountMetadata
from stirrup.tools.mcp import MCPConfig

from agent.stirrup_agent.workspace_bridge import (
    WorkspaceBridgedMCPToolProvider,
)


class _Params(BaseModel):
    page_size: int = 0
    failure_code: str | None = None


class _FakeExecEnvironment:
    def __init__(self, *, fail_writes: bool = False) -> None:
        self.files: dict[str, bytes] = {}
        self.fail_writes = fail_writes

    async def write_file_bytes(self, path: str, content: bytes) -> None:
        if self.fail_writes:
            raise RuntimeError("write failed")
        self.files[path] = content

    async def file_exists(self, path: str) -> bool:
        return path in self.files

    async def read_file_bytes(self, path: str) -> bytes:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]


def _provider(
    exec_env: _FakeExecEnvironment,
    *,
    threshold: int = 32,
) -> WorkspaceBridgedMCPToolProvider:
    config = MCPConfig.model_validate({"mcpServers": {}})
    return WorkspaceBridgedMCPToolProvider(
        config=config,
        exec_env=exec_env,  # type: ignore[arg-type]
        persist_threshold_bytes=threshold,
    )


def _tool(name: str, content: str, calls: list[str]) -> Tool:
    async def executor(params: _Params) -> ToolResult[ToolUseCountMetadata]:
        calls.append(name)
        return ToolResult(
            content=content,
            metadata=ToolUseCountMetadata(),
        )

    return Tool(
        name=name,
        description="Test MCP tool",
        parameters=_Params,
        executor=executor,
    )


@pytest.mark.anyio
async def test_large_result_is_persisted_and_reused() -> None:
    exec_env = _FakeExecEnvironment()
    provider = _provider(exec_env)
    calls: list[str] = []
    content = json.dumps({"work_orders": [{"description": "x" * 200}]})
    tool = provider._wrap_tool(_tool("wo__list_workorders", content, calls))

    assert tool.description == "Test MCP tool"

    first = await tool.executor(_Params())
    first_handle = json.loads(first.content)

    assert calls == ["wo__list_workorders"]
    assert first_handle["cached"] is False
    assert first_handle["workspace_file"].startswith(
        "mcp_results/wo__list_workorders_"
    )
    assert exec_env.files[first_handle["workspace_file"]] == content.encode()
    assert first_handle["bytes"] == len(content.encode())
    assert first_handle["sha256"] == hashlib.sha256(content.encode()).hexdigest()
    assert first_handle["arguments"]["failure_code"] is None
    assert "never dump the entire file" in first_handle["instruction"]

    second = await tool.executor(_Params())
    second_handle = json.loads(second.content)

    assert calls == ["wo__list_workorders"]
    assert second_handle["cached"] is True
    assert second_handle["workspace_file"] == first_handle["workspace_file"]


@pytest.mark.anyio
async def test_small_result_remains_inline() -> None:
    exec_env = _FakeExecEnvironment()
    provider = _provider(exec_env, threshold=1_000)
    calls: list[str] = []
    tool = provider._wrap_tool(_tool("wo__get_failure_codes", "{}", calls))

    result = await tool.executor(_Params())

    assert result.content == "{}"
    assert calls == ["wo__get_failure_codes"]
    assert exec_env.files == {}


@pytest.mark.anyio
async def test_modified_artifact_is_refetched() -> None:
    exec_env = _FakeExecEnvironment()
    provider = _provider(exec_env)
    calls: list[str] = []
    content = json.dumps({"work_orders": ["x" * 200]})
    tool = provider._wrap_tool(_tool("wo__list_workorders", content, calls))

    first = await tool.executor(_Params())
    handle = json.loads(first.content)
    exec_env.files[handle["workspace_file"]] = b"modified"

    await tool.executor(_Params())

    assert calls == ["wo__list_workorders", "wo__list_workorders"]
    assert exec_env.files[handle["workspace_file"]] == content.encode()


@pytest.mark.anyio
async def test_persistence_failure_falls_back_to_inline_result() -> None:
    exec_env = _FakeExecEnvironment(fail_writes=True)
    provider = _provider(exec_env)
    calls: list[str] = []
    content = "x" * 200
    tool = provider._wrap_tool(_tool("wo__list_workorders", content, calls))

    result = await tool.executor(_Params())

    assert result.content == content
    assert calls == ["wo__list_workorders"]


@pytest.mark.anyio
async def test_work_order_mutation_invalidates_cached_reads() -> None:
    exec_env = _FakeExecEnvironment()
    provider = _provider(exec_env)
    read_calls: list[str] = []
    mutation_calls: list[str] = []
    read_content = json.dumps({"work_orders": ["x" * 200]})
    read_tool = provider._wrap_tool(
        _tool("wo__list_workorders", read_content, read_calls)
    )
    mutation_tool = provider._wrap_tool(
        _tool("wo__update_workorder", '{"success": true}', mutation_calls)
    )

    await read_tool.executor(_Params())
    await read_tool.executor(_Params())
    assert len(read_calls) == 1

    await mutation_tool.executor(_Params())
    await read_tool.executor(_Params())

    assert mutation_calls == ["wo__update_workorder"]
    assert len(read_calls) == 2


@pytest.mark.anyio
async def test_refetched_snapshot_does_not_overwrite_prior_artifact() -> None:
    exec_env = _FakeExecEnvironment()
    provider = _provider(exec_env)
    responses = iter(
        [
            json.dumps({"work_orders": ["old" * 100]}),
            json.dumps({"work_orders": ["new" * 100]}),
        ]
    )

    async def read_executor(
        params: _Params,
    ) -> ToolResult[ToolUseCountMetadata]:
        return ToolResult(
            content=next(responses),
            metadata=ToolUseCountMetadata(),
        )

    read_tool = provider._wrap_tool(
        Tool(
            name="wo__list_workorders",
            description="Test MCP tool",
            parameters=_Params,
            executor=read_executor,
        )
    )
    mutation_tool = provider._wrap_tool(
        _tool("wo__update_workorder", '{"success": true}', [])
    )

    first = json.loads((await read_tool.executor(_Params())).content)
    await mutation_tool.executor(_Params())
    second = json.loads((await read_tool.executor(_Params())).content)

    assert first["workspace_file"] != second["workspace_file"]
    assert b"old" in exec_env.files[first["workspace_file"]]
    assert b"new" in exec_env.files[second["workspace_file"]]


@pytest.mark.anyio
async def test_non_work_order_mutation_invalidates_cached_reads() -> None:
    exec_env = _FakeExecEnvironment()
    provider = _provider(exec_env)
    read_calls: list[str] = []
    read_tool = provider._wrap_tool(
        _tool(
            "tsfm__list_models",
            json.dumps({"models": ["x" * 200]}),
            read_calls,
        )
    )
    mutation_tool = provider._wrap_tool(
        _tool("tsfm__register_model", '{"status": "registered"}', [])
    )

    await read_tool.executor(_Params())
    await read_tool.executor(_Params())
    await mutation_tool.executor(_Params())
    await read_tool.executor(_Params())

    assert len(read_calls) == 2


@pytest.mark.anyio
async def test_mutation_calls_are_never_cached() -> None:
    exec_env = _FakeExecEnvironment()
    provider = _provider(exec_env)
    calls: list[str] = []
    content = json.dumps({"work_order": "x" * 200})
    tool = provider._wrap_tool(_tool("wo__update_workorder", content, calls))

    await tool.executor(_Params())
    await tool.executor(_Params())

    assert calls == ["wo__update_workorder", "wo__update_workorder"]
