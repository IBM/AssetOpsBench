from pathlib import Path
from pydantic import BaseModel, Field
from stirrup.core.models import Tool, ToolResult, ToolUseCountMetadata
from stirrup.tools.code_backends.base import CodeExecToolProvider

_WORKSPACE = "/workspace"


class HostPathParams(BaseModel):
    workspace_path: str = Field(
        description="A file you created in code_exec, e.g. '/workspace/chiller6.csv' "
                    "or just 'chiller6.csv'."
    )


def build_handoff_tools(exec_env: CodeExecToolProvider) -> list[Tool]:
    async def host_path(params: HostPathParams) -> ToolResult:
        temp_dir = exec_env.temp_dir
        if temp_dir is None:
            return ToolResult(content="Code workspace is not running.",
                              success=False, metadata=ToolUseCountMetadata())

        rel = params.workspace_path.removeprefix(_WORKSPACE + "/").lstrip("/")
        host = Path(temp_dir) / rel

        try:                                    # refuse ../.. escapes
            host.resolve().relative_to(Path(temp_dir).resolve())
        except ValueError:
            return ToolResult(content=f"Outside the workspace: {params.workspace_path}",
                              success=False, metadata=ToolUseCountMetadata())

        if not host.is_file():
            return ToolResult(
                content=(f"No such file: {params.workspace_path}. Create it with "
                         f"code_exec first, then call this again."),
                success=False, metadata=ToolUseCountMetadata())

        return ToolResult(content=str(host), metadata=ToolUseCountMetadata())

    return [Tool(
        name="workspace_host_path",
        description=(
            "Translate a code_exec file path into the host path MCP tools need. "
            "MCP tools (tsfm, iot, wo, fmsr, vibration) run on the host and cannot "
            "open /workspace paths. Call this with the file you created, then pass "
            "the returned path as dataset_path / data_ref / file arguments."
        ),
        parameters=HostPathParams,
        executor=host_path,
    )]