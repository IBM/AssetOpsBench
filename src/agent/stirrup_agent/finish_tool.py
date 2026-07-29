"""Custom Stirrup finish tool that captures the final response explicitly."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from stirrup.constants import DEFAULT_FINISH_TOOL_NAME
from stirrup.core.models import Tool, ToolResult, ToolUseCountMetadata


class AssetOpsFinishParams(BaseModel):
    """Arguments the agent supplies when ending an AssetOps task."""

    answer: str = Field(
        min_length=1,
        description=(
            "Complete response to send directly to the user. Match any output "
            "format specified in the original request. Include only the requested "
            "content, with no status update, internal reasoning, or additional "
            "commentary unless the user asked for it."
        ),
    )
    reason: str = Field(
        default="",
        description=(
            "Optional internal note explaining why the run is ending. "
            "This is operational metadata and is not shown to the user."
        ),
    )
    paths: list[str] = Field(
        default_factory=list,
        description=(
            "Paths of files created or modified for the user. List individual "
            "files only; do not list directories."
        ),
    )

    @field_validator("answer")
    @classmethod
    def _answer_must_contain_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer must include at least one non-whitespace character")
        return value


async def _finish_executor(
    params: AssetOpsFinishParams,
) -> ToolResult[ToolUseCountMetadata]:
    """Accept completion only when every reported output file exists."""
    from stirrup.core.agent import _SESSION_STATE

    try:
        state = _SESSION_STATE.get(None)
        exec_env = state.exec_env if state else None
    except LookupError:
        exec_env = None

    if exec_env and params.paths:
        missing = [path for path in params.paths if not await exec_env.file_exists(path)]
        if missing:
            return ToolResult(
                content=(
                    f"Cannot finish: reported output files do not exist: {missing}. "
                    "Check each path and save the files before trying again."
                ),
                metadata=ToolUseCountMetadata(),
                success=False,
            )

    return ToolResult(
        content=params.answer,
        metadata=ToolUseCountMetadata(),
        success=True,
    )


ASSETOPS_FINISH_TOOL: Tool[AssetOpsFinishParams, ToolUseCountMetadata] = Tool(
    name=DEFAULT_FINISH_TOOL_NAME,
    description=(
        "End the task when all available work is complete or no further progress "
        "is possible. Put the complete response for the user in answer, an "
        "optional internal completion note in reason, and any created or modified "
        "file paths in paths. Call this tool alone on the final turn."
    ),
    parameters=AssetOpsFinishParams,
    executor=_finish_executor,
)


def structured_finish_answer(finish_params: object) -> str | None:
    """Return the final response, or ``None`` for non-custom finish data."""
    answer = getattr(finish_params, "answer", None)
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    return None
