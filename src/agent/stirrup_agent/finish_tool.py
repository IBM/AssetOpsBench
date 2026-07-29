"""AssetOps-specific Stirrup finish tool with an explicit answer contract."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from stirrup.constants import DEFAULT_FINISH_TOOL_NAME
from stirrup.core.models import Tool, ToolResult, ToolUseCountMetadata


class AssetOpsFinishParams(BaseModel):
    """Structured result returned when an AssetOps task is complete."""

    answer: str = Field(
        min_length=1,
        description=(
            "Final answer to send directly to the user. Follow the output format "
            "requested in the original question exactly. Include no completion "
            "summary, reasoning, or extra text unless the user requested it."
        ),
    )
    reason: str = Field(
        default="",
        description=(
            "Optional internal explanation of why the task is complete. "
            "Do not place the user-facing answer here."
        ),
    )
    paths: list[str] = Field(
        default_factory=list,
        description=(
            "Files created or modified by the task. Include files only, not "
            "directories."
        ),
    )

    @field_validator("answer")
    @classmethod
    def _answer_must_contain_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer must contain non-whitespace content")
        return value


async def _finish_executor(
    params: AssetOpsFinishParams,
) -> ToolResult[ToolUseCountMetadata]:
    """Validate reported output files before accepting task completion."""
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
                    f"ERROR: Files do not exist: {missing}. Verify paths and "
                    "ensure files were saved."
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
        "Finish the task and provide the exact user-facing answer separately "
        "from any internal completion reason. Call this only when the task is "
        "complete; a separate turn is required to finish."
    ),
    parameters=AssetOpsFinishParams,
    executor=_finish_executor,
)


def structured_finish_answer(finish_params: object) -> str | None:
    """Extract a non-empty answer from custom finish parameters."""
    answer = getattr(finish_params, "answer", None)
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    return None
