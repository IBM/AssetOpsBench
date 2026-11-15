"""The self-ask tool.

Self-ask is exposed to the model as a tool rather than forced before every
action, which is a deliberate deviation from ReActXen as published.  The cost
of that choice is that a model may never call it, so the runner counts
invocations and :meth:`ReActXenTrajectory.self_ask_rate` reports them.  A
condition is only on if the counter says it ran.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .prompts import (
    SELF_ASK_SYSTEM,
    SELF_ASK_TOOL_DESCRIPTION,
    SELF_ASK_TOOL_NAME,
)

_log = logging.getLogger(__name__)

SELF_ASK_TOOL_SCHEMA: dict[str, Any] = {
    "name": SELF_ASK_TOOL_NAME,
    "description": SELF_ASK_TOOL_DESCRIPTION,
    "input_schema": {
        "type": "object",
        "properties": {
            "context": {
                "type": "string",
                "description": (
                    "What you are trying to establish and what you have "
                    "already retrieved."
                ),
            }
        },
        "required": ["context"],
    },
}


@dataclass
class SelfAskTool:
    """Callable tool backed by one short model call.

    Args:
        complete: ``async (system, user) -> (text, in_tokens, out_tokens)``.
            Supplied by the runner so this module stays free of any
            provider or router specifics.
    """

    complete: Any
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    log: list[str] = field(default_factory=list)

    @property
    def schema(self) -> dict[str, Any]:
        return SELF_ASK_TOOL_SCHEMA

    async def __call__(self, context: str) -> str:
        self.calls += 1
        try:
            text, in_tok, out_tok = await self.complete(SELF_ASK_SYSTEM, context)
        except Exception:  # noqa: BLE001 - a decomposition failure must not end the run
            # A decomposition failure must not end the run; the agent simply
            # proceeds without the sub-question.
            _log.warning("self_ask call failed", exc_info=True)
            return "Sub-question unavailable. Proceed with the request as stated."
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.log.append(text)
        return text

    def reset_round(self) -> int:
        """Return this round's call count and zero the counter."""
        calls, self.calls = self.calls, 0
        return calls

    def as_stirrup_tool(self):
        """Adapt the callable to Stirrup's typed tool interface."""
        from pydantic import BaseModel, Field
        from stirrup.core.models import Tool, ToolResult, ToolUseCountMetadata

        class SelfAskParams(BaseModel):
            context: str = Field(
                description="Current task and evidence retrieved so far."
            )

        async def execute(params: SelfAskParams):
            return ToolResult(
                content=await self(params.context), metadata=ToolUseCountMetadata()
            )

        return Tool(
            name=SELF_ASK_TOOL_NAME,
            description=SELF_ASK_TOOL_DESCRIPTION,
            parameters=SelfAskParams,
            executor=execute,
        )
