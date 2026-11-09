"""A bounded finish-only continuation after the normal ReAct budget ends."""

import json

from pydantic import ValidationError
from stirrup.core.models import UserMessage

from .trajectory import _assistant_tool_calls


def finish_feedback(messages, finish_tool):
    """Surface specific schema errors without copying unvalidated model prose."""
    for message in reversed(messages):
        if getattr(message, "role", None) != "assistant":
            continue
        for call in reversed(_assistant_tool_calls(message)):
            if getattr(call, "name", None) != finish_tool.name:
                continue
            arguments = getattr(call, "arguments", None)
            if arguments is None:
                arguments = getattr(call, "input", {})
            try:
                if isinstance(arguments, str):
                    finish_tool.parameters.model_validate_json(arguments)
                else:
                    finish_tool.parameters.model_validate(arguments)
            except ValidationError as exc:
                return json.dumps(
                    [
                        {"field": ".".join(map(str, err["loc"])), "error": err["msg"]}
                        for err in exc.errors(include_input=False, include_url=False)
                    ]
                )
            # Schema passed: selection/path constraints can still reject execution.
            for result in reversed(messages):
                if (
                    getattr(result, "role", None) == "tool"
                    and getattr(result, "name", None) == finish_tool.name
                ):
                    return str(getattr(result, "content", ""))
            return (
                "Finish was not accepted; check its schema and selection constraints."
            )
    return "The normal analysis budget ended without an accepted finish call."


async def repair_finish(session, finish_tool, history, max_attempts):
    """Reuse the live conversation, expose only finish, and retain repair turns.

    No new code/MCP calls can execute during repair. All added model responses
    and tool results stay in history so trajectory token totals include them.
    """
    if max_attempts <= 0:
        return None, 0
    if not history:
        raise ValueError("Finish repair requires the existing conversation")
    messages = history[-1]
    original_tools = session._active_tools
    session._active_tools = {finish_tool.name: finish_tool}
    attempts = 0
    try:
        for attempt in range(max_attempts):
            attempts += 1
            messages.append(
                UserMessage(
                    content=(
                        f"Finish repair {attempt + 1}/{max_attempts}. Analysis is over. "
                        "Only the finish tool is available: do not request code execution, MCP, "
                        "or new data. Use the evidence and decision already established. "
                        "Call finish exactly once with the typed fields in its schema. "
                        "evidence_paths must be arrays of nonempty strings, including inside each "
                        "alternative assessment. Respect the field lengths and assess every offered "
                        "alternative. paths must list existing individual files, never directories; "
                        "use paths=[] when no file export is needed. Return final conclusions, never draft deliberation. "
                        "Previous validation feedback: "
                        + finish_feedback(messages, finish_tool)
                    )
                )
            )
            assistant, results, params = await session.step(messages, {})
            messages.extend([assistant, *results])
            if params is not None:
                return params, attempts
        return None, attempts
    finally:
        session._active_tools = original_tools
