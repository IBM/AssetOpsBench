"""Translate a Stirrup run into AssetOpsBench's shared trajectory model.

Stirrup's :meth:`Agent.run` returns ``(finish_params, history, metadata)``
where ``history`` is a ``list[list[ChatMessage]]`` (a list of turns, each a
list of messages).  Stirrup's message objects are strongly typed pydantic
models, so unlike the Goose path there is no fragile JSONL parsing: we read
attributes directly.

Mapping:
  * each ``AssistantMessage`` -> one :class:`~agent.models.TurnRecord`
    (its ``content`` text, ``tool_calls``, ``token_usage``, request timing);
  * each ``ToolMessage`` -> the ``output`` of the matching :class:`ToolCall`,
    joined by ``tool_call_id``.

Stirrup 0.2.0 made assistant messages block-based: ``blocks`` is the stored
content and ``content`` / ``tool_calls`` survive only as deprecated read-only
projections. :func:`_assistant_text` and :func:`_assistant_tool_calls` read
blocks when present and fall back to the projections, so this module works
against both shapes and emits no deprecation warnings on 0.2.

Stirrup exposes MCP tools as ``{server}__{tool}`` and the code-execution tool
as ``code_exec``, so :func:`classify_tool` (shared shape with the Goose
runner) labels each call domain / code / other for the bypass metric.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from ..models import ToolCall, Trajectory, TurnRecord
from .gateway import GATEWAY_CALL_TOOL, GATEWAY_DISCOVERY_TOOLS

# Stirrup's built-in code-execution tool name (LocalCodeExec and Docker both
# register under this name by default).  A call to it = "the agent ran code".
_CODE_TOOL_NAMES = {"code_exec"}
# Default web tools, if ever attached; counted as "other", never domain.
_WEB_TOOL_NAMES = {"web_search", "web_fetch"}


def classify_tool(
    tool_name: str,
    domain_servers: set[str],
    arguments: dict | None = None,
) -> str:
    """Label a Stirrup tool call ``"domain"`` / ``"code"`` / ``"other"``.

    MCP tools arrive as ``{server}__{tool}``; ``code_exec`` is code execution;
    anything else (web, finish, discovery, ...) is ``"other"``.

    Under ``--topology gateway`` every domain call arrives as ``call_tool`` with
    the real tool in its ``name`` argument. Without ``arguments`` that would
    classify as "other", zeroing ``agent.domain_tool_calls`` and making
    ``tool_bypass`` read true on every gateway run. Pass the call's arguments
    and the underlying server is credited instead, so the counts stay
    comparable with a flat run.
    """
    if tool_name in _CODE_TOOL_NAMES:
        return "code"
    if tool_name in _WEB_TOOL_NAMES:
        return "other"
    if tool_name == GATEWAY_CALL_TOOL:
        inner = (arguments or {}).get("name")
        if isinstance(inner, str) and inner.split("__", 1)[0] in domain_servers:
            return "domain"
        return "other"
    if tool_name in GATEWAY_DISCOVERY_TOOLS:
        # Discovery is overhead the gateway pays and flat does not. Counted
        # separately rather than as domain work.
        return "other"
    prefix = tool_name.split("__", 1)[0]
    if prefix in domain_servers:
        return "domain"
    return "other"


def _content_text(content: Any) -> str:
    """Flatten Stirrup ``Content`` (``str | list[ContentBlock]``) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _assistant_text(msg: Any) -> str:
    """Text of an assistant message, preferring 0.2 blocks over projections."""
    blocks = getattr(msg, "blocks", None)
    if blocks:
        parts = [
            block.text
            for block in blocks
            if getattr(block, "kind", None) == "text"
            and isinstance(getattr(block, "text", None), str)
        ]
        return "".join(parts)
    return _content_text(getattr(msg, "content", ""))


def _assistant_tool_calls(msg: Any) -> list[Any]:
    """Tool calls of an assistant message, in emission order."""
    blocks = getattr(msg, "blocks", None)
    if blocks:
        return [b for b in blocks if getattr(b, "kind", None) == "tool_call"]
    return list(getattr(msg, "tool_calls", []) or [])


def _parse_arguments(arguments: Any) -> dict:
    """Stirrup ToolCall.arguments is a JSON string; parse defensively."""
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {"_raw": parsed}
        except json.JSONDecodeError:
            return {"_raw": arguments}
    return {}


def _ms(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    delta = end - start
    return delta * 1000 if delta >= 0 else None


def _flatten(history: Iterable[Any]) -> list[Any]:
    """history is list[list[ChatMessage]]; flatten to a single message list."""
    flat: list[Any] = []
    for turn in history:
        if isinstance(turn, list):
            flat.extend(turn)
        else:
            flat.append(turn)
    return flat


def build_trajectory(history: Iterable[Any]) -> Trajectory:
    """Convert a Stirrup message history into a :class:`Trajectory`."""
    trajectory = Trajectory()
    turn_index = 0
    by_id: dict[str, ToolCall] = {}

    for msg in _flatten(history):
        role = getattr(msg, "role", None)

        if role == "assistant":
            tool_calls: list[ToolCall] = []
            for tc in _assistant_tool_calls(msg):
                call = ToolCall(
                    name=getattr(tc, "name", "") or "",
                    input=_parse_arguments(getattr(tc, "arguments", "")),
                    id=getattr(tc, "tool_call_id", "") or "",
                )
                tool_calls.append(call)
                if call.id:
                    by_id[call.id] = call

            usage = getattr(msg, "token_usage", None)
            in_tok = getattr(usage, "input", 0) if usage else 0
            out_tok = getattr(usage, "output", 0) if usage else 0

            trajectory.turns.append(
                TurnRecord(
                    index=turn_index,
                    text=_assistant_text(msg),
                    tool_calls=tool_calls,
                    input_tokens=int(in_tok or 0),
                    output_tokens=int(out_tok or 0),
                    duration_ms=_ms(
                        getattr(msg, "request_start_time", None),
                        getattr(msg, "request_end_time", None),
                    ),
                )
            )
            turn_index += 1

        elif role == "tool":
            call = by_id.get(getattr(msg, "tool_call_id", "") or "")
            if call is not None:
                call.output = _content_text(getattr(msg, "content", ""))
                call.duration_ms = _ms(
                    getattr(msg, "tool_start_time", None),
                    getattr(msg, "tool_end_time", None),
                )

    return trajectory


def final_answer(history: Iterable[Any], finish_params: Any) -> str:
    """Return the user-facing answer from a completed Stirrup run.

    Prefer the explicit ``finish.answer`` supplied by the AssetOps finish tool.
    For legacy/default finish tools, ``finish.reason`` describes why the agent
    stopped and is not necessarily the answer shown to the user, so prefer
    non-empty assistant content emitted alongside the ``finish`` call while
    retaining the reason as a fallback.
    """
    structured_answer = getattr(finish_params, "answer", None)
    if isinstance(structured_answer, str) and structured_answer.strip():
        return structured_answer.strip()

    messages = _flatten(history)

    for msg in reversed(messages):
        if getattr(msg, "role", None) != "assistant":
            continue
        if any(
            getattr(call, "name", None) == "finish"
            for call in _assistant_tool_calls(msg)
        ):
            text = _assistant_text(msg).strip()
            if text:
                return text

    reason = getattr(finish_params, "reason", None)
    if isinstance(reason, str) and reason.strip():
        return reason.strip()

    for msg in reversed(messages):
        if getattr(msg, "role", None) == "assistant":
            text = _assistant_text(msg).strip()
            if text:
                return text
    return ""
