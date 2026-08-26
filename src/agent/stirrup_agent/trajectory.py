"""Translate a Stirrup run into AssetOpsBench's shared trajectory model.

Stirrup's :meth:`Agent.run` returns ``(finish_params, history, metadata)``
where ``history`` is a ``list[list[ChatMessage]]`` (a list of turns, each a
list of messages).  Stirrup's message objects are strongly typed pydantic
models, so unlike the Goose path there is no fragile JSONL parsing: we read
attributes directly.

Mapping:
  * each ``AssistantMessage`` -> one :class:`~agent.models.TurnRecord`
    (its text blocks, tool calls, ``token_usage``, request timing);
  * each ``ToolMessage`` -> the ``output`` of the matching :class:`ToolCall`,
    joined by ``tool_call_id``.

Stirrup 0.2.0 made assistant messages block-based: ``blocks`` is the stored
content and ``content`` / ``tool_calls`` survive only as deprecated read-only
projections.  :func:`_assistant_text` and :func:`_assistant_tool_calls` read
blocks when they are present and fall back to the projections, so this module
works against both shapes and emits no deprecation warnings on 0.2.

Sub-agent flattening
--------------------
Under ``--topology subagent`` the root's own history shows a single tool call
named ``{server}_agent`` where a flat run would show a series of
``{server}__{tool}`` calls.  Left alone that would silently zero
``agent.domain_tool_calls`` and make the ``tool_bypass`` metric meaningless,
because the domain work happened one level down.  :func:`build_trajectory`
therefore accepts the recorded sub-agent histories and splices each one into
the turn list directly after the root turn that invoked it, tagged with
``agent`` and ``depth``.  Tool counts and token totals are then tree-wide by
construction, and root-only figures are a filter on ``depth == 0``.

Stirrup exposes MCP tools as ``{server}__{tool}`` and the code-execution tool
as ``code_exec``, so :func:`classify_tool` (shared shape with the Goose
runner) labels each call domain / code / other for the bypass metric.  A
delegation call is deliberately *not* domain: it is named ``{server}_agent``
with no ``__`` separator, so it falls through to "other" while the real domain
calls spliced beneath it are counted once.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any, Iterable, Mapping

from ..models import ToolCall, Trajectory, TurnRecord

# Stirrup's built-in code-execution tool name (LocalCodeExec and Docker both
# register under this name by default).  A call to it = "the agent ran code".
_CODE_TOOL_NAMES = {"code_exec"}
# Default web tools, if ever attached; counted as "other", never domain.
_WEB_TOOL_NAMES = {"web_search", "web_fetch"}


def classify_tool(tool_name: str, domain_servers: set[str]) -> str:
    """Label a Stirrup tool call ``"domain"`` / ``"code"`` / ``"other"``.

    MCP tools arrive as ``{server}__{tool}``; ``code_exec`` is code execution;
    anything else (web, finish, delegation, ...) is ``"other"``.
    """
    if tool_name in _CODE_TOOL_NAMES:
        return "code"
    if tool_name in _WEB_TOOL_NAMES:
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
        if parts:
            return "".join(parts)
        return ""
    return _content_text(getattr(msg, "content", ""))


def _assistant_tool_calls(msg: Any) -> list[Any]:
    """Tool calls of an assistant message, in emission order."""
    blocks = getattr(msg, "blocks", None)
    if blocks:
        return [block for block in blocks if getattr(block, "kind", None) == "tool_call"]
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


def _append_turns(
    history: Iterable[Any],
    turns: list[TurnRecord],
    *,
    pending: dict[str, deque],
    agent: str,
    depth: int,
) -> None:
    """Append one agent's turns to ``turns``, splicing its sub-agents inline.

    ``pending`` maps a delegation tool name to its recorded histories in call
    order; each is consumed exactly once, by the call that produced it.  A tool
    call with no history left (the sub-agent errored before returning, or the
    recording was lost) simply contributes no nested turns rather than raising:
    a partial trajectory is more useful than none.
    """
    by_id: dict[str, ToolCall] = {}

    for msg in _flatten(history):
        role = getattr(msg, "role", None)

        if role == "assistant":
            tool_calls: list[ToolCall] = []
            nested: list[str] = []
            for tc in _assistant_tool_calls(msg):
                name = getattr(tc, "name", "") or ""
                call = ToolCall(
                    name=name,
                    input=_parse_arguments(getattr(tc, "arguments", "")),
                    id=getattr(tc, "tool_call_id", "") or "",
                )
                tool_calls.append(call)
                if call.id:
                    by_id[call.id] = call
                if name in pending:
                    nested.append(name)

            usage = getattr(msg, "token_usage", None)
            in_tok = getattr(usage, "input", 0) if usage else 0
            out_tok = getattr(usage, "output", 0) if usage else 0

            turns.append(
                TurnRecord(
                    index=len(turns),
                    text=_assistant_text(msg),
                    tool_calls=tool_calls,
                    input_tokens=int(in_tok or 0),
                    output_tokens=int(out_tok or 0),
                    duration_ms=_ms(
                        getattr(msg, "request_start_time", None),
                        getattr(msg, "request_end_time", None),
                    ),
                    agent=agent,
                    depth=depth,
                )
            )

            # Splice each delegated run immediately after the turn that made the
            # call, so reading the trajectory top to bottom follows execution.
            for name in nested:
                queue = pending.get(name)
                if not queue:
                    continue
                _append_turns(
                    queue.popleft(),
                    turns,
                    pending=pending,
                    agent=name,
                    depth=depth + 1,
                )

        elif role == "tool":
            call = by_id.get(getattr(msg, "tool_call_id", "") or "")
            if call is not None:
                call.output = _content_text(getattr(msg, "content", ""))
                call.duration_ms = _ms(
                    getattr(msg, "tool_start_time", None),
                    getattr(msg, "tool_end_time", None),
                )


def build_trajectory(
    history: Iterable[Any],
    sub_histories: Mapping[str, list] | None = None,
) -> Trajectory:
    """Convert a Stirrup message history into a :class:`Trajectory`.

    Args:
        history: The root agent's ``list[list[ChatMessage]]``.
        sub_histories: Recorded sub-agent histories keyed by delegation tool
            name, each a list of per-call histories in call order (see
            :class:`~agent.stirrup_agent.subagents.SubAgentHistoryRecorder`).
            ``None`` or empty for a flat run.
    """
    trajectory = Trajectory()
    pending = {
        name: deque(calls) for name, calls in (sub_histories or {}).items() if calls
    }
    _append_turns(history, trajectory.turns, pending=pending, agent="root", depth=0)
    return trajectory


def final_answer(history: Iterable[Any], finish_params: Any) -> str:
    """Return the user-facing answer from a completed Stirrup run.

    Prefer the explicit ``finish.answer`` supplied by the AssetOps finish tool.
    For legacy/default finish tools, ``finish.reason`` describes why the agent
    stopped and is not necessarily the answer shown to the user, so prefer
    non-empty assistant content emitted alongside the ``finish`` call while
    retaining the reason as a fallback.

    Only the root history is consulted: a sub-agent's answer reaches the user
    through the root's own finish call, never directly.
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