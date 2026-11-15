"""Reflect agent.

Turns a review into guidance for the next round.  Kept separate from Review
because the two jobs pull in opposite directions: Review looks backwards and
judges, Reflect looks forwards and commits.  Collapsing them produces an
agent that rehearses what went wrong instead of saying what to do.

Reflect never sees a reference answer and is instructed never to guess one.
Its output is prepended to the next round's system prompt.
"""

from __future__ import annotations

import logging
from typing import Any

from ..models import Trajectory
from .models import ReviewVerdict
from .prompts import REFLECT_SYSTEM, REFLECT_USER, ROUND_GUIDANCE_HEADER

_log = logging.getLogger(__name__)


def summarise(trajectory: Trajectory, answer: str, limit: int = 20) -> str:
    """One line per tool call plus the answer, for the Reflect prompt."""
    names = [c.name for c in trajectory.all_tool_calls[:limit]]
    joined = " -> ".join(names) if names else "(no tool calls)"
    return f"{joined}\nanswer: {answer[:400] or '(empty)'}"


async def reflect(
    *,
    complete: Any,
    question: str,
    answer: str,
    trajectory: Trajectory,
    verdict: ReviewVerdict,
) -> tuple[str, int, int]:
    """Produce next-round guidance.

    Args:
        complete: ``async (system, user) -> (text, in_tokens, out_tokens)``.

    Returns:
        ``(guidance, input_tokens, output_tokens)``.  ``guidance`` is empty
        when reflection fails, in which case the next round runs unchanged
        and the record shows why.
    """
    failed = (
        "\n".join(
            f"{p.name}: {p.detail or 'failed'}" for p in verdict.failed_preconditions
        )
        or "(none)"
    )
    alternatives = "\n".join(f"- {a}" for a in verdict.alternatives) or "(none)"

    user = REFLECT_USER.format(
        question=question,
        summary=summarise(trajectory, answer),
        outcome=verdict.outcome.value,
        critique=verdict.critique or "(none)",
        alternatives=alternatives,
        failed=failed,
    )

    try:
        text, in_tok, out_tok = await complete(REFLECT_SYSTEM, user)
    except Exception:  # noqa: BLE001 - a reflection failure must not end the run
        _log.warning("reflect call failed; next round runs unchanged", exc_info=True)
        return "", 0, 0

    return text.strip(), in_tok, out_tok


def as_prompt_block(guidance: str) -> str:
    """Wrap guidance for injection into the next round's system prompt."""
    if not guidance.strip():
        return ""
    return f"\n\n{ROUND_GUIDANCE_HEADER}\n{guidance.strip()}\n"
