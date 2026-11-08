"""Review / Critique agent.

Two halves, in this order.

**Mechanical preconditions** run first and consult no language model.  They
are the checks ReActXen's own Limitations section names as the open problem:
is the result well formed, does the asset or sensor it names exist in what the
tools returned, was the declared evidence consulted at all.  Being
deterministic, they cannot be talked out of a verdict.

**Critique** runs second, and is given the precondition results.  Unaided
self-critique is unreliable; anchored critique is the usable form.  It names
the construction defect and proposes alternatives to select among rather than
a revision to apply.

Neither half scores.  The verdict decides whether to spend another round.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from ..models import Trajectory
from .models import Outcome, PreconditionResult, ReviewVerdict
from .prompts import REVIEW_SYSTEM, REVIEW_USER

_log = logging.getLogger(__name__)

_SECTION = re.compile(r"^\s*(OUTCOME|CRITIQUE|ALTERNATIVES)\s*:\s*(.*)$", re.IGNORECASE)
_OUTCOMES = {
    "accomplished": Outcome.ACCOMPLISHED,
    "partially_accomplished": Outcome.PARTIAL,
    "partially accomplished": Outcome.PARTIAL,
    "not_accomplished": Outcome.NOT_ACCOMPLISHED,
    "not accomplished": Outcome.NOT_ACCOMPLISHED,
}


def _looks_like_json(answer: str) -> bool:
    stripped = answer.strip()
    return stripped.startswith(("{", "["))


def check_preconditions(
    *,
    question: str,
    answer: str,
    trajectory: Trajectory,
    declared_evidence: Sequence[str] = (),
    expect_json: bool = False,
) -> list[PreconditionResult]:
    """Deterministic checks over the answer and the trajectory.

    No model call, no network call.  Every check is decidable from what the
    run already produced, which is what makes them worth running before the
    critique rather than after it.
    """
    results: list[PreconditionResult] = []
    calls = trajectory.all_tool_calls

    results.append(
        PreconditionResult(
            "answer_nonempty",
            bool(answer and answer.strip()),
            "" if answer.strip() else "the run returned no answer",
        )
    )

    if expect_json or _looks_like_json(answer):
        try:
            json.loads(answer)
            results.append(PreconditionResult("answer_parses", True))
        except (ValueError, TypeError) as exc:
            results.append(
                PreconditionResult("answer_parses", False, f"invalid JSON: {exc}")
            )

    results.append(
        PreconditionResult(
            "domain_tool_called",
            bool(calls),
            "" if calls else "no tool was called; the answer is unsupported",
        )
    )

    if declared_evidence:
        blob = json.dumps(
            [{"name": c.name, "input": c.input} for c in calls], default=str
        ).lower()
        missed = [e for e in declared_evidence if e.lower() not in blob]
        results.append(
            PreconditionResult(
                "declared_evidence_consulted",
                not missed,
                "" if not missed else f"never referenced: {', '.join(missed)}",
            )
        )

    # The repetition signal from the failure analysis: failing runs repeat an
    # identical call far more often than passing ones.
    signatures = Counter(
        (c.name, json.dumps(c.input, sort_keys=True, default=str)) for c in calls
    )
    repeats = {k: v for k, v in signatures.items() if v > 2}
    results.append(
        PreconditionResult(
            "no_spinning",
            not repeats,
            ""
            if not repeats
            else f"{len(repeats)} call(s) repeated identically more than twice",
        )
    )

    return results


def _render_calls(trajectory: Trajectory, limit: int = 40) -> str:
    lines = [
        f"{c.name}({json.dumps(c.input, default=str)[:200]})"
        for c in trajectory.all_tool_calls[:limit]
    ]
    return "\n".join(lines) or "(none)"


def _render_preconditions(results: Sequence[PreconditionResult]) -> str:
    return (
        "\n".join(
            f"{'PASS' if r.passed else 'FAIL'} {r.name}"
            + (f" — {r.detail}" if r.detail else "")
            for r in results
        )
        or "(none run)"
    )


def parse_review(text: str) -> tuple[Outcome, str, list[str]]:
    """Parse the Review agent's three sections, tolerantly."""
    outcome = Outcome.PARTIAL
    sections: dict[str, list[str]] = {"CRITIQUE": [], "ALTERNATIVES": []}
    current: str | None = None

    for raw in text.splitlines():
        match = _SECTION.match(raw)
        if match:
            key, rest = match.group(1).upper(), match.group(2).strip()
            if key == "OUTCOME":
                outcome = _OUTCOMES.get(rest.lower().strip(". "), Outcome.PARTIAL)
                current = None
            else:
                current = key
                if rest:
                    sections[key].append(rest)
            continue
        if current and raw.strip():
            sections[current].append(raw.strip())

    critique = " ".join(sections["CRITIQUE"]).strip()
    if critique.upper().startswith("NONE"):
        critique = ""

    alternatives = [
        re.sub(r"^[-*\d.)\s]+", "", line).strip()
        for line in sections["ALTERNATIVES"]
        if line.strip() and not line.strip().upper().startswith("NONE")
    ]
    return outcome, critique, [a for a in alternatives if a][:3]


async def review(
    *,
    complete: Any,
    question: str,
    answer: str,
    trajectory: Trajectory,
    declared_evidence: Sequence[str] = (),
    expect_json: bool = False,
) -> ReviewVerdict:
    """Run preconditions, then the anchored critique.

    Args:
        complete: ``async (system, user) -> (text, in_tokens, out_tokens)``.
    """
    preconditions = check_preconditions(
        question=question,
        answer=answer,
        trajectory=trajectory,
        declared_evidence=declared_evidence,
        expect_json=expect_json,
    )

    user = REVIEW_USER.format(
        question=question,
        evidence="\n".join(declared_evidence) or "(none declared)",
        tool_calls=_render_calls(trajectory),
        answer=answer or "(empty)",
        preconditions=_render_preconditions(preconditions),
    )

    try:
        text, in_tok, out_tok = await complete(REVIEW_SYSTEM, user)
    except Exception:  # noqa: BLE001 - a critic failure must not end the run; preconditions still stand
        _log.warning("review call failed; falling back to preconditions", exc_info=True)
        failed = [p for p in preconditions if not p.passed]
        return ReviewVerdict(
            outcome=Outcome.NOT_ACCOMPLISHED if failed else Outcome.ACCOMPLISHED,
            preconditions=preconditions,
            critique="Review unavailable; verdict from preconditions only.",
        )

    outcome, critique, alternatives = parse_review(text)

    # A failed mechanical check overrides an optimistic model verdict.  The
    # deterministic half is the half that cannot be argued with.
    if any(not p.passed for p in preconditions) and outcome is Outcome.ACCOMPLISHED:
        outcome = Outcome.PARTIAL

    return ReviewVerdict(
        outcome=outcome,
        preconditions=preconditions,
        critique=critique,
        alternatives=alternatives,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )
