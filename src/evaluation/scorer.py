"""Score scenario results using the existing AssetOpsBench judge."""

from __future__ import annotations

import logging
import math
from typing import Any

_log = logging.getLogger(__name__)

# The canonical evaluation agent lives in ``reactxen`` and in
# ``src/tmp/evaluation_agent/``.  Both depend on WatsonX credentials and the
# ``reactxen`` package.  We try to import the ``graders`` module from the
# ``scenario_server`` sub-package first (it bundles all three scoring levels);
# if unavailable we fall back to simple deterministic checks.
_HAS_EVAL_AGENT = False
try:
    from scenario_server.grading.graders import (
        evaluation_agent as _llm_judge,
    )

    _HAS_EVAL_AGENT = True
except Exception:  # pragma: no cover
    _log.debug("scenario_server graders unavailable — using deterministic scoring only")


def _deterministic_score(actual: str, expected: Any) -> dict | None:
    """Try deterministic scoring; return ``None`` if not applicable."""
    if expected is None:
        return None
    expected_str = str(expected).strip()
    actual_str = str(actual).strip()

    # Numeric check
    try:
        exp_f = float(expected_str)
        act_f = float(actual_str)
        if math.isclose(act_f, exp_f, rel_tol=1e-6):
            return {"passed": True, "criteria": [], "method": "numeric"}
    except (ValueError, TypeError):
        pass

    # Exact string match (case-insensitive)
    if actual_str.lower() == expected_str.lower():
        return {"passed": True, "criteria": [], "method": "exact"}

    return None


def score_result(
    scenario: Any,
    answer: str,
    trace: str = "",
    judge_model_id: int = 16,
) -> dict:
    """Score an agent's *answer* against a *scenario*.

    Tries deterministic scoring first (numeric / exact string match).  When
    the scenario is non-deterministic **and** the LLM judge is available,
    delegates to the 6-criteria ``evaluation_agent``.

    Args:
        scenario: A :class:`Scenario` (or duck-typed object with ``text``,
                  ``expected_result``, ``characteristic_form``,
                  ``deterministic`` attributes).
        answer: The agent's final answer string.
        trace: Serialised execution trace (step results).
        judge_model_id: WatsonX model id passed to ``EvaluationAgent``.

    Returns:
        A dict with keys ``passed`` (bool), ``criteria`` (list), and
        ``method`` (str).
    """
    expected = getattr(scenario, "expected_result", None)
    is_deterministic = getattr(scenario, "deterministic", False)

    # 1. Deterministic fast path
    if is_deterministic and expected is not None:
        det = _deterministic_score(answer, expected)
        if det is not None:
            return det
        # Deterministic but no match — fail without calling judge.
        return {"passed": False, "criteria": [], "method": "deterministic_fail"}

    # 2. LLM judge
    if _HAS_EVAL_AGENT:
        characteristic = getattr(scenario, "characteristic_form", "") or ""
        query = getattr(scenario, "text", "")
        try:
            passed, criteria = _llm_judge(
                actual=answer,
                charactistic=characteristic,
                query=query,
                trace=trace,
                model_id=judge_model_id,
            )
            return {"passed": passed, "criteria": criteria, "method": "llm_judge"}
        except Exception:
            _log.exception(
                "LLM judge failed for scenario %s", getattr(scenario, "id", "?")
            )

    # 3. Fallback — cannot judge
    _log.warning(
        "No scoring method available for scenario %s (non-deterministic, no LLM judge)",
        getattr(scenario, "id", "?"),
    )
    return {"passed": False, "criteria": [], "method": "unscored"}
