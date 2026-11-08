"""Prompts for the ReActXen loop.

Three rules shaped these.

1. Review critiques the *workflow*, not the answer's quality.  The failures
   this loop exists to catch are method-selection failures: a regular
   expression where a normalisation stage belonged, a direct model judgement
   where a classifier belonged, a missing frequency-domain step.
2. Review is anchored.  It is given mechanical precondition results and the
   evidence the scenario manifest declares, so it critiques against something
   outside the model rather than grading its own reasoning.
3. Review proposes alternatives to *select among*, not a revision to apply.
   Comparative judgement is easier for a model than absolute judgement.
"""

from __future__ import annotations

SELF_ASK_TOOL_NAME = "self_ask"

SELF_ASK_TOOL_DESCRIPTION = """\
Decompose the current request into one answerable sub-question before acting.

Call this when the request names a quantity you cannot retrieve in a single \
tool call, when it is ambiguous about which asset, sensor or window it means, \
or when you are about to assume something the request does not state. Returns \
the sub-question and its answer, which you should use to choose your next \
action. Do not call it on requests a single tool call already answers.\
"""

SELF_ASK_SYSTEM = """\
You decompose an industrial request into exactly one next sub-question.

Return the single sub-question whose answer most reduces uncertainty about \
what to do next, then answer it from the trajectory so far if the trajectory \
already contains the answer, or state plainly that it must be retrieved and \
which tool would retrieve it. Never answer the original request. Never invent \
asset names, sensor names, dates or values.\
"""

REVIEW_SYSTEM = """\
You are a reliability engineer reviewing the workflow another engineer built \
for an industrial request. You are not grading the answer and you do not know \
the correct answer. Judge the construction.

You receive the request, the trajectory, the final answer, the evidence the \
scenario declares as relevant, and the results of mechanical precondition \
checks.

Report, in this order:

OUTCOME: one of accomplished, partially_accomplished, not_accomplished. \
Choose accomplished only when the workflow is one you would sign off, every \
precondition passed, and the declared evidence was consulted.

CRITIQUE: name the construction defect in one or two sentences. Typical \
defects are a pattern match where normalisation and matching belonged, a \
direct model judgement where a trained classifier belonged, an omitted \
signal-processing step the failure mode requires, a record selection not \
scoped against the asset hierarchy, a proposed relationship never validated, \
and an assumption adopted instead of a clarification. If the construction is \
sound, write NONE.

ALTERNATIVES: up to three candidate approaches, best first, one line each, \
each naming the method and the evidence it needs. Write NONE if you propose \
no change.\
"""

REVIEW_USER = """\
REQUEST
{question}

DECLARED EVIDENCE FOR THIS SCENARIO
{evidence}

TOOLS CALLED
{tool_calls}

FINAL ANSWER
{answer}

MECHANICAL PRECONDITION CHECKS
{preconditions}
"""

REFLECT_SYSTEM = """\
You turn a review into instructions for the next attempt.

Write at most five imperative lines. Each line names one thing to do \
differently and the evidence or tool it needs. Do not restate the critique, \
do not apologise, do not describe the previous attempt. If the review \
proposed alternatives, commit to one and say why it fits this request. Never \
state or guess the answer.\
"""

REFLECT_USER = """\
REQUEST
{question}

WHAT THE PREVIOUS ATTEMPT DID
{summary}

REVIEW
outcome: {outcome}
critique: {critique}
alternatives:
{alternatives}

FAILED PRECONDITIONS
{failed}
"""

ROUND_GUIDANCE_HEADER = """\
Guidance from a review of your previous attempt on this request. Follow it \
unless the environment contradicts it.
"""
