"""Data models for the ReActXen loop.

These sit alongside the shared :class:`~agent.models.Trajectory`, which stays
the per-round record.  :class:`ReActXenTrajectory` is the multi-round wrapper
that the evaluator and the cost analysis read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..models import Trajectory


class Outcome(str, Enum):
    """Review Agent verdict.

    Deliberately not a score.  ``ACCOMPLISHED`` stops the loop; it does not
    mark the scenario passed.  Only the benchmark's program-computed
    reference decides that.
    """

    ACCOMPLISHED = "accomplished"
    PARTIAL = "partially_accomplished"
    NOT_ACCOMPLISHED = "not_accomplished"


@dataclass
class PreconditionResult:
    """One mechanical check against the environment.

    Deterministic where it can be: these do not consult a language model.
    """

    name: str
    passed: bool
    detail: str = ""


@dataclass
class ReviewVerdict:
    """Output of the Review/Critique agent for one round."""

    outcome: Outcome
    preconditions: list[PreconditionResult] = field(default_factory=list)
    critique: str = ""
    """Free text naming what the constructed workflow got wrong."""
    alternatives: list[str] = field(default_factory=list)
    """Candidate alternative approaches, best first.  Empty when the critic
    proposes no change."""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def failed_preconditions(self) -> list[PreconditionResult]:
        return [p for p in self.preconditions if not p.passed]

    @property
    def should_continue(self) -> bool:
        return self.outcome is not Outcome.ACCOMPLISHED


@dataclass
class RoundRecord:
    """One pass of ReAct, Review and Reflect."""

    index: int
    answer: str
    trajectory: Trajectory
    review: ReviewVerdict | None = None
    reflection: str = ""
    """Guidance carried into the next round's system prompt."""
    self_ask_calls: int = 0
    """How often the model invoked the self-ask tool this round.  Self-ask is
    a tool rather than a mandatory step, so a condition can be nominally on
    and mechanically absent; this is the counter that shows it."""
    self_ask_input_tokens: int = 0
    self_ask_output_tokens: int = 0
    review_trajectory: Trajectory | None = None
    reflect_trajectory: Trajectory | None = None
    review_report: str = ""
    review_plan: dict = field(default_factory=dict)
    reflection_plan: dict = field(default_factory=dict)
    actor_completed: bool = True
    actor_round_notice: str = ""
    final_actor_attempt: bool = False
    advisory_clarified_question: str = ""
    distillation: dict = field(default_factory=dict)
    distill_input_tokens: int = 0
    distill_output_tokens: int = 0
    phase_errors: dict[str, str] = field(default_factory=dict)
    reflect_input_tokens: int = 0
    reflect_output_tokens: int = 0

    @property
    def react_steps(self) -> int:
        return len(self.trajectory.turns)


@dataclass
class ReActXenTrajectory:
    """Full multi-round record returned as ``AgentResult.trajectory``."""

    rounds: list[RoundRecord] = field(default_factory=list)
    started_at: str | None = None
    workspace_dir: str | None = None
    phase_configuration: dict = field(default_factory=dict)
    candidate_ledger: list[dict] = field(default_factory=list)
    stopped_because: str = ""
    """``"accomplished"``, ``"max_rounds"`` or ``"no_progress"``."""

    @property
    def final_round(self) -> RoundRecord | None:
        return self.rounds[-1] if self.rounds else None

    @property
    def total_input_tokens(self) -> int:
        return sum(
            r.trajectory.total_input_tokens
            + (r.review.input_tokens if r.review else 0)
            + r.reflect_input_tokens
            + r.self_ask_input_tokens
            + r.distill_input_tokens
            for r in self.rounds
        )

    @property
    def total_output_tokens(self) -> int:
        return sum(
            r.trajectory.total_output_tokens
            + (r.review.output_tokens if r.review else 0)
            + r.reflect_output_tokens
            + r.self_ask_output_tokens
            + r.distill_output_tokens
            for r in self.rounds
        )

    @property
    def total_self_ask_calls(self) -> int:
        return sum(r.self_ask_calls for r in self.rounds)

    @property
    def turns(self) -> list:
        """Flatten rounds for the shared trajectory display interface."""
        return [
            turn
            for r in self.rounds
            for trajectory in (r.trajectory, r.review_trajectory, r.reflect_trajectory)
            if trajectory is not None
            for turn in trajectory.turns
        ]

    @property
    def all_tool_calls(self) -> list:
        return [tc for turn in self.turns for tc in turn.tool_calls]

    def self_ask_rate(self) -> float:
        """Self-ask invocations per ReAct step, across all rounds.

        Report this per model.  A rate near zero means the self-ask condition
        did not run for that model, whatever the configuration said.
        """
        steps = sum(r.react_steps for r in self.rounds)
        return self.total_self_ask_calls / steps if steps else 0.0
