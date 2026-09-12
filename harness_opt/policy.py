"""Which node to work on next, and with which operator.

AIDE's ``search_policy`` (MIT, arXiv 2502.13138) with its defaults intact:
seed a few independent drafts, spend half the remaining steps repairing broken
candidates, and otherwise improve the best one. The single change is what
"best" means: AIDE reads a validation metric on the task it is solving, this
reads a pass rate on a held-out *category*, because a harness has to generalize
and a solution script does not.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .journal import Journal, Node


@dataclass(frozen=True)
class SearchConfig:
    """AIDE's tested defaults, plus the capability/steering schedule."""

    num_drafts: int = 5
    debug_prob: float = 0.5
    max_debug_depth: int = 3
    #: iterations spent on capability patches before switching to steering.
    #: AutoSaddler: k = E * ceil(|train| / B), with E = 1 working well.
    capability_iterations: int = 0

    def __post_init__(self) -> None:
        if self.num_drafts < 1:
            raise ValueError("num_drafts must be at least 1")
        if not 0.0 <= self.debug_prob <= 1.0:
            raise ValueError("debug_prob must be a probability")
        if self.max_debug_depth < 0:
            raise ValueError("max_debug_depth must not be negative")


@dataclass(frozen=True)
class Decision:
    """What to do this iteration."""

    operator: str
    parent: Node | None
    #: "capability" while the schedule is in its exploration phase
    phase: str

    @property
    def drafts_from_baseline(self) -> bool:
        return self.operator == "draft"


def phase_at(iteration: int, config: SearchConfig) -> str:
    """Capability patches first, steering after.

    AutoSaddler removes only this schedule and loses 5.9 points on GAIA2, more
    than it loses by removing deep diagnosis. Capability patches change what the
    agent can do; steering patches change what it chooses. Structure first.
    """
    return "capability" if iteration < config.capability_iterations else "steering"


def select(
    journal: Journal,
    config: SearchConfig,
    iteration: int = 0,
    rng: random.Random | None = None,
) -> Decision:
    """Pick the next node and operator.

    Order matters and mirrors AIDE:

    1. seed ``num_drafts`` independent candidates off the baseline
    2. with probability ``debug_prob``, repair a buggy leaf within depth
    3. with no good node to build on, draft again
    4. otherwise improve the best node
    """
    rng = rng or random.Random()
    phase = phase_at(iteration, config)

    if len(journal.drafts) < config.num_drafts:
        return Decision(operator="draft", parent=None, phase=phase)

    if rng.random() < config.debug_prob:
        debuggable = [
            n
            for n in journal.buggy
            if journal.is_leaf(n.id)
            and journal.debug_depth(n.id) <= config.max_debug_depth
        ]
        if debuggable:
            return Decision(
                operator="debug", parent=rng.choice(debuggable), phase=phase
            )

    best = journal.best()
    if best is None:
        return Decision(operator="draft", parent=None, phase=phase)
    return Decision(operator="improve", parent=best, phase=phase)


def allowed_subtypes(phase: str) -> frozenset[str]:
    from .journal import CAPABILITY_SUBTYPES, PATCH_SUBTYPES, STEERING_SUBTYPES

    if phase == "capability":
        return CAPABILITY_SUBTYPES
    if phase == "steering":
        return PATCH_SUBTYPES  # steering phase may still repair capabilities
    raise ValueError(f"unknown phase {phase!r}")
