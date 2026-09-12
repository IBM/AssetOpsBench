"""The paired outcome table, which is both the accept rule and the metric.

Comparing two dev *means* would have the loop accepting noise: the scorer is
deterministic but the agent is not, so the same harness can pass a task on one
run and fail it on the next. A paired comparison against the parent cancels
per-task difficulty, which a difference of means does not.

The four buckets are AutoSaddler's reflection categories and McNemar's 2x2 at
the same time:

               patch passes   patch fails
parent passes  still_passing  regressed
parent fails   fixed          still_failing
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb


@dataclass(frozen=True)
class PairedOutcome:
    """One patch, compared task by task against its parent."""

    fixed: tuple[str, ...] = ()
    regressed: tuple[str, ...] = ()
    still_passing: tuple[str, ...] = ()
    still_failing: tuple[str, ...] = ()

    @property
    def n(self) -> int:
        return (
            len(self.fixed)
            + len(self.regressed)
            + len(self.still_passing)
            + len(self.still_failing)
        )

    @property
    def discordant(self) -> int:
        """Tasks whose outcome changed. McNemar looks only at these."""
        return len(self.fixed) + len(self.regressed)

    @property
    def net(self) -> int:
        return len(self.fixed) - len(self.regressed)

    @property
    def parent_score(self) -> float:
        return _ratio(len(self.still_passing) + len(self.regressed), self.n)

    @property
    def patch_score(self) -> float:
        return _ratio(len(self.still_passing) + len(self.fixed), self.n)

    @property
    def fix_rate(self) -> float:
        """Share of tasks the parent failed that the patch now passes."""
        return _ratio(len(self.fixed), len(self.fixed) + len(self.still_failing))

    @property
    def regression_rate(self) -> float:
        """Share of tasks the parent passed that the patch now fails."""
        return _ratio(
            len(self.regressed), len(self.regressed) + len(self.still_passing)
        )

    def accepts(self) -> bool:
        """The gate: strictly more fixes than regressions.

        Deliberately weak. On a dev set of this size a significance test would
        need roughly six fixes and no regressions to clear p<0.05, which almost
        no real patch achieves, so a significance gate rejects nearly
        everything. Significance is claimed on the trend across iterations
        instead, not on any single patch.
        """
        return self.net > 0

    def p_value(self) -> float:
        """Exact two-sided McNemar p, for reporting rather than for gating."""
        n, k = self.discordant, min(len(self.fixed), len(self.regressed))
        if n == 0:
            return 1.0
        tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
        return min(1.0, 2 * tail)

    def to_json(self) -> dict:
        return {
            "fixed": list(self.fixed),
            "regressed": list(self.regressed),
            "still_passing": list(self.still_passing),
            "still_failing": list(self.still_failing),
            "n": self.n,
            "net": self.net,
            "parent_score": round(self.parent_score, 4),
            "patch_score": round(self.patch_score, 4),
            "fix_rate": round(self.fix_rate, 4),
            "regression_rate": round(self.regression_rate, 4),
            "p_value": round(self.p_value(), 4),
        }


def compare(parent: dict[str, bool], patch: dict[str, bool]) -> PairedOutcome:
    """Bucket every task both harnesses were scored on.

    Tasks missing from either side are excluded rather than assumed failed, so
    a partial evaluation narrows the comparison instead of faking a regression.
    """
    shared = sorted(set(parent) & set(patch))
    buckets: dict[str, list[str]] = {
        "fixed": [],
        "regressed": [],
        "still_passing": [],
        "still_failing": [],
    }
    for task in shared:
        before, after = bool(parent[task]), bool(patch[task])
        if before and after:
            buckets["still_passing"].append(task)
        elif before and not after:
            buckets["regressed"].append(task)
        elif not before and after:
            buckets["fixed"].append(task)
        else:
            buckets["still_failing"].append(task)
    return PairedOutcome(**{k: tuple(v) for k, v in buckets.items()})


def flip_rate(runs: list[dict[str, bool]]) -> float:
    """Per-task disagreement between identical baseline runs: the noise floor.

    Run the baseline three times and pass the results here. This is the number
    every accept decision is implicitly compared against, and it has to be
    measured before the first patch is proposed.
    """
    if len(runs) < 2:
        raise ValueError("need at least two runs to measure a flip rate")
    shared = set(runs[0])
    for run in runs[1:]:
        shared &= set(run)
    if not shared:
        return 0.0
    flipped = sum(1 for t in shared if len({bool(r[t]) for r in runs}) > 1)
    return flipped / len(shared)


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
