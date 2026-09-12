"""Group-wise train/dev/test splits over the scenario suite.

Split by scenario *category*, not by sampling individual tasks. AutoSaddler
splits GAIA2 by persona and SWE-Bench Pro by repository for the same reason:
"the test set contains tasks from groups unseen during optimization, providing
a stronger measure of out-of-distribution generalization than random task-level
splits."

Choose the categories by baseline headroom, not by size. A category the
baseline already passes has nothing to teach the search.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class SplitError(ValueError):
    """The split is unusable. Raised at load time."""


@dataclass(frozen=True)
class Split:
    """Which scenario ids sit in which arm, pinned to a file and committed."""

    train: tuple[str, ...]
    dev: tuple[str, ...]
    reserve: tuple[str, ...] = ()
    #: category -> arm, kept so the grouping is auditable after the fact
    categories: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arms = {"train": self.train, "dev": self.dev, "reserve": self.reserve}
        seen: dict[str, str] = {}
        for arm, ids in arms.items():
            if not ids and arm != "reserve":
                raise SplitError(f"{arm} split is empty")
            for scenario_id in ids:
                if scenario_id in seen:
                    raise SplitError(
                        f"scenario {scenario_id} is in both {seen[scenario_id]} "
                        f"and {arm}; the arms must be disjoint"
                    )
                seen[scenario_id] = arm

    @property
    def all_ids(self) -> tuple[str, ...]:
        return self.train + self.dev + self.reserve

    def arm_of(self, scenario_id: str) -> str | None:
        for arm in ("train", "dev", "reserve"):
            if scenario_id in getattr(self, arm):
                return arm
        return None

    @classmethod
    def from_categories(
        cls,
        profile: dict[str, list],
        train: list[str],
        dev: list[str],
        reserve: list[str] | None = None,
    ) -> "Split":
        """Build a split from a suite profile keyed by category.

        ``profile`` is ``all.yaml`` as already loaded: category -> scenario ids.
        """
        reserve = reserve or []
        assigned = train + dev + reserve
        duplicated = {c for c in assigned if assigned.count(c) > 1}
        if duplicated:
            raise SplitError(f"category {sorted(duplicated)} assigned twice")
        unknown = sorted(set(assigned) - set(profile))
        if unknown:
            raise SplitError(
                f"unknown categories {unknown}; profile has {sorted(profile)}"
            )

        def ids_for(categories: list[str]) -> tuple[str, ...]:
            out: list[str] = []
            for category in categories:
                out.extend(str(i) for i in (profile[category] or []))
            return tuple(out)

        for arm, categories in (("train", train), ("dev", dev)):
            if not categories:
                raise SplitError(f"{arm} needs at least one category")
            if not ids_for(categories):
                raise SplitError(
                    f"{arm} categories {categories} contain no scenarios"
                )

        return cls(
            train=ids_for(train),
            dev=ids_for(dev),
            reserve=ids_for(reserve),
            categories={
                **{c: "train" for c in train},
                **{c: "dev" for c in dev},
                **{c: "reserve" for c in reserve},
            },
        )

    def to_json(self) -> dict:
        return {
            "train": list(self.train),
            "dev": list(self.dev),
            "reserve": list(self.reserve),
            "categories": dict(self.categories),
        }

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path | str) -> "Split":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            train=tuple(raw["train"]),
            dev=tuple(raw["dev"]),
            reserve=tuple(raw.get("reserve", ())),
            categories=dict(raw.get("categories", {})),
        )


def minibatches(ids: tuple[str, ...], size: int, seed: int = 0) -> list[list[str]]:
    """Deterministic mini-batches covering the training set exactly once.

    Deterministic on purpose: a fixed batch order lets per-task results be
    cached across iterations, which is where roughly a third of the evaluation
    budget is saved.
    """
    if size <= 0:
        raise SplitError(f"mini-batch size must be positive, got {size}")
    import random

    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    return [shuffled[i : i + size] for i in range(0, len(shuffled), size)]
