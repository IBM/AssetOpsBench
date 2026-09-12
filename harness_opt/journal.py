"""The search memory: nodes, their parents, and what each one scored.

Structure follows AIDE's ``Journal``/``Node`` (MIT, arXiv 2502.13138) with two
deliberate changes.

A node stores a **complete adapter snapshot**, not a diff against its parent.
The adapter is a handful of small files, so copying it costs nothing and
removes an entire class of failure: patches that no longer apply once the
frontier moves, three-way merge conflicts, ordering bugs. Diffs are computed
for display and for the proposer's context.

A node's metric is a **paired dev outcome**, not a scalar on the training
batch. That substitution is the whole adaptation from AIDE to a harness: AIDE
may overfit its one task, a harness may not.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .reflect import PairedOutcome

#: AutoSaddler's Table 1, restricted to the subtypes a frozen server allows.
CAPABILITY_SUBTYPES = frozenset(
    {"argument_modification", "new_tool_addition", "agent_loop_change"}
)
STEERING_SUBTYPES = frozenset(
    {"rule_addition", "rule_modification", "tool_description_fix", "pre_tool_hook"}
)
PATCH_SUBTYPES = CAPABILITY_SUBTYPES | STEERING_SUBTYPES


def patch_kind(subtype: str) -> str:
    if subtype in CAPABILITY_SUBTYPES:
        return "capability"
    if subtype in STEERING_SUBTYPES:
        return "steering"
    raise ValueError(f"unknown patch subtype {subtype!r}; expected one of "
                     f"{sorted(PATCH_SUBTYPES)}")


@dataclass
class Node:
    """One explored harness."""

    #: the complete adapter spec this node represents
    spec: dict
    #: why the proposer made it, in its own words
    plan: str = ""
    subtype: str = "rule_addition"
    parent_id: str | None = None
    operator: str = "draft"

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    step: int = 0
    ctime: float = field(default_factory=time.time)

    #: set when the patch failed a gate. A buggy node is the *input* to the
    #: debug operator, never a discard: dropping it would leave the policy with
    #: nothing to debug and silently halve the search.
    is_buggy: bool = False
    analysis: str = ""

    #: the paired dev comparison against the parent, once it reaches that gate
    outcome: PairedOutcome | None = None
    #: per-task pass/fail on dev, kept so a child can be paired against it
    dev_results: dict[str, bool] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Dev pass rate. Buggy nodes sort last."""
        if self.is_buggy or not self.dev_results:
            return -1.0
        return sum(1 for v in self.dev_results.values() if v) / len(self.dev_results)

    @property
    def kind(self) -> str:
        return patch_kind(self.subtype)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "step": self.step,
            "parent_id": self.parent_id,
            "operator": self.operator,
            "subtype": self.subtype,
            "kind": self.kind if self.subtype in PATCH_SUBTYPES else "unknown",
            "plan": self.plan,
            "is_buggy": self.is_buggy,
            "analysis": self.analysis,
            "score": self.score,
            "outcome": self.outcome.to_json() if self.outcome else None,
            "spec": self.spec,
        }


class Journal:
    """Every node explored, with parent links. The optimizer's memory."""

    def __init__(self) -> None:
        self._nodes: list[Node] = []
        self._by_id: dict[str, Node] = {}

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self):
        return iter(self._nodes)

    def append(self, node: Node) -> Node:
        if node.id in self._by_id:
            raise ValueError(f"node {node.id} is already in the journal")
        if node.parent_id is not None and node.parent_id not in self._by_id:
            raise ValueError(f"node {node.id} names unknown parent {node.parent_id}")
        node.step = len(self._nodes)
        self._nodes.append(node)
        self._by_id[node.id] = node
        return node

    def get(self, node_id: str) -> Node:
        return self._by_id[node_id]

    def children(self, node_id: str) -> list[Node]:
        return [n for n in self._nodes if n.parent_id == node_id]

    def is_leaf(self, node_id: str) -> bool:
        return not self.children(node_id)

    def debug_depth(self, node_id: str) -> int:
        """Consecutive debug operators ending at this node."""
        depth, node = 0, self._by_id[node_id]
        while node.operator == "debug" and node.parent_id is not None:
            depth += 1
            node = self._by_id[node.parent_id]
        return depth

    @property
    def drafts(self) -> list[Node]:
        return [n for n in self._nodes if n.operator == "draft"]

    @property
    def buggy(self) -> list[Node]:
        return [n for n in self._nodes if n.is_buggy]

    @property
    def good(self) -> list[Node]:
        return [n for n in self._nodes if not n.is_buggy and n.dev_results]

    def best(self) -> Node | None:
        """Highest dev pass rate. Ties break toward the earlier node, which
        keeps the frontier from drifting between equally-scoring siblings."""
        candidates = self.good
        if not candidates:
            return None
        return max(candidates, key=lambda n: (n.score, -n.step))

    def lineage(self, node_id: str) -> list[Node]:
        chain, node = [], self._by_id[node_id]
        while True:
            chain.append(node)
            if node.parent_id is None:
                break
            node = self._by_id[node.parent_id]
        return list(reversed(chain))

    # -- the summarized view, never the whole thing ------------------------
    def summary(self) -> dict:
        """What the proposer is given, instead of the full journal.

        AIDE has a summarization operator, AutoSaddler exposes a query CLI, and
        StarHarness has the proposer read the ledger rather than receive it.
        Three groups, one rule: never serialize the memory into the prompt.
        """
        best = self.best()
        return {
            "nodes": len(self._nodes),
            "buggy": len(self.buggy),
            "good": len(self.good),
            "best_id": best.id if best else None,
            "best_score": best.score if best else None,
            "accepted": [
                {
                    "id": n.id,
                    "subtype": n.subtype,
                    "plan": n.plan[:200],
                    "net": n.outcome.net if n.outcome else 0,
                }
                for n in self.good
                if n.outcome and n.outcome.accepts()
            ],
            "rejected_reasons": [
                {"id": n.id, "subtype": n.subtype, "why": n.analysis[:160]}
                for n in self._nodes
                if n.is_buggy
            ],
        }

    def trend(self) -> list[dict]:
        """Fix and regression rate per scored node, in order.

        This is the plot that says whether the loop is working. A falling
        regression rate with a comparable fix rate is the signal; a rising one
        means the dev gate is not binding.
        """
        return [
            {
                "step": n.step,
                "id": n.id,
                "fix_rate": round(n.outcome.fix_rate, 4),
                "regression_rate": round(n.outcome.regression_rate, 4),
                "net": n.outcome.net,
            }
            for n in self._nodes
            if n.outcome is not None
        ]

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "summary": self.summary(),
                    "trend": self.trend(),
                    "nodes": [n.to_json() for n in self._nodes],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path
