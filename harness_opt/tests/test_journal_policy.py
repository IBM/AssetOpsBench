"""The search memory and the policy that reads it.

The defect this file exists to prevent: a validator that discards failing
candidates instead of recording them. With no buggy nodes in the journal the
debug operator selects nothing, half the search budget evaporates, and the loop
degenerates to draft-and-improve without ever saying so.
"""

from __future__ import annotations

import random

import pytest

from harness_opt.journal import (
    CAPABILITY_SUBTYPES,
    Journal,
    Node,
    patch_kind,
)
from harness_opt.policy import Decision, SearchConfig, phase_at, select
from harness_opt.reflect import compare


def node(**kw) -> Node:
    kw.setdefault("spec", {})
    return Node(**kw)


def scored(journal: Journal, results: dict[str, bool], **kw) -> Node:
    n = journal.append(node(**kw))
    n.dev_results = results
    return n


# -- journal ----------------------------------------------------------------


def test_nodes_are_numbered_in_insertion_order() -> None:
    j = Journal()
    a, b = j.append(node()), j.append(node())

    assert (a.step, b.step) == (0, 1)
    assert len(j) == 2


def test_a_node_cannot_name_an_unknown_parent() -> None:
    j = Journal()

    with pytest.raises(ValueError, match="unknown parent"):
        j.append(node(parent_id="nope"))


def test_a_node_cannot_be_appended_twice() -> None:
    j = Journal()
    n = j.append(node())

    with pytest.raises(ValueError, match="already in the journal"):
        j.append(n)


def test_best_is_the_highest_dev_pass_rate() -> None:
    j = Journal()
    scored(j, {"a": True, "b": False})
    good = scored(j, {"a": True, "b": True})

    assert j.best() is good
    assert good.score == 1.0


def test_ties_break_toward_the_earlier_node() -> None:
    """Otherwise the frontier drifts between equally-scoring siblings."""
    j = Journal()
    first = scored(j, {"a": True})
    scored(j, {"a": True})

    assert j.best() is first


def test_a_buggy_node_never_becomes_the_frontier() -> None:
    j = Journal()
    good = scored(j, {"a": False, "b": False})
    bad = j.append(node(is_buggy=True))
    bad.dev_results = {"a": True, "b": True}

    assert j.best() is good
    assert bad.score == -1.0


def test_best_is_none_before_anything_is_scored() -> None:
    j = Journal()
    j.append(node(is_buggy=True))

    assert j.best() is None


def test_debug_depth_counts_consecutive_debugs() -> None:
    j = Journal()
    root = j.append(node(operator="draft"))
    d1 = j.append(node(operator="debug", parent_id=root.id))
    d2 = j.append(node(operator="debug", parent_id=d1.id))

    assert j.debug_depth(root.id) == 0
    assert j.debug_depth(d1.id) == 1
    assert j.debug_depth(d2.id) == 2


def test_lineage_runs_root_first() -> None:
    j = Journal()
    root = j.append(node())
    child = j.append(node(parent_id=root.id))

    assert [n.id for n in j.lineage(child.id)] == [root.id, child.id]


def test_summary_reports_counts_without_the_whole_journal() -> None:
    j = Journal()
    accepted = scored(j, {"a": True}, plan="strip nulls")
    accepted.outcome = compare({"a": False}, {"a": True})
    j.append(node(is_buggy=True, analysis="scope violation"))

    summary = j.summary()

    assert summary["nodes"] == 2
    assert summary["buggy"] == 1
    assert summary["best_id"] == accepted.id
    assert summary["accepted"][0]["plan"] == "strip nulls"
    assert "scope violation" in summary["rejected_reasons"][0]["why"]
    assert "spec" not in summary


def test_trend_lists_only_scored_nodes_in_order() -> None:
    j = Journal()
    a = scored(j, {"x": True})
    a.outcome = compare({"x": False}, {"x": True})
    j.append(node(is_buggy=True))
    b = scored(j, {"x": True})
    b.outcome = compare({"x": True}, {"x": True})

    trend = j.trend()

    assert [row["step"] for row in trend] == [0, 2]


def test_patch_kinds_are_known() -> None:
    assert patch_kind("argument_modification") == "capability"
    assert patch_kind("rule_addition") == "steering"
    with pytest.raises(ValueError, match="unknown patch subtype"):
        patch_kind("rm -rf")


def test_journal_saves(tmp_path) -> None:
    import json

    j = Journal()
    scored(j, {"a": True})

    body = json.loads(j.save(tmp_path / "j.json").read_text())

    assert body["summary"]["nodes"] == 1
    assert len(body["nodes"]) == 1


# -- policy -----------------------------------------------------------------


def test_drafting_continues_until_the_seed_count_is_met() -> None:
    j = Journal()
    config = SearchConfig(num_drafts=3, debug_prob=0.0)

    for _ in range(3):
        decision = select(j, config, rng=random.Random(0))
        assert decision.operator == "draft"
        assert decision.parent is None
        scored(j, {"a": True}, operator="draft")

    assert select(j, config, rng=random.Random(0)).operator == "improve"


def test_unscored_drafts_keep_the_policy_drafting() -> None:
    """With nothing to build on, drafting again is the only sound move.

    The budget ends the loop; the policy must not invent a frontier out of
    candidates that were never evaluated.
    """
    j = Journal()
    for _ in range(3):
        j.append(node(operator="draft"))

    decision = select(j, SearchConfig(num_drafts=3, debug_prob=0.0), rng=random.Random(0))

    assert decision.operator == "draft"


def test_a_buggy_leaf_is_selected_for_debugging() -> None:
    j = Journal()
    for _ in range(2):
        scored(j, {"a": True}, operator="draft")
    broken = j.append(node(operator="improve", is_buggy=True))

    decision = select(j, SearchConfig(num_drafts=2, debug_prob=1.0), rng=random.Random(0))

    assert decision.operator == "debug"
    assert decision.parent is broken


def test_a_buggy_node_with_children_is_not_debugged_again() -> None:
    j = Journal()
    for _ in range(2):
        scored(j, {"a": True}, operator="draft")
    broken = j.append(node(operator="improve", is_buggy=True))
    j.append(node(operator="debug", parent_id=broken.id))

    decision = select(j, SearchConfig(num_drafts=2, debug_prob=1.0), rng=random.Random(0))

    assert decision.operator == "improve"


def test_debugging_stops_at_the_depth_limit() -> None:
    j = Journal()
    for _ in range(2):
        scored(j, {"a": True}, operator="draft")
    chain = j.append(node(operator="improve", is_buggy=True))
    for _ in range(3):
        chain = j.append(node(operator="debug", parent_id=chain.id, is_buggy=True))

    decision = select(
        j, SearchConfig(num_drafts=2, debug_prob=1.0, max_debug_depth=2),
        rng=random.Random(0),
    )

    assert decision.operator == "improve"


def test_with_no_good_node_the_policy_drafts_again() -> None:
    j = Journal()
    for _ in range(2):
        j.append(node(operator="draft", is_buggy=True))

    decision = select(j, SearchConfig(num_drafts=2, debug_prob=0.0), rng=random.Random(0))

    assert decision.operator == "draft"


def test_otherwise_the_policy_improves_the_best_node() -> None:
    j = Journal()
    scored(j, {"a": False}, operator="draft")
    best = scored(j, {"a": True}, operator="draft")

    decision = select(j, SearchConfig(num_drafts=2, debug_prob=0.0), rng=random.Random(0))

    assert decision.operator == "improve"
    assert decision.parent is best


def test_the_capability_phase_comes_first() -> None:
    config = SearchConfig(capability_iterations=5)

    assert phase_at(0, config) == "capability"
    assert phase_at(4, config) == "capability"
    assert phase_at(5, config) == "steering"


def test_the_capability_phase_restricts_the_subtypes() -> None:
    from harness_opt.policy import allowed_subtypes

    assert allowed_subtypes("capability") == CAPABILITY_SUBTYPES
    assert "rule_addition" not in allowed_subtypes("capability")
    assert "rule_addition" in allowed_subtypes("steering")


def test_a_bad_config_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="probability"):
        SearchConfig(debug_prob=1.5)
    with pytest.raises(ValueError, match="num_drafts"):
        SearchConfig(num_drafts=0)
