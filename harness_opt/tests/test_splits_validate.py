"""Group-wise splits, and the gates a candidate clears before it costs a run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness_opt.splits import Split, SplitError, minibatches
from harness_opt.validate import (
    EntityCorpus,
    check_no_entity_values,
    check_no_leak,
    check_scope,
    check_spec,
    validate,
)

PROFILE = {
    "car": [1, 2, 3],
    "fcc": [10, 11],
    "fmsr": [20, 21],
    "wosr": [30, 31, 32],
    "health": [40],
}


# -- splits -----------------------------------------------------------------


def test_a_split_groups_whole_categories() -> None:
    split = Split.from_categories(PROFILE, train=["wosr", "car"], dev=["fmsr"])

    assert split.train == ("30", "31", "32", "1", "2", "3")
    assert split.dev == ("20", "21")
    assert split.arm_of("30") == "train"
    assert split.arm_of("20") == "dev"


def test_ids_are_strings_so_they_match_the_report_keys() -> None:
    split = Split.from_categories(PROFILE, train=["car"], dev=["fmsr"])

    assert all(isinstance(i, str) for i in split.all_ids)


def test_a_category_cannot_sit_in_two_arms() -> None:
    with pytest.raises(SplitError, match="assigned twice"):
        Split.from_categories(PROFILE, train=["car"], dev=["car"])


def test_an_unknown_category_is_rejected() -> None:
    with pytest.raises(SplitError, match="unknown categories"):
        Split.from_categories(PROFILE, train=["nope"], dev=["fmsr"])


def test_an_empty_dev_arm_is_rejected() -> None:
    """The dev gate is the whole design; a split without one is unusable."""
    with pytest.raises(SplitError, match="needs at least one category"):
        Split.from_categories(PROFILE, train=["car"], dev=[])


def test_overlapping_ids_are_caught_even_if_categories_differ() -> None:
    with pytest.raises(SplitError, match="disjoint"):
        Split(train=("1",), dev=("1",))


def test_reserve_may_be_empty() -> None:
    split = Split.from_categories(PROFILE, train=["car"], dev=["fmsr"])

    assert split.reserve == ()


def test_a_split_round_trips(tmp_path: Path) -> None:
    split = Split.from_categories(
        PROFILE, train=["wosr"], dev=["fmsr"], reserve=["health"]
    )

    reloaded = Split.load(split.save(tmp_path / "split.json"))

    assert reloaded.train == split.train
    assert reloaded.categories["health"] == "reserve"


def test_minibatches_cover_the_training_set_exactly_once() -> None:
    ids = tuple(str(i) for i in range(10))

    batches = minibatches(ids, size=3, seed=7)

    flat = [i for b in batches for i in b]
    assert sorted(flat) == sorted(ids)
    assert len(flat) == len(set(flat))
    assert [len(b) for b in batches] == [3, 3, 3, 1]


def test_minibatches_are_deterministic_so_results_can_be_cached() -> None:
    ids = tuple(str(i) for i in range(10))

    assert minibatches(ids, 3, seed=1) == minibatches(ids, 3, seed=1)
    assert minibatches(ids, 3, seed=1) != minibatches(ids, 3, seed=2)


def test_a_nonpositive_batch_size_is_rejected() -> None:
    with pytest.raises(SplitError, match="must be positive"):
        minibatches(("1",), 0)


# -- validator --------------------------------------------------------------

EDITABLE = ["src/agent/stirrup_agent/adapter/", "src/agent/_prompts.py"]


def test_scope_allows_the_editable_surface() -> None:
    assert check_scope(["src/agent/stirrup_agent/adapter/adapter.json"], EDITABLE)


def test_scope_rejects_a_server_edit() -> None:
    """The servers are the environment. Editing them makes the task easier."""
    verdict = check_scope(["src/servers/wo/server.py"], EDITABLE)

    assert not verdict
    assert verdict.gate == "scope"
    assert "src/servers/wo/server.py" in verdict.reason


def test_scope_rejects_an_evaluation_edit() -> None:
    assert not check_scope(["src/evaluation/scorers/static_json.py"], EDITABLE)


def test_a_malformed_spec_fails_the_spec_gate() -> None:
    verdict = check_spec({"tools": {"t": {"exec": "whatever"}}})

    assert not verdict
    assert verdict.gate == "spec"


def test_a_wellformed_spec_passes() -> None:
    assert check_spec({"tools": {"t": {"strip_empty": True}}})


def test_no_leak_catches_a_shared_eight_word_run() -> None:
    answer = "the chiller condenser water flow dropped to zero on the fourth day"
    spec_text = "note: the chiller condenser water flow dropped to zero on the fourth"

    verdict = check_no_leak(spec_text, [answer])

    assert not verdict
    assert verdict.gate == "no_leak"


def test_no_leak_allows_ordinary_text() -> None:
    assert check_no_leak("strip empty arguments before the call", ["2876"])


def test_short_text_cannot_trip_the_leak_gate() -> None:
    assert check_no_leak("too short", ["too short"])


# -- the entity gate, and the distinction that makes it usable --------------


def corpus() -> EntityCorpus:
    return EntityCorpus(
        values=frozenset({"MAIN", "CHILLER6", "1000045", "BEARING-WEAR",
                          "Chiller 6 Condenser Water Flow"})
    )


def test_a_schema_identifier_is_allowed() -> None:
    """An adapter narrowing a work-order schema has to name the field."""
    verdict = check_no_entity_values(
        json.dumps({"tools": {"wo__q": {"require": ["siteid", "status"]}}}), corpus()
    )

    assert verdict


def test_a_data_value_is_banned() -> None:
    verdict = check_no_entity_values(
        json.dumps({"tools": {"wo__q": {"defaults": {"siteid": "MAIN"}}}}), corpus()
    )

    assert not verdict
    assert verdict.gate == "no_entity_values"
    assert "MAIN" in verdict.reason


def test_an_asset_name_in_a_description_is_banned() -> None:
    verdict = check_no_entity_values(
        json.dumps({"tools": {"t": {"description": "For CHILLER6 use flow."}}}),
        corpus(),
    )

    assert not verdict


def test_the_corpus_is_harvested_from_shared_files(tmp_path: Path) -> None:
    (tmp_path / "wo").mkdir()
    (tmp_path / "wo" / "workorders.csv").write_text(
        "wonum,siteid,status\n1000045,MAIN,WAPPR\n"
    )
    (tmp_path / "assets.json").write_text(json.dumps([{"assetnum": "Chiller 6"}]))

    harvested = EntityCorpus.from_shared(tmp_path)

    assert "1000045" in harvested.values
    assert "Chiller 6" in harvested.values
    assert "siteid" not in harvested.values


def test_harvesting_skips_values_too_short_to_attribute(tmp_path: Path) -> None:
    (tmp_path / "x.csv").write_text("a,b\nOK,MAIN\n")

    harvested = EntityCorpus.from_shared(tmp_path)

    assert "OK" not in harvested.values
    assert "MAIN" in harvested.values


def test_harvesting_skips_long_prose(tmp_path: Path) -> None:
    """A sentence is not an identifier; banning it would reject real patches."""
    (tmp_path / "x.csv").write_text(
        "d\n\"during a site level maintenance review the engineers separated work\"\n"
    )

    harvested = EntityCorpus.from_shared(tmp_path)

    assert not any(len(v.split()) > 6 for v in harvested.values)


# -- the gates in order -----------------------------------------------------


def test_validate_reports_the_first_failing_gate() -> None:
    verdict = validate(
        spec_raw={"tools": {"t": {"bogus": 1}}},
        changed_paths=["src/servers/wo/server.py"],
        editable=EDITABLE,
    )

    assert verdict.gate == "scope"


def test_validate_passes_a_clean_patch() -> None:
    assert validate(
        spec_raw={"tools": {"wo__q": {"strip_empty": True}}},
        changed_paths=["src/agent/stirrup_agent/adapter/adapter.json"],
        editable=EDITABLE,
        corpus=corpus(),
        answers=["1000045"],
    )


def test_validate_runs_the_entity_gate_last_but_still_runs_it() -> None:
    verdict = validate(
        spec_raw={"tools": {"wo__q": {"defaults": {"siteid": "MAIN"}}}},
        changed_paths=["src/agent/stirrup_agent/adapter/adapter.json"],
        editable=EDITABLE,
        corpus=corpus(),
    )

    assert verdict.gate == "no_entity_values"
