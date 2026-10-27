import json

import pytest

from evaluation.models import Scenario
from evaluation.scorers.fmea import fmea_score


def _scenario(gold, *, tag="asset_modes", weights=None):
    dimensions = weights or [
        {"name": "coverage", "weight": 0.35},
        {"name": "precision", "weight": 0.35},
        {"name": "instruction_following", "weight": 0.2},
        {"name": "grounding", "weight": 0.1},
    ]
    return Scenario(
        id="9001",
        text="",
        expected_answer=json.dumps(gold),
        evaluation_metadata={
            "scenario_meta": {"tag": tag},
            "rubric": {"dimensions": dimensions},
        },
    )


def test_asset_modes_awards_set_precision_and_recall():
    scenario = _scenario({"asset": "Electric Motor", "failure_modes": {"A": "", "B": ""}})
    answer = json.dumps({"asset": "electric motor", "failure_modes": {"A": "", "C": ""}})

    result = fmea_score(scenario, answer, "")

    assert result.passed is False
    assert result.details["dimensions"]["coverage"] == 0.5
    assert result.details["dimensions"]["precision"] == 0.5
    assert result.details["asset_match"] is True
    assert result.details["missing_modes"] == ["B"]
    assert result.details["extra_modes"] == ["C"]
    assert result.score == 0.5


def test_asset_modes_perfect_answer_passes_case_insensitive_asset():
    gold = {"asset": "Electric Motor", "failure_modes": {"A": ""}}
    result = fmea_score(
        _scenario(gold), '{"asset":"electric motor","failure_modes":{"A":""}}', ""
    )
    assert result.passed is True
    assert result.score == 1.0


def test_markdown_fence_gets_content_credit_but_fails_contract():
    gold = {"asset": "Electric Motor", "failure_modes": {"A": ""}}
    result = fmea_score(_scenario(gold), f"```json\n{json.dumps(gold)}\n```", "")
    assert result.passed is True
    assert result.details["semantic_exact"] is True
    assert result.details["strict_success"] is False
    assert result.details["json_only"] is False
    assert result.details["dimensions"]["instruction_following"] == 0.0


def test_analytics_scores_atomic_facts_and_ordered_arrays():
    scenario = _scenario(
        {
            "ranking": ["Seal", "Bearing", "Motor"],
            "counts": {"Seal": 16, "Bearing": 13, "Motor": 7},
        },
        tag="ranking_top3",
    )
    answer = json.dumps(
        {
            "ranking": ["Bearing", "Seal", "Motor"],
            "counts": {"Seal": 16, "Bearing": 13, "Motor": 7},
        }
    )
    result = fmea_score(scenario, answer, "")
    assert result.passed is False
    assert result.details["dimensions"]["coverage"] == pytest.approx(4 / 6)
    assert result.details["wrong_value_keys"] == ["answer.ranking[0]", "answer.ranking[1]"]


def test_duplicate_nested_mode_key_fails_json_contract():
    scenario = _scenario({"asset": "Volute", "failure_modes": {"Corrosion": ""}})
    answer = '{"asset":"Volute","failure_modes":{"Corrosion":"","Corrosion":""}}'
    result = fmea_score(scenario, answer, "")
    assert result.passed is True
    assert result.details["semantic_exact"] is True
    assert result.details["strict_success"] is False
    assert result.details["duplicate_keys"] == ["Corrosion"]


def test_trajectory_reports_catalog_and_out_of_scope_tool_use():
    scenario = _scenario({"asset": "Electric Motor", "failure_modes": {"A": ""}})
    trajectory = json.dumps(
        {"turns": [{"tool_calls": [
            {"name": "utilities__get_asset_catalog"},
            {"name": "utilities__get_failure_mode_catalog"},
            {"name": "fmsr__get_failure_modes"},
        ]}]}
    )
    result = fmea_score(scenario, scenario.expected_answer, trajectory)
    assert result.details["missing_catalog_tools"] == []
    assert result.details["out_of_scope_domain_tools"] == ["fmsr__get_failure_modes"]
    assert result.details["catalog_process_compliant"] is False


def test_pass_requires_at_least_point_eight_f1():
    scenario = _scenario(
        {"asset": "Electric Motor", "failure_modes": {str(i): "" for i in range(7)}}
    )
    below = {
        "asset": "Electric Motor",
        "failure_modes": {
            **{str(i): "" for i in range(4)},
            **{f"extra-{i}": "" for i in range(3)},
        },
    }
    result = fmea_score(scenario, json.dumps(below), "")
    assert result.score == pytest.approx(4 / 7, abs=1e-6)
    assert result.passed is False
    assert result.details["f1_pass_threshold"] == 0.8


def test_f1_exactly_at_point_eight_passes_despite_float_error():
    scenario = _scenario(
        {"asset": "Electric Motor", "failure_modes": {str(i): "" for i in range(7)}}
    )
    at_threshold = {
        "asset": "Electric Motor",
        "failure_modes": {
            **{str(i): "" for i in range(6)},
            **{f"extra-{i}": "" for i in range(2)},
        },
    }
    result = fmea_score(scenario, json.dumps(at_threshold), "")
    assert result.details["semantic_f1"] < 0.8
    assert result.score == 0.8
    assert result.passed is True


def test_f1_above_point_eight_passes_without_requiring_exact_match():
    scenario = _scenario(
        {"asset": "Electric Motor", "failure_modes": {str(i): "" for i in range(8)}}
    )
    partial = {
        "asset": "Electric Motor",
        "failure_modes": {str(i): "" for i in range(6)},
    }
    result = fmea_score(scenario, json.dumps(partial), "")
    assert result.score == pytest.approx(6 / 7, abs=1e-6)
    assert result.passed is True
    assert result.details["strict_success"] is False


def _six_of_eight(threshold):
    """An answer with F1 6/7 (0.857) against a scenario carrying ``threshold``."""
    scenario = _scenario(
        {"asset": "Electric Motor", "failure_modes": {str(i): "" for i in range(8)}}
    )
    if threshold != "unset":
        scenario.evaluation_metadata["scenario_meta"]["f1_pass_threshold"] = threshold
    answer = {"asset": "Electric Motor", "failure_modes": {str(i): "" for i in range(6)}}
    return fmea_score(scenario, json.dumps(answer), "")


def test_scenario_threshold_overrides_the_default():
    result = _six_of_eight(0.9)
    assert result.details["f1_pass_threshold"] == 0.9
    assert result.passed is False


@pytest.mark.parametrize("threshold", ["unset", None])
def test_unset_threshold_uses_the_default(threshold):
    result = _six_of_eight(threshold)
    assert result.details["f1_pass_threshold"] == 0.8
    assert result.passed is True


@pytest.mark.parametrize("threshold", [True, -0.1, 1.5, 80, "0.8", float("nan")])
def test_invalid_threshold_fails_the_scenario_with_a_rationale(threshold):
    result = _six_of_eight(threshold)
    assert result.passed is False
    assert result.score == 0.0
    assert result.details["error"] == "invalid_scenario_meta"
    assert "f1_pass_threshold" in result.rationale


# --- asset_locations family (failure-location generation, SSEE matching) ---

from evaluation.scorers import fmea as fmea_mod

_SYNONYMS = {
    "non-drive end bearing": "nde",
    "nde bearing": "nde",
    "non drive end bearing": "nde",
    "drive end bearing": "de",
    "de bearing": "de",
}


def _fake_similarity(left, right, encoder):
    def key(text):
        norm = text.strip().casefold()
        return _SYNONYMS.get(norm, norm)

    return [[1.0 if key(a) == key(b) else 0.1 for b in right] for a in left]


@pytest.fixture
def fake_encoder(monkeypatch):
    monkeypatch.setattr(fmea_mod, "_similarity", _fake_similarity)


def _loc_scenario(locations, **meta):
    gold = {"asset": "centrifugal pump", "failure_locations": {name: "" for name in locations}}
    return Scenario(
        id="9101",
        text="",
        expected_answer=json.dumps(gold),
        evaluation_metadata={"scenario_meta": {"tag": "asset_locations", **meta}},
    )


def _loc_answer(names, asset="Centrifugal Pump"):
    return json.dumps({"asset": asset, "failure_locations": {n: "x" for n in names}})


GOLD = ["Mechanical Seal", "Drive End Bearing", "Non-Drive End Bearing", "Impeller"]


def test_locations_route_by_gold_shape_without_tag(fake_encoder):
    gold = {"asset": "pump", "failure_locations": {"Impeller": ""}}
    scenario = Scenario(id="x", text="", expected_answer=json.dumps(gold))
    result = fmea_score(scenario, json.dumps(gold), "")
    assert result.details["family"] == "asset_locations"
    assert result.score == 1.0


def test_locations_paraphrase_and_case_match(fake_encoder):
    answer = _loc_answer(["mechanical seal", "DE Bearing", "NDE Bearing", "Impeller"])
    result = fmea_score(_loc_scenario(GOLD), answer, "")
    assert result.score == 1.0
    assert result.details["missing_locations"] == []
    assert result.details["asset_match"] is True


def test_locations_one_to_one_blocks_paraphrase_padding(fake_encoder):
    # Three names for the same gold item earn one match, not three.
    answer = _loc_answer(["NDE Bearing", "Non Drive End Bearing", "Non-Drive End Bearing"])
    result = fmea_score(_loc_scenario(GOLD), answer, "")
    dims = result.details["dimensions"]
    assert dims["coverage"] == 0.25
    assert dims["precision"] == pytest.approx(1 / 3)
    assert len(result.details["extra_locations"]) == 2


def test_locations_cap_scores_first_n_only(fake_encoder):
    names = ["Bolt", "Nut", "Mechanical Seal", "Impeller"]
    result = fmea_score(_loc_scenario(GOLD, max_locations=2), _loc_answer(names), "")
    assert result.details["over_cap"] is True
    assert result.details["returned_count"] == 4
    assert result.details["scored_locations"] == ["Bolt", "Nut"]
    assert result.score == 0.0


def test_locations_novel_metrics_exclude_seeds(fake_encoder):
    scenario = _loc_scenario(GOLD, seed_locations=["Mechanical Seal", "Drive End Bearing"])
    answer = _loc_answer(["Mechanical Seal", "Drive End Bearing", "Impeller", "Coupling"])
    result = fmea_score(scenario, answer, "")
    d = result.details
    assert d["dimensions"]["coverage"] == 0.75
    assert d["novel_recall"] == 0.5        # Impeller of {NDE Bearing, Impeller}
    assert d["novel_precision"] == 0.5     # Impeller of {Impeller, Coupling}
    assert d["seed_locations"] == ["Drive End Bearing", "Mechanical Seal"]


def test_locations_threshold_from_meta(monkeypatch):
    monkeypatch.setattr(fmea_mod, "_similarity", lambda g, m, e: [[0.75 for _ in m] for _ in g])
    answer = _loc_answer(["Something"])
    assert fmea_score(_loc_scenario(["Impeller"]), answer, "").score == 0.0
    assert fmea_score(_loc_scenario(["Impeller"], match_threshold=0.7), answer, "").score == 1.0


def test_locations_invalid_meta_fails_cleanly(fake_encoder):
    result = fmea_score(_loc_scenario(GOLD, max_locations=0), _loc_answer(GOLD), "")
    assert result.details["error"] == "invalid_scenario_meta"


def test_locations_encoder_unavailable_fails_cleanly(monkeypatch):
    def boom(*_):
        raise OSError("no network")

    monkeypatch.setattr(fmea_mod, "_similarity", boom)
    result = fmea_score(_loc_scenario(GOLD), _loc_answer(GOLD), "")
    assert result.passed is False
    assert result.details["error"] == "encoder_unavailable"


def test_ssee_assign_prefers_most_confident_rows():
    sim = [[0.9, 0.85], [0.95, 0.1]]
    # Row 1 is most confident and takes col 0; row 0 falls back to col 1.
    assert fmea_mod._ssee_assign(sim, 0.8) == {1: 0, 0: 1}


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("sentence_transformers") is None,
    reason="sentence-transformers not installed",
)
def test_locations_real_encoder_smoke():
    answer = _loc_answer(["Mechanical seal", "Pump impeller", "Electric motor winding"])
    result = fmea_score(_loc_scenario(GOLD), answer, "")
    if result.details.get("error") == "encoder_unavailable":
        pytest.skip(f"encoder not downloadable here: {result.rationale}")
    assert result.details["family"] == "asset_locations"
    matched = {gold for gold, _, _ in result.details["matches"]}
    assert {"Mechanical Seal", "Impeller"} <= matched
    assert "Electric motor winding" in result.details["extra_locations"]
