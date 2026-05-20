"""Smart Grid scenario JSON shape tests."""

from __future__ import annotations

import json
from pathlib import Path

_SCENARIOS_DIR = Path(__file__).resolve().parents[3] / "scenarios" / "local"
_REQUIRED_FIELDS = {
    "id",
    "type",
    "text",
    "category",
    "characteristic_form",
    "expected_tools",
    "ground_truth",
    "difficulty",
    "domain_tags",
}
_NEGATIVE_REQUIRED_FIELDS = {
    "id",
    "type",
    "text",
    "category",
    "characteristic_form",
    "expected_tools",
    "domain_tags",
}
_VALID_TYPES = {"FMSR", "IoT", "Multi", "TSFM", "WO"}
_VALID_DIFFICULTIES = {"easy", "medium", "hard"}
_VALID_TOOL_PREFIXES = ("fmsr.", "iot.", "tsfm.", "wo.")


def _load(filename: str) -> list[dict]:
    path = _SCENARIOS_DIR / filename
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, list), f"{filename} must be a JSON array"
    return raw


def test_smart_grid_scenarios_count():
    records = _load("smart_grid.json")
    # AOB-FMSR-001 is the original AOB-style scenario. The other 35
    # records are SGT-NNN ports from the HPML Smart Grid MCP scenario corpus.
    assert len(records) == 36


def test_smart_grid_negative_checks_count():
    records = _load("smart_grid_negative_checks.json")
    assert len(records) == 5


def test_smart_grid_scenarios_have_expected_shape():
    for raw in _load("smart_grid.json"):
        missing = _REQUIRED_FIELDS - set(raw)
        assert not missing, f"{raw.get('id', '<missing id>')} missing {sorted(missing)}"
        assert raw["type"] in _VALID_TYPES
        assert raw["difficulty"] in _VALID_DIFFICULTIES
        assert isinstance(raw["text"], str) and raw["text"].strip()
        assert (
            isinstance(raw["characteristic_form"], str)
            and raw["characteristic_form"].strip()
        )
        assert isinstance(raw["expected_tools"], list) and raw["expected_tools"]
        assert isinstance(raw["ground_truth"], dict) and raw["ground_truth"]
        assert isinstance(raw["domain_tags"], list) and raw["domain_tags"]
        for tool_name in raw["expected_tools"]:
            assert tool_name.startswith(_VALID_TOOL_PREFIXES), tool_name


def test_smart_grid_negative_checks_have_expected_shape():
    for raw in _load("smart_grid_negative_checks.json"):
        missing = _NEGATIVE_REQUIRED_FIELDS - set(raw)
        assert not missing, f"{raw.get('id', '<missing id>')} missing {sorted(missing)}"
        assert raw["id"].startswith("SG-NEG-")


def test_smart_grid_scenario_ids_unique():
    main = _load("smart_grid.json")
    neg = _load("smart_grid_negative_checks.json")
    ids = [r["id"] for r in main + neg]
    assert len(ids) == len(set(ids)), f"duplicate scenario IDs detected: {ids}"
