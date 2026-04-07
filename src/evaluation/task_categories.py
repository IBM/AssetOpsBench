"""Map AssetOpsBench scenarios to paper task categories."""

from __future__ import annotations

from typing import Any

TASK_CATEGORIES: dict[str, dict[str, Any]] = {
    "asset_configuration": {
        "description": "FMEA retrieval, KPI selection, sensor listing",
        "keywords": [
            "failure mode",
            "sensor",
            "asset",
            "site",
            "kpi",
            "configure",
            "list",
            "fmea",
        ],
        "servers": ["iot", "fmsr"],
        "scenario_types": ["knowledge", "tool skill"],
    },
    "diagnostics_reasoning": {
        "description": "Root cause analysis, fault explanation, vibration diagnosis",
        "keywords": [
            "diagnos",
            "fault",
            "root cause",
            "vibration",
            "bearing",
            "spectrum",
            "fft",
            "envelope",
            "severity",
            "anomal",
            "failure",
        ],
        "servers": ["fmsr", "vibration"],
        "scenario_types": ["reasoning", "skill"],
    },
    "forecasting_monitoring": {
        "description": "Anomaly detection, trend forecasting, sensor monitoring",
        "keywords": [
            "forecast",
            "predict",
            "anomaly",
            "trend",
            "time series",
            "tsad",
            "monitor",
            "history",
            "tsfm",
        ],
        "servers": ["tsfm", "iot"],
        "scenario_types": ["skill", "tool skill"],
    },
    "maintenance_planning": {
        "description": "Work order generation, maintenance scheduling, health summary",
        "keywords": [
            "work order",
            "maintenance",
            "preventive",
            "corrective",
            "schedule",
            "health",
            "alert",
            "event",
            "failure code",
        ],
        "servers": ["wo"],
        "scenario_types": ["skill", "tool skill", "reasoning"],
    },
}


def categorize_scenario(scenario: Any) -> str:
    """Return the best-matching task category for *scenario*.

    Matching is done by keyword overlap with the scenario ``text`` field.
    Falls back to ``"uncategorized"`` when no keywords match.
    """
    text = getattr(scenario, "text", "").lower()
    best_cat = "uncategorized"
    best_score = 0
    for cat, meta in TASK_CATEGORIES.items():
        score = sum(1 for kw in meta["keywords"] if kw in text)
        if score > best_score:
            best_score = score
            best_cat = cat
    return best_cat


def categorize_scenarios(scenarios: list) -> dict[str, list]:
    """Group *scenarios* by task category.

    Returns a mapping of category name -> list of scenarios.
    """
    groups: dict[str, list] = {cat: [] for cat in TASK_CATEGORIES}
    groups["uncategorized"] = []
    for s in scenarios:
        groups[categorize_scenario(s)].append(s)
    return groups
