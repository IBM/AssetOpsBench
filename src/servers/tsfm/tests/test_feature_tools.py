"""The four ported feature-computation tools: count / describe / extract / select_features.

These run the real extractor library (reasoning.feature_selection.EXTRACTORS) on a series - no
model, no torch. count/describe read the feature catalog; extract/select compute over data.
"""
import asyncio
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from ..io import refs
from ..main import mcp
from ..reasoning import feature_selection as FS


def call(name, args):
    content, _ = asyncio.run(mcp.call_tool(name, args))
    return json.loads(content[0].text)


def _series(n=240, asset="feat"):
    t = np.arange(n)
    sig = 20 + 4 * np.sin(t / 24 * 2 * np.pi) + 0.02 * t + np.random.RandomState(0).normal(0, .3, n)
    return refs.materialize_iot(sig, asset_id=asset)


THREE = list(FS.EXTRACTORS)[:3]          # real extractor names, e.g. mean/std/min


def test_count_features_returns_totals():
    d = call("count_features", {})
    assert set(d) >= {"extractors", "transforms", "total"}
    assert d["total"] == d["extractors"] + d["transforms"]


def test_describe_features_reports_unknown_names():
    d = call("describe_features", {"names": ["definitely_not_a_feature"]})
    assert d["unknown"] == ["definitely_not_a_feature"]
    assert d["features"] == []


def test_describe_features_requires_a_name():
    d = call("describe_features", {"names": []})
    assert "error" in d


def test_extract_features_whole_series():
    ref = _series(asset="extract_whole")
    d = call("extract_features", {"dataset_path": ref, "extractors": THREE,
                                  "target_columns": ["value"]})
    assert "error" not in d
    assert d["n_windows"] == 1
    assert d["columns"] == THREE
    assert len(d["features"]) == 1 and len(d["features"][0]) == 3


def test_extract_features_windowed():
    ref = _series(n=240, asset="extract_win")
    d = call("extract_features", {"dataset_path": ref, "extractors": THREE[:2],
                                  "target_columns": ["value"], "window": 48})
    assert "error" not in d
    assert d["n_windows"] == 5                       # 240 / 48
    assert len(d["features"]) == 5 and len(d["features"][0]) == 2


def test_extract_features_rejects_unknown_extractor():
    ref = _series(asset="extract_bad")
    d = call("extract_features", {"dataset_path": ref, "extractors": ["not_real"],
                                  "target_columns": ["value"]})
    assert "error" in d and "unknown extractor" in d["error"]


def test_extract_features_requires_target_columns():
    ref = _series(asset="extract_notarget")
    d = call("extract_features", {"dataset_path": ref, "extractors": THREE, "target_columns": []})
    assert "error" in d


def test_select_features_returns_a_shortlist():
    ref = _series(asset="select")
    d = call("select_features", {"dataset_path": ref, "channel": "value",
                                 "extractors": list(FS.EXTRACTORS)[:6]})
    assert "error" not in d
    assert isinstance(d["selected"], list)          # a ranked shortlist (names only)
    assert d["detail_file"].startswith("file://")


def test_select_features_rejects_unknown_extractor():
    ref = _series(asset="select_bad")
    d = call("select_features", {"dataset_path": ref, "channel": "value",
                                 "extractors": ["not_real"]})
    assert "error" in d and "unknown extractor" in d["error"]
