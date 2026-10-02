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


# --------------------------------------------------------------------------- #
# Missing values: no features from gapped data unless the caller says how to fill
# --------------------------------------------------------------------------- #
def _gapped(n=240, asset="gapped", every=17):
    t = np.arange(n, dtype=float)
    sig = 20 + 4 * np.sin(t / 24 * 2 * np.pi) + 0.02 * t
    sig[::every] = np.nan
    return refs.materialize_iot(sig, asset_id=asset), int(np.isnan(sig).sum())


def test_extract_features_refuses_gapped_series_without_impute():
    ref, n_gaps = _gapped(asset="extract_gap_refuse")
    d = call("extract_features", {"dataset_path": ref, "extractors": ["mean", "std"],
                                  "target_columns": ["value"]})
    assert "error" in d
    assert "missing values" in d["error"] and f"{n_gaps} of 240" in d["error"]
    assert "interpolate" in d["error"]                   # tells the agent what to do


def test_extract_features_with_interpolate_gives_real_values():
    ref, _ = _gapped(asset="extract_gap_interp")
    d = call("extract_features", {"dataset_path": ref, "extractors": ["mean", "std"],
                                  "target_columns": ["value"], "impute": "interpolate"})
    assert "error" not in d
    mean, std = d["features"][0]
    assert 19 < mean < 25 and std > 1                    # not the old silent 0.0
    assert "impute='interpolate'" in d["message"]


def test_extract_features_with_drop_and_zero():
    ref, n_gaps = _gapped(asset="extract_gap_drop")
    d = call("extract_features", {"dataset_path": ref, "extractors": ["length"],
                                  "target_columns": ["value"], "impute": "drop"})
    assert "error" not in d and d["features"][0][0] == 240 - n_gaps
    z = call("extract_features", {"dataset_path": ref, "extractors": ["min"],
                                  "target_columns": ["value"], "impute": "zero"})
    assert "error" not in z and z["features"][0][0] == 0.0  # the caller asked for zeros


def test_extract_features_rejects_unknown_impute():
    ref, _ = _gapped(asset="extract_gap_badimpute")
    d = call("extract_features", {"dataset_path": ref, "extractors": ["mean"],
                                  "target_columns": ["value"], "impute": "ffill"})
    assert "error" in d and "unknown impute" in d["error"]


def test_extract_features_reports_uncomputable_values_as_null(monkeypatch):
    monkeypatch.setitem(FS.EXTRACTORS, "always_nan", lambda w: float("nan"))
    ref = _series(asset="extract_null")
    d = call("extract_features", {"dataset_path": ref, "extractors": ["mean", "always_nan"],
                                  "target_columns": ["value"]})
    assert "error" not in d
    assert d["features"][0][1] is None                   # null, never 0.0
    assert "always_nan" in d["message"]


def test_select_features_refuses_gapped_series_without_impute():
    ref, _ = _gapped(asset="select_gap_refuse")
    d = call("select_features", {"dataset_path": ref, "channel": "value",
                                 "extractors": list(FS.EXTRACTORS)[:6]})
    assert "error" in d and "missing values" in d["error"]
    ok = call("select_features", {"dataset_path": ref, "channel": "value",
                                  "extractors": list(FS.EXTRACTORS)[:6], "impute": "interpolate"})
    assert "error" not in ok


def test_gate_gaps_drop_keeps_channels_aligned():
    from ..engine import composition

    a = np.array([1.0, np.nan, 3.0, 4.0])
    b = np.array([1.0, 2.0, np.inf, 4.0])
    out = composition.gate_gaps({"a": a, "b": b}, "drop")
    assert out["a"].tolist() == [1.0, 4.0] and out["b"].tolist() == [1.0, 4.0]


def test_flux_percentile_extractors_are_registered():
    # They used np.percentile(interpolation=...), which NumPy 2 rejects, so the registry's
    # import-time probe dropped them silently while the catalog still listed them.
    for name in ("flux_percentile_ratio_mid20", "flux_percentile_ratio_mid35",
                 "flux_percentile_ratio_mid50", "flux_percentile_ratio_mid65",
                 "flux_percentile_ratio_mid80", "percent_difference_flux_percentile"):
        assert name in FS.EXTRACTORS, name
