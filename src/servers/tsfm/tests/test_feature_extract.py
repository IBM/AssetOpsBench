"""Tests for the feature-extraction engine helper (engine/composition.extract_features).

Placement: src/servers/tsfm/tests/test_extract_features.py
Patches feature_selection.EXTRACTORS to a tiny deterministic library so assertions are exact.
"""

import numpy as np
import pytest

from ..engine import composition as C
from ..reasoning import feature_selection as FS


@pytest.fixture
def stub_extractors(monkeypatch):
    lib = {"mean": lambda w: float(np.mean(w)), "std": lambda w: float(np.std(w))}
    monkeypatch.setattr(FS, "EXTRACTORS", lib)
    yield lib


def test_whole_series_univariate(stub_extractors):
    cols, F = C.extract_features({"a": np.array([1, 2, 3, 4.0])}, ["mean", "std"])
    assert cols == ["mean", "std"]                 # single channel -> no prefix
    assert F.shape == (1, 2)                        # whole series -> one row
    assert F[0, 0] == 2.5


def test_windowed_univariate(stub_extractors):
    cols, F = C.extract_features({"a": np.array([1, 2, 3, 4.0])}, ["mean"], window=2)
    assert F.shape == (2, 1)                        # tiles [1,2],[3,4]
    assert F[:, 0].tolist() == [1.5, 3.5]


def test_multivariate_whole_series(stub_extractors):
    cols, F = C.extract_features(
        {"a": np.array([1, 2, 3, 4.0]), "b": np.array([10, 20, 30, 40.0])}, ["mean"]
    )
    assert cols == ["a.mean", "b.mean"]            # multivariate -> channel-prefixed
    assert F.tolist() == [[2.5, 25.0]]


def test_multivariate_windowed(stub_extractors):
    cols, F = C.extract_features(
        {"a": np.array([1, 2, 3, 4.0]), "b": np.array([10, 20, 30, 40.0])},
        ["mean", "std"],
        window=2,
    )
    assert cols == ["a.mean", "a.std", "b.mean", "b.std"]
    assert F.shape == (2, 4)
    assert F[0].tolist() == [1.5, 0.5, 15.0, 5.0]


def test_window_larger_than_series_falls_back(stub_extractors):
    cols, F = C.extract_features({"a": np.array([1, 2, 3.0])}, ["mean"], window=99)
    assert F.shape == (1, 1)                        # one whole-series window
    assert F[0, 0] == 2.0


def test_nan_is_zeroed(stub_extractors):
    cols, F = C.extract_features({"a": np.array([np.nan, np.nan])}, ["mean"])
    assert F[0, 0] == 0.0                           # np.nan_to_num applied