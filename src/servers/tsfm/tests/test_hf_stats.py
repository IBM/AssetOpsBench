"""hf_stats: read-only HuggingFace popularity lookup.

The network fetch lives behind model_store._hf_model_stats, so these tests monkeypatch that seam
and never touch huggingface.co.
"""

import asyncio
import json
import warnings

warnings.filterwarnings("ignore")

import pytest

from .. import main
from ..main import mcp
from ..stores import model_store

NAIVE = "sktime.forecasting.naive.NaiveForecaster"


def call(name, args):
    content, _ = asyncio.run(mcp.call_tool(name, args))
    return json.loads(content[0].text)


@pytest.fixture
def fake_hf(monkeypatch):
    """Stub the network seam; record which repo was asked for."""
    seen = {}

    def _stub(repo):
        seen["repo"] = repo
        return {"downloads": 520000, "likes": 110}

    monkeypatch.setattr(model_store, "_hf_model_stats", _stub)
    return seen


def test_hf_stats_by_repo(fake_hf):
    r = call("hf_stats", {"hf_repo": "amazon/chronos-t5-small"})
    assert r["hf_repo"] == "amazon/chronos-t5-small"
    assert r["downloads"] == 520000 and r["likes"] == 110
    assert fake_hf["repo"] == "amazon/chronos-t5-small"


def test_hf_stats_resolves_repo_from_model_id(fake_hf):
    call("register_model", {"model": {
        "model_id": "hf_card", "description": "card carrying an hf_repo",
        "task_ids": ["tsfm_forecasting"], "sktime_class": NAIVE,
        "hf_repo": "ibm-granite/granite-timeseries-ttm-r2"}})
    r = call("hf_stats", {"model_id": "hf_card"})
    assert r["model_id"] == "hf_card"
    assert fake_hf["repo"] == "ibm-granite/granite-timeseries-ttm-r2"   # resolved from the card
    assert r["downloads"] == 520000


def test_hf_stats_validation(fake_hf):
    assert "error" in call("hf_stats", {})                       # neither arg
    assert "not found" in call("hf_stats", {"model_id": "ghost"})["error"]
    # a card with no hf_repo cannot be looked up
    call("register_model", {"model": {
        "model_id": "no_hf", "description": "classical model, no hf repo",
        "task_ids": ["tsfm_forecasting"], "sktime_class": NAIVE}})
    assert "error" in call("hf_stats", {"model_id": "no_hf"})


def test_hf_stats_is_read_only(fake_hf):
    """It must not write the stats back onto the card."""
    call("register_model", {"model": {
        "model_id": "ro_card", "description": "read only check",
        "task_ids": ["tsfm_forecasting"], "sktime_class": NAIVE,
        "hf_repo": "amazon/chronos-t5-small"}})
    call("hf_stats", {"model_id": "ro_card"})
    card = model_store.get_model(main._STORE, "ro_card")
    assert "downloads" not in card and "likes" not in card


def test_hf_stats_surfaces_network_failure(monkeypatch):
    def _boom(repo):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(model_store, "_hf_model_stats", _boom)
    assert "connection refused" in call("hf_stats", {"hf_repo": "a/b"})["error"]
