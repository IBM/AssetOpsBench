"""Docstring contract for the model-catalog tools.

Two things are checked: the docstrings follow the Google/IoT format (summary + Args + Returns,
every parameter documented), and the caveats they promise are actually true of the code. The
second half matters most - a docstring that lies is worse than none.
"""

import asyncio
import ast
import inspect
import warnings

warnings.filterwarnings("ignore")

import pytest

from .. import main as M
from ..main import mcp
from ..stores import model_store

WRITE_TOOLS = ["register_model", "model_template", "register_finetuned", "update_model",
               "deprecate_model", "new_model_version", "resolve_model"]
NAIVE = "sktime.forecasting.naive.NaiveForecaster"


def _descriptions():
    return {t.name: (t.description or "") for t in asyncio.run(mcp.list_tools())}


def _params(name):
    fn = getattr(M, name)
    return [p for p in inspect.signature(fn).parameters]


@pytest.mark.parametrize("name", WRITE_TOOLS)
def test_docstring_reaches_the_mcp_boundary(name):
    assert _descriptions()[name].strip(), f"{name} has no description over MCP"


@pytest.mark.parametrize("name", WRITE_TOOLS)
def test_docstring_has_summary_and_returns(name):
    d = _descriptions()[name]
    assert d.splitlines()[0].strip().endswith("."), f"{name}: first line is not a summary sentence"
    assert "Returns:" in d, f"{name}: no Returns: section"


@pytest.mark.parametrize("name", WRITE_TOOLS)
def test_every_parameter_is_documented(name):
    d, ps = _descriptions()[name], _params(name)
    if not ps:
        return                                    # no-arg tools need no Args: section
    assert "Args:" in d, f"{name}: has params {ps} but no Args: section"
    missing = [p for p in ps if f"{p}:" not in d]
    assert not missing, f"{name}: undocumented params {missing}"


# ---- the caveats must be TRUE, not aspirational ----
def test_register_model_really_does_overwrite():
    """Its docstring warns the card is overwritten without warning."""
    assert "OVERWRITTEN" in _descriptions()["register_model"]
    card = {"model_id": "ovw", "description": "first version", "task_ids": ["tsfm_forecasting"],
            "sktime_class": NAIVE}
    M.register_model(card)
    M.register_model({**card, "description": "second version"})
    assert model_store.get_model(M._STORE, "ovw")["description"] == "second version"


def test_register_finetuned_really_falls_back_to_ttm():
    """Its docstring warns an unknown base_model_id silently defaults sktime_class to TTM."""
    d = _descriptions()["register_finetuned"]
    assert "not checked for existence" in d.lower() and "tinytimemixer" in d.lower()
    r = M.register_finetuned(model_id="ghost_ft", checkpoint_path="/ckpt/g",
                             base_model_id="no_such_base", context_length=96,
                             prediction_length=28, description="finetune of a ghost")
    assert r.model_dump()["sktime_class"].endswith("TinyTimeMixerForecaster")


def test_register_finetuned_really_inherits_from_the_base():
    M.register_model({"model_id": "inh_base", "description": "naive base",
                      "task_ids": ["tsfm_forecasting"], "sktime_class": NAIVE,
                      "params": {"strategy": "drift"}})
    r = M.register_finetuned(model_id="inh_ft", checkpoint_path="/ckpt/i",
                             base_model_id="inh_base", context_length=96,
                             prediction_length=28, description="finetune of naive")
    card = r.model_dump()
    assert card["sktime_class"] == NAIVE                    # inherited
    assert card["params"]["model_path"] == "/ckpt/i"        # checkpoint wired in
    assert card["params"]["strategy"] == "drift"            # base params kept


def test_card_returning_tools_really_are_flat():
    """update/deprecate/new_version/register_finetuned document a FLAT card; register nests it."""
    for n in ["update_model", "deprecate_model", "new_model_version", "register_finetuned"]:
        assert "FLAT" in _descriptions()[n], f"{n} should document the flat shape"
    M.register_model({"model_id": "flat_x", "description": "flat shape check",
                      "task_ids": ["tsfm_forecasting"], "sktime_class": NAIVE})
    assert "card" in M.register_model({"model_id": "flat_y", "description": "nested shape",
                                       "task_ids": ["tsfm_forecasting"],
                                       "sktime_class": NAIVE}).model_dump()
    d = M.update_model("flat_x", {"domain": "energy"}).model_dump()
    assert d["model_id"] == "flat_x" and "card" not in d    # flat


def test_new_model_version_really_defaults_the_id_and_links():
    assert "<model_id>_v<version>" in _descriptions()["new_model_version"]
    M.register_model({"model_id": "nv", "description": "version base",
                      "task_ids": ["tsfm_forecasting"], "sktime_class": NAIVE, "version": "1"})
    out = M.new_model_version("nv", {"context_length": 128}).model_dump()
    assert out["model_id"] == "nv_v2" and out["supersedes"] == "nv"
    assert model_store.get_model(M._STORE, "nv")["status"] == "superseded"


def test_deprecate_is_really_reversible():
    assert "Reversible" in _descriptions()["deprecate_model"]
    M.register_model({"model_id": "dep", "description": "deprecate me",
                      "task_ids": ["tsfm_forecasting"], "sktime_class": NAIVE})
    assert M.deprecate_model("dep", "obsolete").model_dump()["status"] == "deprecated"
    assert M.update_model("dep", {"status": "active"}).model_dump()["status"] == "active"


def test_resolve_model_really_is_read_only():
    assert "Does NOT download weights or fit" in _descriptions()["resolve_model"]
    M.register_model({"model_id": "ro", "description": "resolve check",
                      "task_ids": ["tsfm_forecasting"], "sktime_class": NAIVE})
    before = dict(model_store.get_model(M._STORE, "ro"))
    M.resolve_model("ro")
    assert model_store.get_model(M._STORE, "ro") == before   # untouched


def test_model_template_is_static_and_its_example_registers():
    d = _descriptions()["model_template"]
    assert "reads nothing from the catalog" in d
    t = M.model_template()
    assert set(t.required_fields) == {"model_id", "description", "task_ids"}
    assert "error" not in M.register_model(t.example).model_dump()
