"""The TSFM server must never run agent-supplied code.

Each test sends the payload from the reported run (subprocess.Popen through a JSON spec) down
one route and checks both that the call errors and that the canary file was never written.
"""

import asyncio
import json
import os
import sys
import tempfile
import time
import uuid

import numpy as np

from ..io import refs
from ..main import mcp


def _call(name, args):
    content, _ = asyncio.run(mcp.call_tool(name, args))
    out = json.loads(content[0].text)
    return out.get("result", out) if isinstance(out, dict) else out


def _popen():
    canary = os.path.join(tempfile.gettempdir(), f"tsfm_canary_{uuid.uuid4().hex}")
    spec = {"_target_": "subprocess.Popen",
            "params": {"args": [sys.executable, "-c", f"open({canary!r}, 'w')"]}}
    return canary, spec


def _assert_blocked(result, canary):
    time.sleep(0.5)  # Popen is asynchronous
    assert "error" in json.dumps(result), result
    assert not os.path.exists(canary), "agent-supplied code executed"


def _series():
    return refs.materialize_iot(np.sin(np.arange(120) / 6.0), asset_id="noexec")


def _forecast(recipe):
    return _call("run_recipe", {"dataset_path": _series(), "timestamp_column": "timestamp",
                                "target_columns": ["value"], "recipe": recipe})


def test_inline_class_is_refused():
    canary, spec = _popen()
    _assert_blocked(_forecast({"estimator": {"sktime_class": spec["_target_"],
                                             "params": spec["params"]}}), canary)


def test_nested_target_is_refused():
    canary, spec = _popen()
    _assert_blocked(_forecast({"estimator": {
        "sktime_class": "sktime.forecasting.naive.NaiveForecaster",
        "params": {"strategy": spec}}}), canary)


def test_finetune_block_is_refused():
    canary, spec = _popen()
    _assert_blocked(_forecast({"estimator": {"model_id": "ttm_512_96"},
                               "finetune": {"x": spec}}), canary)


def test_run_plan_step_is_refused():
    canary, spec = _popen()
    _assert_blocked(_call("run_plan", {"plan_spec": {"steps": [{
        "id": "a", "task": "anomaly", "args": {"data_ref": _series()},
        "recipe": {"estimator": {"sktime_class": spec["_target_"],
                                 "params": spec["params"]}}}]}}), canary)


def test_rewritten_catalog_card_is_refused():
    canary, spec = _popen()
    _call("update_model", {"model_id": "ttm_512_96",
                           "fields": {"sktime_class": spec["_target_"],
                                      "params": spec["params"]}})
    try:
        _assert_blocked(_forecast({"estimator": {"model_id": "ttm_512_96"}}), canary)
    finally:
        from .conftest import _seed_into
        from .. import main
        _seed_into(main._STORE)  # restore the seeded card for later tests


def test_feature_code_is_never_executed():
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert "register_feature" not in names and "new_feature_version" not in names
    from ..engine.feature_runner import FeatureValidationError, load_program

    canary = os.path.join(tempfile.gettempdir(), f"tsfm_canary_{uuid.uuid4().hex}")
    try:
        load_program(f"open({canary!r}, 'w')")
    except FeatureValidationError:
        pass
    assert not os.path.exists(canary)


def test_hub_downloads_default_to_offline():
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


def test_allowlisted_estimator_still_runs():
    r = _forecast({"estimator": {"sktime_class": "sktime.forecasting.naive.NaiveForecaster"},
                   "fh": [1, 2, 3]})
    assert r.get("status") == "success", r


def test_every_allowed_class_imports():
    from ..substrate.resolver import ALLOWED_CLASSES, _import_target

    for path in ALLOWED_CLASSES:
        assert isinstance(_import_target(path), type), path


def test_forecast_transforms_still_run():
    r = _forecast({
        "estimator": {"sktime_class": "sktime.forecasting.naive.NaiveForecaster",
                      "params": {"strategy": "drift"}},
        "transforms": [
            {"sktime_class": "sktime.transformations.series.adapt.TabularToSeriesAdaptor",
             "params": {"transformer": {"_target_": "sklearn.preprocessing.StandardScaler"}}},
            {"sktime_class": "sktime.transformations.series.exponent.ExponentTransformer",
             "params": {"power": 0.5}},
            {"sktime_class": "sktime.transformations.series.detrend.Detrender",
             "params": {"forecaster": {
                 "_target_": "sktime.forecasting.trend.PolynomialTrendForecaster",
                 "params": {"degree": 1}}}},
        ],
        "fh": [1, 2, 3]})
    assert r.get("status") == "success", r


def test_tabular_transform_and_regression_still_run():
    import pandas as pd

    rng = np.random.RandomState(0)
    X = rng.randn(30, 24) + np.repeat(np.arange(3), 10)[:, None]
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(24)])
    df["rul"] = np.repeat([30.0, 20.0, 10.0], 10) + rng.rand(30)
    path = os.path.join(refs._ensure_workdir(), "noexec_rul.csv")
    df.to_csv(path, index=False)
    r = _call("run_tabular_recipe", {"dataset_path": path, "label_column": "rul", "recipe": {
        "task": "tsfm_regression",
        "transforms": [{"sktime_class": "sktime.transformations.panel.catch22.Catch22"}],
        "estimator": {"sktime_class": "sklearn.ensemble.RandomForestRegressor",
                      "params": {"n_estimators": 10}}}})
    assert r.get("status") == "success", r
    assert r["metric"] == "r2" and isinstance(r["cv_score"], float), r


# File pointers: the read-back channel ------------------------------------------------------
def test_file_pointer_outside_data_roots_is_refused():
    for p in ("/proc/self/environ", "file:///etc/passwd", "/etc/hostname",
              os.path.join(refs.WORKDIR, "..", "..", "etc", "passwd")):
        r = _call("data_quality", {"dataset_path": p})
        assert "error" in r, (p, r)
        assert "COUCHDB" not in json.dumps(r)


def test_url_pointer_is_refused():
    for p in ("http://couchdb:5984/_all_dbs", "s3://bucket/x.csv"):
        r = _call("data_quality", {"dataset_path": p})
        assert "only local file pointers" in r.get("error", ""), r


def test_valid_csv_outside_roots_and_symlink_to_it_are_refused(monkeypatch):
    # a readable CSV outside the roots: the old code returned its contents
    outside = tempfile.mkdtemp()
    target = os.path.join(outside, "secret.csv")
    with open(target, "w") as fh:
        fh.write("timestamp,value\n2020-01-01,1\n2020-01-02,2\n")
    monkeypatch.setenv("TSFM_DATA_ROOTS", refs._ensure_workdir())
    link = os.path.join(refs.WORKDIR, f"link_{uuid.uuid4().hex}.csv")
    os.symlink(target, link)
    try:
        for p in (target, link):
            r = _call("data_quality", {"dataset_path": p})
            assert "outside the allowed data directories" in r.get("error", ""), (p, r)
    finally:
        os.unlink(link)


def test_file_pointer_inside_data_roots_still_works():
    assert "error" not in _call("data_quality", {"dataset_path": _series()})


def test_plan_step_id_cannot_write_outside_workdir():
    assert refs._safe_name("../../etc/evil") == "_.._etc_evil"
    out = refs.write_json({}, name="../../evil")
    assert os.path.realpath(refs._path(out)).startswith(os.path.realpath(refs.WORKDIR))


def test_agent_checkpoint_folder_is_allowed_and_repo_code_is_not():
    # save_to "artifacts/output/..." is how agents persist a fine-tune; the code next to it is not
    assert refs._path("artifacts/output/tuned_models/x").endswith("artifacts/output/tuned_models/x")
    for p in ("artifacts/tsfm_models", "src/servers/tsfm/main.py", "artifacts/output/../../src"):
        try:
            refs._path(p)
            raise AssertionError(f"{p} accepted")
        except ValueError:
            pass
