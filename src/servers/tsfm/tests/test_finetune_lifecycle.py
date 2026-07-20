"""End-to-end: fine-tune -> save checkpoint -> register -> serve, with no torch and no network.

Uses fake_hf_forecaster.FakeTTMForecaster, a double implementing the four things the server relies
on from a HuggingFace-backed wrapper: a model_path param, a fit_strategy defaulting to "minimal"
(i.e. it TRAINS unless told otherwise, exactly like sktime's TTM), a .model exposing
save_pretrained, and from_pretrained to read it back.

Each test here pins a bug that used to be live:
  * recipe["finetune"] set the regime to fine_tune and never reached the estimator
  * run_recipe trained a model and dropped the weights; nothing produced a checkpoint_path
  * register_finetuned left training_regime unset, so serving a checkpoint re-trained it
  * register_finetuned accepted bases whose wrapper takes no model_path
"""
import json
import os
import shutil

import numpy as np
import pytest

from servers.tsfm import main as M
from servers.tsfm.io import refs
from servers.tsfm.stores import model_store
from servers.tsfm.substrate import resolver as R
from servers.tsfm.tests import fake_hf_forecaster as FK

FAKE = "servers.tsfm.tests.fake_hf_forecaster.FakeTTMForecaster"
SP, N, FH = 24, 480, [1, 2, 3, 4, 5, 6]


@pytest.fixture()
def series():
    t = np.arange(N)
    y = 26 + 7 * np.sin(t / SP * 2 * np.pi) + 0.004 * t + np.random.RandomState(0).normal(0, .3, N)
    return refs.materialize_iot(y, asset_id="chiller_6")


@pytest.fixture()
def base_card():
    M.register_model({
        "model_id": "fk_base", "description": "fake TTM base, zero-shot pinned",
        "task_ids": ["tsfm_forecasting"], "provenance": "pretrained", "sktime_class": FAKE,
        "params": {"model_path": "fake-hub/ttm", "fit_strategy": "zero-shot", "sp": SP},
        "context_length": 512, "prediction_length": 96,
    })
    return "fk_base"


@pytest.fixture()
def ckpt(tmp_path):
    p = str(tmp_path / "ck")
    shutil.rmtree(p, ignore_errors=True)
    return p


def _run(ref, recipe):
    return M.run_recipe(dataset_path=ref, timestamp_column="timestamp",
                        target_columns=["value"], recipe=recipe).model_dump()


def test_zero_shot_card_does_not_train(series, base_card):
    FK.reset()
    r = _run(series, {"estimator": {"model_id": base_card}, "fh": FH,
                      "eval": {"metrics": ["mape"]}})
    assert "error" not in r
    assert r["training_regime"] == "zero_shot"
    assert FK.CALLS["trained"] == 0


def test_unpinned_foundation_card_is_fine_tune_not_zero_shot():
    """The estimator's OWN default decides. FakeTTM (like sktime's TTM) defaults to
    fit_strategy='minimal', which trains - so an unpinned card must NOT read as zero_shot."""
    assert R.training_regime({"sktime_class": FAKE, "params": {"model_path": "x"}}) == "fine_tune"
    assert R.training_regime(
        {"sktime_class": FAKE, "params": {"model_path": "x", "fit_strategy": "zero-shot"}}
    ) == "zero_shot"


def test_finetune_block_reaches_the_estimator_and_trains(series, base_card):
    """recipe['finetune'] used to only flip the regime label; the params never reached fit()."""
    FK.reset()
    r = _run(series, {"estimator": {"model_id": base_card},
                      "finetune": {"fit_strategy": "minimal",
                                   "training_args": {"num_train_epochs": 3}},
                      "fh": FH, "eval": {"metrics": ["mape"]}})
    assert "error" not in r
    assert r["training_regime"] == "fine_tune"
    assert FK.CALLS["trained"] > 0          # it ACTUALLY trained


def test_save_to_writes_a_checkpoint_and_reports_the_path(series, base_card, ckpt):
    r = _run(series, {"estimator": {"model_id": base_card},
                      "finetune": {"fit_strategy": "minimal"},
                      "save_to": ckpt, "fh": FH, "eval": {"metrics": ["mape"]}})
    assert "error" not in r
    assert r["checkpoint_path"] == os.path.abspath(ckpt)
    assert os.path.isdir(ckpt) and "config.json" in os.listdir(ckpt)
    rec = json.loads(open(r["results_file"][7:]).read())
    assert rec["checkpoint_path"] == os.path.abspath(ckpt)
    assert rec["trained"] is True


def test_folds_are_surfaced_so_scores_are_interpretable(series, base_card):
    """zero_shot scores ONE holdout, fit_on_series averages ~20 folds, both report as
    backtest_score. Without the fold count the caller cannot tell if two rows are comparable."""
    zs = _run(series, {"estimator": {"model_id": base_card}, "fh": FH,
                       "eval": {"metrics": ["mape"]}})
    ft = _run(series, {"estimator": {"model_id": base_card},
                       "finetune": {"fit_strategy": "minimal"}, "fh": FH,
                       "eval": {"metrics": ["mape"]}})
    assert zs["folds"] == 1
    assert ft["folds"] > 1


def test_full_lifecycle_finetune_save_register_serve(series, base_card, ckpt):
    r = _run(series, {"estimator": {"model_id": base_card},
                      "finetune": {"fit_strategy": "minimal"},
                      "save_to": ckpt, "fh": FH, "eval": {"metrics": ["mape"]}})
    assert "error" not in r

    card = M.register_finetuned(
        model_id="fk_ft", checkpoint_path=r["checkpoint_path"], base_model_id=base_card,
        context_length=512, prediction_length=96,
        description="fake TTM fine-tuned on chiller 6",
    ).model_dump()
    assert card["provenance"] == "finetuned"          # history
    assert card["training_regime"] == "zero_shot"     # what the NEXT fit does: load + predict
    assert card["params"]["fit_strategy"] == "zero-shot"

    FK.reset()
    served = _run(series, {"estimator": {"model_id": "fk_ft"}, "fh": FH,
                           "eval": {"metrics": ["mape"]}})
    assert "error" not in served
    assert served["training_regime"] == "zero_shot"
    assert FK.CALLS["trained"] == 0                            # serving must not re-train
    assert set(FK.CALLS["loaded_from"]) == {"checkpoint"}      # from OUR weights, not the hub

    lin = M.get_model_lineage("fk_ft").model_dump()
    assert lin["ancestors"] == [base_card]


def test_save_to_errors_clearly_without_a_checkpoint_format(series):
    r = _run(series, {"estimator": {"sktime_class": "sktime.forecasting.theta.ThetaForecaster",
                                    "params": {"sp": SP}},
                      "save_to": "/tmp/should_not_exist", "fh": FH})
    assert "error" in r
    assert "save_pretrained" in r["error"]
    assert not os.path.isdir("/tmp/should_not_exist")


def test_register_finetuned_rejects_a_base_without_model_path():
    M.register_model({"model_id": "classical", "description": "classical base",
                      "task_ids": ["tsfm_forecasting"], "provenance": "trained",
                      "sktime_class": "sktime.forecasting.theta.ThetaForecaster",
                      "params": {"sp": SP}})
    d = M.register_finetuned(model_id="bad", checkpoint_path="/x", base_model_id="classical",
                             context_length=96, prediction_length=24,
                             description="should be rejected").model_dump()
    assert "error" in d and "model_path" in d["error"]
    assert model_store.get_model(M._STORE, "bad") is None
