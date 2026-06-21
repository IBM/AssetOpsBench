"""The agent reasons; the server provides evidence (profile) and scores choices (scoring)."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tsfm.bootstrap import fresh_store
from tsfm.reasoning import profile
from tsfm.reasoning import scoring


def test_profile_gives_facts_not_decisions():
    s = fresh_store()
    ev = profile.profile_series(s, "chiller_6")
    assert ev["n_channels"] == 3 and ev["dominant_period"] and ev["non_stationary"] in (True, False)
    # evidence only — no "recommended_lookback"/"chosen_model" keys
    assert not any(k.startswith("recommend") or k.startswith("chosen") for k in ev)
    ctx = profile.available_contexts(s, "tsfm_forecasting")
    assert any(m["context_length"] for m in ctx["models"])


def test_scoring_rewards_reasoned_choices_over_defaults():
    s = fresh_store()
    p = profile.profile_series(s, "chiller_6")["dominant_period"]
    good = {"lookback": 2 * p, "context_length": 512, "forecast_horizon": 48,
            "feature_ids": ["efe_time_robust_norm_v1"], "model_id": "ttm_energy_512_96"}
    naive = {"lookback": 8, "context_length": 96, "forecast_horizon": 16,
             "feature_ids": [], "model_id": "ttm_96_28"}
    sg = scoring.score_forecasting_choices(s, "chiller_6", good, question="forecast next 48 hours")
    sn = scoring.score_forecasting_choices(s, "chiller_6", naive, question="forecast next 48 hours")
    assert sg["score"] > sn["score"] and sg["score"] >= 0.8


def test_anomaly_scoring_data_driven():
    s = fresh_store()
    good = {"detection_window": 50, "model_id": "akits_relationshipad", "thresholding": "dynamic"}
    naive = {"detection_window": 8, "model_id": "akits_windowad", "thresholding": "static"}
    assert (scoring.score_anomaly_choices(s, "chiller_6", good)["score"]
            > scoring.score_anomaly_choices(s, "chiller_6", naive)["score"])


def test_param_audit_flags():
    s = fresh_store()
    ev = profile.profile_series(s, "chiller_6")
    a = scoring.param_audit(ev, {"lookback": 100, "context_length": 512})
    assert a["context_covers_lookback"] is True
    a2 = scoring.param_audit(ev, {"lookback": 600, "context_length": 96})
    assert a2["context_covers_lookback"] is False
