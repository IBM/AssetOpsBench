"""The server provides evidence (profile) — facts, not decisions — for the agent to reason from."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .conftest import seeded_store
from ..reasoning import profile


def test_profile_gives_facts_not_decisions():
    s = seeded_store()
    ev = profile.profile_series(s, "chiller_6")
    assert ev["n_channels"] == 3 and ev["dominant_period"] and ev["non_stationary"] in (True, False)
    # evidence only — no "recommended_lookback"/"chosen_model" keys
    assert not any(k.startswith("recommend") or k.startswith("chosen") for k in ev)
    ctx = profile.available_contexts(s, "tsfm_forecasting")
    assert any(m["context_length"] for m in ctx["models"])
