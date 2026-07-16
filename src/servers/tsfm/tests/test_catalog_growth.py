"""Lock in the catalog growth: 100+ FLOps extractors + migrated sktime foundation models."""

import math, warnings
warnings.filterwarnings("ignore")

import numpy as np

from ..reasoning import feature_selection as F
from .conftest import seeded_store
from ..stores import model_store, feature_store


def test_extractor_library_is_100_plus_and_robust():
    assert len(F.EXTRACTORS) >= 100
    # every extractor returns a finite float across diverse windows (incl. constant/short)
    for w in [np.sin(np.arange(40) / 3.0), np.ones(16), np.arange(8.0), np.zeros(10)]:
        for name, fn in F.EXTRACTORS.items():
            v = fn(w)
            assert isinstance(v, float) and math.isfinite(v), f"{name} → {v!r}"


def test_migrated_foundation_models_present_and_resolvable():
    s = seeded_store()
    fc = model_store.list_models(s, task_id="tsfm_forecasting")
    fams = {m.get("model_family") for m in fc}
    assert {"chronos", "moirai", "moment", "timesfm", "timemoe"} <= fams
    chronos = model_store.get_model(s, "amazon__chronos-t5-small")
    assert chronos["sktime_class"].endswith("ChronosForecaster")
    assert chronos["params"]["model_path"] == "amazon/chronos-t5-small"
    assert chronos["training_regime"] == "zero_shot" and chronos["framework"] == "sktime"


def test_migrated_models_carry_provenance_and_validate():
    s = seeded_store()
    migrated = [m for m in model_store.list_models(s) if m.get("created_by") == "migrated_curated"]
    assert len(migrated) >= 25
    for m in migrated:
        assert m["sktime_class"] and m["task_ids"] and m["description"] and m["provenance"] == "pretrained"


def test_ttm_cards_are_runnable():
    """The base TTM cards resolve to an sktime forecaster (model_path set) so run_recipe can
    forecast on them — the IoT-file → TSFM forecast workflow's model entry."""
    from ..substrate import resolver as R
    s = seeded_store()
    for mid in ["ttm_96_28", "ttm_512_96", "ttm_chiller6_512_96_ft"]:
        c = model_store.get_model(s, mid)
        assert c["sktime_class"].endswith("TinyTimeMixerForecaster")
        assert c["params"]["model_path"]                       # points at a checkpoint
        assert R.is_foundation(c) and R.training_regime(c) == "zero_shot"
    runnable = [m for m in model_store.list_models(s, task_id="tsfm_forecasting") if m.get("sktime_class")]
    assert len(runnable) >= 38


def test_every_seed_model_card_validates():
    """Lint: every model card in the catalog must satisfy ModelCard (so update/version/deprecate,
    which re-validate on write, never fail on seed data)."""
    from ..core import schemas
    s = seeded_store()
    bad = []
    for m in model_store.list_models(s, status=None):
        try:
            schemas.ModelCard(**m)
        except Exception as e:
            bad.append((m.get("model_id"), str(e).splitlines()[-2] if "\n" in str(e) else str(e)[:60]))
    assert not bad, f"invalid model cards: {bad[:5]}"
