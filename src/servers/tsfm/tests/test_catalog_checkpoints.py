"""Every local checkpoint in the catalog resolves, fits and forecasts.

This is the test that the rest of the plumbing cannot substitute for.
`preload_models.py --check` proves a directory exists; schema validation proves
a card is well formed. Neither proves the weights load into the estimator the
card names, or that a fit returns the horizon the card advertises. A card can be
valid, its checkpoint present, and the pair still unusable.

Run against the repo's own catalog, or point it elsewhere:

    uv run pytest src/servers/tsfm/tests/test_catalog_checkpoints.py -v
    AOB_MODEL_CATALOG=/path/to/private_catalog.json uv run pytest ... -v

Skips rather than fails when sktime or the TTM extra is missing, so it does not
break a checkout that never installed the tsfm group.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# Before numpy/pandas: they arrive with the tsfm group, so importing them first
# turns a missing group into a collection error rather than a skip.
pytest.importorskip("sktime", reason="requires the tsfm dependency group")

import numpy as np
import pandas as pd

from servers.tsfm.substrate import resolver as R

DEFAULT_CATALOG = Path("src/couchdb/scenarios_data/shared/tsfm/model_catalog.json")


def _catalog_path() -> Path:
    env = os.environ.get("AOB_MODEL_CATALOG")
    if env:
        return Path(env)
    root = os.environ.get("SCENARIOS_DATA_DIR")
    if root:
        return Path(root) / "shared/tsfm/model_catalog.json"
    return DEFAULT_CATALOG


def _local_cards() -> list[dict]:
    """Active cards whose weights are a directory on disk, not a Hub repo."""
    path = _catalog_path()
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    cards = raw if isinstance(raw, list) else raw.get("docs", [raw])
    out = []
    for c in cards:
        if (c.get("status") or "active") != "active":
            continue
        if c.get("hf_repo"):
            continue
        mp = (c.get("params") or {}).get("model_path")
        if mp and Path(mp).is_dir():
            out.append(c)
    return out


def _ids(cards):
    return [c["model_id"] for c in cards]


CARDS = _local_cards()


@pytest.mark.parametrize("card", CARDS, ids=_ids(CARDS) if CARDS else [])
def test_checkpoint_forecasts(card: dict) -> None:
    """resolve -> fit -> predict, and the horizon matches what the card claims."""
    ctx = int(card["context_length"])
    horizon = int(card["prediction_length"])

    # The estimator needs at least context_length of history. Give it a series
    # with actual structure rather than noise, so a silently broken checkpoint
    # producing constants is visible in the variance assertion below.
    n = max(ctx * 3, ctx + horizon + 10)
    t = np.arange(n)
    y = pd.Series(10 + np.sin(t / 7.0) * 2 + t * 0.01)

    forecaster = R.resolve(card)
    forecaster.fit(y, fh=list(range(1, horizon + 1)))
    pred = np.asarray(forecaster.predict()).ravel()

    assert len(pred) == horizon, (
        f"{card['model_id']} claims prediction_length={horizon} "
        f"but returned {len(pred)} points"
    )
    assert np.isfinite(pred).all(), f"{card['model_id']} returned non-finite values"
    assert pred.std() > 0, (
        f"{card['model_id']} returned a constant forecast, which usually means "
        "the weights did not load and the head is at its initialisation"
    )


@pytest.mark.parametrize("card", CARDS, ids=_ids(CARDS) if CARDS else [])
def test_checkpoint_serves_rather_than_trains(card: dict) -> None:
    """A checkpoint card must not fine-tune on fit.

    sktime's TTM defaults to fit_strategy="minimal", which trains. A card that
    pins neither params.fit_strategy nor training_regime inherits that default,
    re-tunes the already-tuned weights on every fit, and sends run_recipe down
    the expanding-window refit loop instead of a single holdout.
    """
    assert R.training_regime(card) == "zero_shot", (
        f"{card['model_id']} resolves to '{R.training_regime(card)}'. Pin both "
        'params.fit_strategy="zero-shot" and training_regime="zero_shot".'
    )


@pytest.mark.skipif(not CARDS, reason="no active local-artifact cards in the catalog")
def test_card_location_fields_agree() -> None:
    """artifact_path, model_checkpoint and params.model_path name one place.

    Only params.model_path is read at load time. The others are metadata that
    drifts silently, and register_finetuned sets all three identically, so a
    seeded card that disagrees teaches the agent a shape the code does not use.
    """
    bad = []
    for c in CARDS:
        mp = (c.get("params") or {}).get("model_path")
        for field in ("artifact_path", "model_checkpoint"):
            val = c.get(field)
            if val is not None and val != mp:
                bad.append(f"{c['model_id']}: {field}={val!r} != params.model_path={mp!r}")
    assert not bad, "location fields disagree:\n  " + "\n  ".join(bad)


@pytest.mark.skipif(not CARDS, reason="no active local-artifact cards in the catalog")
def test_distinct_cards_have_distinct_weights() -> None:
    """Report cards that share a checkpoint's bytes.

    Two cards pointing at byte-identical weights cannot be told apart by any
    scenario that asks an agent to choose between them, so a preference test
    over such a pair measures nothing. Informational: xfail rather than fail,
    because shipping a placeholder copy is a legitimate interim state.
    """
    import hashlib

    by_digest: dict[str, list[str]] = {}
    for c in CARDS:
        w = Path((c.get("params") or {})["model_path"]) / "model.safetensors"
        if not w.exists():
            continue
        digest = hashlib.sha256(w.read_bytes()).hexdigest()
        by_digest.setdefault(digest, []).append(c["model_id"])

    dupes = {d: ids for d, ids in by_digest.items() if len(ids) > 1}
    if dupes:
        pytest.xfail("cards sharing identical weights: " + "; ".join(
            ", ".join(ids) for ids in dupes.values()))
