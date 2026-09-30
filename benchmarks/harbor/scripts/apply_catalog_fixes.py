#!/usr/bin/env python
"""Apply the audited fixes to a model catalog, in place.

Every edit is reported; one already present is skipped. The fixes:

  ttm, tspulse_ad, tspulse_clf
      "params": {} hands the wrapper its own defaults, which name a checkpoint:
      TinyTimeMixerForecaster falls back to "ibm/TTM", both TSPulse estimators
      to "ibm-granite/granite-timeseries-tspulse-r1", and the classifier to a
      non-main revision. Nothing in the card says so, so preload_models.py
      reports them as classical models and caches nothing. They then download
      at first use: fine on a laptop, a network error inside the image.

  ttm (again)
      TinyTimeMixerForecaster's fit_strategy defaults to "minimal", which
      fine-tunes. Pinning params.fit_strategy and training_regime makes it serve.

  chronos
      No hf_repo, so classify() guesses the target from the path shape.

  google__timesfm-2.5-200m
      created_by/source claim a migration that did not happen.

    uv run python benchmarks/harbor/scripts/apply_catalog_fixes.py
    uv run python benchmarks/harbor/scripts/apply_catalog_fixes.py --write

Dry run by default. --move-energy also repoints ttm_energy_168_24 from
artifacts/output/tuned_models/ to artifacts/tsfm_models/, where the repo now
keeps it. The repo's own catalog already points there; use the flag on a
catalog that does not yet, such as a private suite's (--catalog PATH).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_CATALOG = Path("src/couchdb/scenarios_data/shared/tsfm/model_catalog.json")
TTM_REPO = "ibm-granite/granite-timeseries-ttm-r2"
TSPULSE_REPO = "ibm-granite/granite-timeseries-tspulse-r1"
TSPULSE_CLF_REV = "tspulse-block-dualhead-512-p16-r1"
ENERGY_OLD = "artifacts/output/tuned_models/ttm_energy_168_24"
ENERGY_NEW = "artifacts/tsfm_models/ttm_energy_168_24"


def catalog_path(explicit: Path | None) -> Path:
    if explicit:
        return explicit
    if os.environ.get("AOB_MODEL_CATALOG"):
        return Path(os.environ["AOB_MODEL_CATALOG"])
    if os.environ.get("SCENARIOS_DATA_DIR"):
        return Path(os.environ["SCENARIOS_DATA_DIR"]) / "shared/tsfm/model_catalog.json"
    return DEFAULT_CATALOG


def setval(target: dict, key: str, value, changes: list[str], label: str) -> None:
    if target.get(key) == value:
        return
    target[key] = value
    changes.append(f"{label}={value!r}")


def fix(card: dict, move_energy: bool) -> list[str]:
    mid = card.get("model_id")
    params = card.setdefault("params", {})
    ch: list[str] = []

    if mid == "ttm":
        setval(params, "model_path", TTM_REPO, ch, "params.model_path")
        setval(params, "fit_strategy", "zero-shot", ch, "params.fit_strategy")
        setval(card, "hf_repo", TTM_REPO, ch, "hf_repo")
        setval(card, "training_regime", "zero_shot", ch, "training_regime")
    elif mid == "tspulse_ad":
        setval(params, "model_path", TSPULSE_REPO, ch, "params.model_path")
        setval(card, "hf_repo", TSPULSE_REPO, ch, "hf_repo")
    elif mid == "tspulse_clf":
        setval(params, "model_path", TSPULSE_REPO, ch, "params.model_path")
        setval(params, "revision", TSPULSE_CLF_REV, ch, "params.revision")
        setval(card, "hf_repo", TSPULSE_REPO, ch, "hf_repo")
    elif mid == "chronos":
        mp = params.get("model_path")
        if mp:
            setval(card, "hf_repo", mp, ch, "hf_repo")
    elif mid == "google__timesfm-2.5-200m":
        setval(card, "created_by", "seed", ch, "created_by")
        setval(card, "source", "sktime-foundation-survey", ch, "source")
    elif mid == "ttm_energy_168_24" and move_energy:
        for key in ("artifact_path", "model_checkpoint"):
            if card.get(key) == ENERGY_OLD:
                setval(card, key, ENERGY_NEW, ch, key)
        if params.get("model_path") == ENERGY_OLD:
            setval(params, "model_path", ENERGY_NEW, ch, "params.model_path")
    return ch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", type=Path, default=None)
    p.add_argument("--write", action="store_true", help="apply (default: print)")
    p.add_argument("--move-energy", action="store_true",
                   help="repoint ttm_energy_168_24 at artifacts/tsfm_models/ "
                        "(move the directory on disk first)")
    args = p.parse_args()

    cat = catalog_path(args.catalog)
    if not cat.is_file():
        print(f"no catalog at {cat}", file=sys.stderr)
        return 1
    raw = json.loads(cat.read_text(encoding="utf-8"))
    cards = raw if isinstance(raw, list) else raw.get("docs", [raw])

    print(f"catalog: {cat}\n")
    touched = 0
    for card in cards:
        changes = fix(card, args.move_energy)
        if changes:
            touched += 1
            print(f"  {card.get('model_id', '?'):26}")
            for c in changes:
                print(f"      + {c}")

    # Report, do not resolve: which of two cards on one repo should survive is
    # a judgement about what the scenarios need, not something to guess.
    active = [c for c in cards if (c.get("status") or "active") == "active"]
    seen: dict[str, list[str]] = {}
    for c in active:
        ref = str(c.get("hf_repo") or (c.get("params") or {}).get("model_path") or "")
        if ref:
            seen.setdefault(ref, []).append(c.get("model_id", "?"))
    dupes = {r: ids for r, ids in seen.items() if len(ids) > 1}

    print()
    if args.write and touched:
        cat.write_text(json.dumps(cards if isinstance(raw, list) else raw, indent=2) + "\n",
                       encoding="utf-8")
        print(f"wrote {touched} card(s) to {cat}")
    elif touched:
        print(f"{touched} card(s) would change; pass --write")
    else:
        print("nothing to change.")

    if dupes:
        print("\nactive cards now sharing weights, decide which to deprecate:",
              file=sys.stderr)
        for ref, ids in dupes.items():
            print(f"  {ref}  <- {', '.join(ids)}", file=sys.stderr)
    stale = [
        c for c in cards
        if c.get("model_id") == "ttm_energy_168_24"
        and (c.get("params") or {}).get("model_path") == ENERGY_OLD
    ]
    if stale and not args.move_energy:
        print(f"\nttm_energy_168_24 still points at {ENERGY_OLD}, which no longer "
              f"exists.\nRerun with --move-energy to repoint it at {ENERGY_NEW}.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
