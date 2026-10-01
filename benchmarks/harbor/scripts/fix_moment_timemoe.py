#!/usr/bin/env python
"""Repair the MOMENT and TimeMoE cards in a model catalog.

MOMENT. MOMENTForecaster pairs two defaults:

    pretrained_model_name_or_path="AutonLab/MOMENT-1-large"
    transformer_backbone="google/flan-t5-large"

The backbone decides d_model, so overriding only the first loads -base or
-small weights into a large encoder and every tensor is the wrong width. That
is the "size mismatch ... for MOMENTPipeline" error. Each checkpoint needs its
matching backbone from SUPPORTED_HUGGINGFACE_MODELS.

TimeMoE. The wrapper declares transformers<=4.40.1, but the default path uses
sktime's own vendored copy (sktime.libs.timemoe), so the pin guards code sktime
ships. `ignore_deps=True` clears the check. The vendored model class imports
cleanly on transformers 5.x; generation is the untested part.

    uv run python benchmarks/harbor/scripts/fix_moment_timemoe.py
    uv run python benchmarks/harbor/scripts/fix_moment_timemoe.py --write

Dry run by default: it prints the exact edits and changes nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DEFAULT_CATALOG = Path("src/couchdb/scenarios_data/shared/tsfm/model_catalog.json")

# From sktime/libs/momentfm/models/moment.py: SUPPORTED_HUGGINGFACE_MODELS
BACKBONES = {
    "small": "google/flan-t5-small",
    "base": "google/flan-t5-base",
    "large": "google/flan-t5-large",
    "xl": "google/flan-t5-xl",
    "xxl": "google/flan-t5-xxl",
}


def catalog_path(explicit: Path | None) -> Path:
    if explicit:
        return explicit
    env = os.environ.get("AOB_MODEL_CATALOG")
    if env:
        return Path(env)
    root = os.environ.get("SCENARIOS_DATA_DIR")
    if root:
        return Path(root) / "shared/tsfm/model_catalog.json"
    return DEFAULT_CATALOG


def repo_of(card: dict) -> str:
    return str(card.get("hf_repo") or (card.get("params") or {}).get("model_path") or "")


def fix_moment(card: dict) -> list[str]:
    repo = repo_of(card).split("@")[0]
    size = repo.rsplit("-", 1)[-1].lower()
    backbone = BACKBONES.get(size)
    if not backbone:
        return [f"cannot infer backbone from {repo!r}; set transformer_backbone by hand"]

    params = card.setdefault("params", {})
    changes = []
    if params.get("pretrained_model_name_or_path") != repo:
        params["pretrained_model_name_or_path"] = repo
        changes.append(f'pretrained_model_name_or_path="{repo}"')
    if params.get("transformer_backbone") != backbone:
        params["transformer_backbone"] = backbone
        changes.append(f'transformer_backbone="{backbone}"')
    # The forecasting head is built fresh from forecast_horizon and
    # freeze_head defaults to False, so MOMENT always trains it. Record that
    # rather than letting a zero-shot benchmark quietly include a tuned model.
    # MomentPytorchDataset sizes its windows as
    #     n_timestamps - seq_len - fh + 1
    # and the validation split gets train_val_split of the series. With a 2800
    # point history that is 560 points, so 560 - 512 - 96 + 1 = -47 and the
    # DataLoader raises "__len__() should return >= 0". sktime's own test params
    # use train_val_split=0.0, which skips the validation dataset entirely
    # (guarded by `if not y_test.empty`). For a benchmark run there is nothing
    # to early-stop on, so 0.0 is both the simplest and the deterministic choice.
    if params.get("train_val_split") != 0.0:
        params["train_val_split"] = 0.0
        changes.append("train_val_split=0.0 (avoids a negative-length validation "
                       "window: 560 - 512 - 96 + 1 = -47)")

    if card.get("training_regime") != "fine_tune":
        card["training_regime"] = "fine_tune"
        changes.append('training_regime="fine_tune" (head starts random; it must train)')
    return changes


def fix_timemoe(card: dict) -> list[str]:
    params = card.setdefault("params", {})
    if params.get("ignore_deps") is True:
        return []
    params["ignore_deps"] = True
    return [("ignore_deps=true (bypasses the transformers<=4.40.1 pin on the "
             "vendored implementation)")]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--catalog", type=Path, default=None)
    p.add_argument("--write", action="store_true", help="apply the edits (default: print)")
    args = p.parse_args()

    cat = catalog_path(args.catalog)
    if not cat.is_file():
        print(f"no catalog at {cat}", file=sys.stderr)
        return 1
    raw = json.loads(cat.read_text(encoding="utf-8"))
    listed = isinstance(raw, list)
    cards = raw if listed else raw.get("docs", [raw])

    print(f"catalog: {cat}\n")
    touched = 0
    for c in cards:
        repo = repo_of(c).lower()
        if "moment" in repo:
            changes = fix_moment(c)
        elif "timemoe" in repo or "time-moe" in repo:
            changes = fix_timemoe(c)
        else:
            continue
        if not changes:
            print(f"  {c.get('model_id', '?'):32} already correct")
            continue
        touched += 1
        print(f"  {c.get('model_id', '?'):32}")
        for ch in changes:
            print(f"      + {ch}")

    print()
    if not touched:
        print("nothing to change.")
        return 0
    if args.write:
        cat.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {touched} card(s) to {cat}")
        print("\nNext: uv run python benchmarks/harbor/scripts/smoke_forecast.py")
    else:
        print(f"{touched} card(s) would change; pass --write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
