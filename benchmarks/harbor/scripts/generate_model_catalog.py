#!/usr/bin/env python
"""Build the model catalog from the checkpoints actually on disk.

Hand-maintaining the catalog is how it drifts: swap a checkpoint directory and
every card silently points at nothing, which `preload_models.py --check` then
reports as five missing models. Generating from the directories makes the
weights the source of truth, so that failure mode disappears.

    # see what would be written
    uv run python benchmarks/harbor/scripts/generate_model_catalog.py

    # write it
    uv run python benchmarks/harbor/scripts/generate_model_catalog.py --write

Each card's context_length, prediction_length and channel count are read from
the checkpoint's own config.json, never assumed from the directory name, and
every card is validated against the repo schema before anything is written.

Facts the weights cannot supply - domain, description, lineage, where it came
from - go in an optional `meta.json` beside the checkpoint:

    artifacts/tsfm_models/ttm_energy_168_24/meta.json
    {
      "domain": "energy",
      "provenance": "finetuned",
      "base_model_id": "ttm_512_96",
      "source_repo": "EnergyFM/energy-ttm",
      "description": "Fine-tuned on EnergyBench smart-meter data...",
      "tags": ["smart-meter", "load-forecasting"],
      "trained_on": ["EnergyBench"]
    }

Keeping that beside the weights rather than in the catalog means moving a
checkpoint carries its metadata with it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SKTIME_CLASS = "sktime.forecasting.ttm.TinyTimeMixerForecaster"
DEFAULT_ROOTS = [Path("artifacts/tsfm_models"), Path("artifacts/output/tuned_models")]
DEFAULT_OUT = Path("src/couchdb/scenarios_data/shared/tsfm/model_catalog.json")


def geometry_ok(cfg: dict) -> tuple[bool, str]:
    """sktime's TTM wrapper requires context_length / num_patches == patch_length
    == patch_stride. When it does not hold the wrapper rewrites patch_length,
    every tensor shape changes, and a zero-shot load fails. Refuse to emit a
    card for a checkpoint that cannot load."""
    ctx, npatch = cfg.get("context_length"), cfg.get("num_patches")
    plen, pstr = cfg.get("patch_length"), cfg.get("patch_stride")
    if not all(isinstance(v, int) and v > 0 for v in (ctx, npatch, plen, pstr)):
        return False, f"incomplete geometry ctx={ctx} num_patches={npatch} patch_length={plen}"
    size = ctx / npatch
    if size != plen or size != pstr:
        return False, (f"ctx/num_patches={size:g} != patch_length={plen}; "
                       f"sktime would rewrite it to {max(1, int(size))}")
    return True, f"ctx={ctx} num_patches={npatch} patch_length={plen}"


def build_card(ckpt: Path) -> tuple[dict | None, str]:
    cfg = json.loads((ckpt / "config.json").read_text(encoding="utf-8"))
    ok, detail = geometry_ok(cfg)
    if not ok:
        return None, f"SKIP {ckpt.name}: {detail}"

    meta = {}
    meta_path = ckpt / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    ctx = int(cfg["context_length"])
    horizon = int(cfg["prediction_length"])
    channels = int(cfg.get("num_input_channels") or 1)
    path = ckpt.as_posix()
    domain = meta.get("domain", "general")
    provenance = meta.get("provenance", "pretrained")

    card = {
        "model_id": ckpt.name,
        "model_family": "TinyTimeMixer",
        "sktime_class": SKTIME_CLASS,
        "framework": "tinytimemixer",
        "modality": "timeseries",

        "provenance": provenance,
        "created_by": "seed",
        "created_at": meta.get("created_at", "2026-09-27T00:00:00+00:00"),
        "version": str(meta.get("version", 1)),
        "status": meta.get("status", "active"),

        # All four location fields name the same directory. Only
        # params.model_path is read at load time; the others are metadata the
        # agent imitates when it registers its own cards.
        "source": "local_artifact",
        "hf_repo": None,
        "artifact_path": path,
        "model_checkpoint": path,
        # fit_strategy pins the estimator; training_regime pins run_recipe.
        # Without both, TTM's default "minimal" re-tunes the weights on every
        # fit and run_recipe takes the expanding-window refit path.
        "params": {"model_path": path, "fit_strategy": "zero-shot"},
        "training_regime": "zero_shot",

        "task_ids": ["tsfm_forecasting"],
        "context_length": ctx,
        "prediction_length": horizon,
        "domain": domain,
        "frequency": meta.get("frequency", "any"),
        "trained_on": meta.get("trained_on", [domain] if domain != "general"
                               else ["pretraining-corpus"]),
        "tags": sorted({"ttm", "forecasting", "local-artifact",
                        "finetuned" if provenance == "finetuned" else "zero-shot",
                        *( [domain] if domain != "general" else [] ),
                        *meta.get("tags", [])}),
        "description": meta.get(
            "description",
            f"TinyTimeMixer, context {ctx}, horizon {horizon}."
            + (f" Fine-tuned for the {domain} domain." if domain != "general" else "")),
    }
    if meta.get("base_model_id"):
        card["base_model_id"] = meta["base_model_id"]
    if meta.get("source_repo"):
        card["description"] += f" Source: {meta['source_repo']}."

    note = f"  {ckpt.name:22} ctx={ctx:<5} h={horizon:<4} ch={channels}  {detail}"
    if channels > 1:
        note += (f"\n      WARNING {channels} input channels. The tsfm engine calls "
                 "fit(y, fh=fh) with no X, so an exogenous model cannot be driven "
                 "through run_recipe as the code stands.")
    return card, note


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roots", type=Path, nargs="*", default=DEFAULT_ROOTS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--write", action="store_true", help="write the file (default: print)")
    args = p.parse_args()

    ckpts = sorted({c.parent for r in args.roots for c in r.rglob("config.json")})
    if not ckpts:
        print(f"no checkpoints under {[str(r) for r in args.roots]}", file=sys.stderr)
        return 1

    cards, skipped = [], []
    print(f"{len(ckpts)} checkpoint(s):\n")
    for ckpt in ckpts:
        card, note = build_card(ckpt)
        print(note)
        (cards if card else skipped).append(card or ckpt.name)

    # Validate before writing: a catalog that fails the repo schema is worse
    # than no catalog, because it fails at seed time rather than here.
    try:
        sys.path.insert(0, "src")
        from servers.tsfm.core import schemas
        bad = []
        for c in cards:
            try:
                schemas.validate_model(dict(c))
            except Exception as exc:  # noqa: BLE001 - report all, not the first
                bad.append(f"{c['model_id']}: {exc}")
        if bad:
            print("\nschema validation FAILED:", file=sys.stderr)
            for b in bad:
                print(f"  {b}", file=sys.stderr)
            return 1
        print(f"\n  all {len(cards)} card(s) pass schemas.validate_model")
    except ImportError:
        print("\n  (schema validator unavailable; cards not validated)", file=sys.stderr)

    body = json.dumps(cards, indent=2) + "\n"
    if args.write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(body, encoding="utf-8")
        print(f"  wrote {len(cards)} card(s) to {args.out}")
        print("\nNext: uv run python benchmarks/harbor/scripts/preload_models.py --check")
    else:
        print(f"\n--- would write to {args.out} (pass --write) ---\n")
        print(body)
    if skipped:
        print(f"\n{len(skipped)} checkpoint(s) skipped: {', '.join(skipped)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
