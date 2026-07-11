#!/usr/bin/env python3
"""Regenerate feature_catalog.json from the consolidated EXTRACTORS registry.

Keeps the 6 transform cards and the existing extractor cards (with their curated descriptions)
untouched, and ADDS a card for every newly-registered extractor (from feature_extraction.REGISTRY)
using its docstring-derived description. Non-breaking: existing names/descriptions are preserved.

Run with the tsfm package importable (e.g. `uv run python -m servers.tsfm.reasoning.regen_feature_catalog`).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..... import feature_selection as FS          # EXTRACTORS = core 111 + feature_extraction.REGISTRY
from ..... import feature_extraction as FE          # DESCRIPTIONS for the new features

CATALOG = (
    Path(__file__).resolve().parents[3]
    / "couchdb/scenarios_data/shared/tsfm/feature_catalog.json"
)


def main() -> None:
    cards = json.loads(CATALOG.read_text())
    transforms = [c for c in cards if c.get("kind") != "extractor"]
    existing = {
        c["extractor_name"]: c
        for c in cards
        if c.get("kind") == "extractor" and c.get("extractor_name")
    }

    out = list(transforms)
    added = 0
    for name in sorted(FS.EXTRACTORS):
        if name in existing:
            out.append(existing[name])                       # keep curated card verbatim
            continue
        desc = getattr(FE, "DESCRIPTIONS", {}).get(name, f"Feature '{name}'.")
        out.append(
            {
                "_id": f"feature:{name}",
                "feature_id": name,
                "kind": "extractor",
                "extractor_name": name,
                "name": name,
                "description": desc,
                "status": "active",
                "metrics": [],
            }
        )
        added += 1

    CATALOG.write_text(json.dumps(out, indent=2) + "\n")
    n_ex = sum(1 for c in out if c.get("kind") == "extractor")
    print(
        f"{CATALOG.name}: {len(transforms)} transforms + {n_ex} extractors "
        f"({added} new, {len(existing)} kept). Registry size = {len(FS.EXTRACTORS)}."
    )


if __name__ == "__main__":
    main()
