#!/usr/bin/env python3
"""Export/check the code image's locked subset without adding app dependencies.

Run after `uv lock` when changing the scientific dependency group:
    python scripts/export_scientific_requirements.py
    python scripts/export_scientific_requirements.py --check
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "src/agent/stirrup_agent/requirements-scientific.txt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = subprocess.run(
        ["uv", "export", "--locked", "--only-group", "scientific",
         "--no-emit-project", "--no-header", "--format", "requirements-txt"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    # Hashes are intentionally retained for pip --require-hashes in Docker.
    content = (
        "# Generated from uv.lock's scientific group; do not edit manually.\n"
        "# Regenerate: python scripts/export_scientific_requirements.py\n"
        + result.stdout
    )
    if args.check:
        if not DESTINATION.exists() or DESTINATION.read_text() != content:
            print("Scientific image requirements are stale; regenerate with "
                  "python scripts/export_scientific_requirements.py", file=sys.stderr)
            return 1
        print("Scientific image requirements match uv.lock.")
    else:
        DESTINATION.write_text(content)
        print(f"Wrote {DESTINATION.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
