"""config.py — environment + shared constants (single source of truth).

Centralizes the few env knobs and collection names so modules don't hard-code them. Existing
modules keep working unchanged; new code should read settings from here.
"""

from __future__ import annotations

# run/plan ledgers
RUNS_COLLECTION = "tsfm_runs"
PLANS_COLLECTION = "tsfm_plans"
