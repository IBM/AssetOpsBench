"""config.py — environment + shared constants (single source of truth).

Centralizes the few env knobs and collection names so modules don't hard-code them. Existing
modules keep working unchanged; new code should read settings from here.
"""

from __future__ import annotations

import os

# storage backend — CouchDB by default, exactly like the other AssetOpsBench servers
STORE_BACKEND = os.environ.get("TSFM_STORE", "couch")           # "couch" (default) | "memory" (tests)
COUCH_URL = os.environ.get("COUCHDB_URL", "http://localhost:5984")
COUCH_AUTH = os.environ.get("TSFM_COUCH_AUTH")                   # "user:password" | None

# file-pointer working dir (IoT CSVs the agent passes in/out)
WORKDIR = os.environ.get("TSFM_WORKDIR", "/tmp/tsfm_work")

# run/plan ledgers
RUNS_COLLECTION = "tsfm_runs"
PLANS_COLLECTION = "tsfm_plans"
