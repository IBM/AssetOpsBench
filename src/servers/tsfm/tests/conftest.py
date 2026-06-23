"""Test config — keep the suite hermetic.

Production uses CouchDB (TSFM_STORE=couch, like the other AssetOpsBench servers). Tests force the
in-memory backend so every run starts from the curated seeds with no service dependency — the
same skipif-on-missing-service pattern the sibling servers use for their CouchDB integration tests.
"""

import os

os.environ["TSFM_STORE"] = "memory"
