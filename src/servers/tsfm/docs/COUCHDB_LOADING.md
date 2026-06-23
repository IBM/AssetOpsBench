# Loading the catalogs into CouchDB

The model & feature catalogs load into CouchDB through the **standard AssetOpsBench loader**
(`src/couchdb/loader.py` + `collections.json` + scenario manifests) — exactly like `iot`, `wo`,
and `vibration`. No TSFM-specific loading code. The server then reads those two databases via its
CouchDB backend (`TSFM_STORE=couch`, the default).

The data files are the curated catalogs, kept with the other shared CouchDB collection data:
`src/couchdb/scenarios_data/shared/tsfm/model_catalog.json` (48 docs) and `feature_catalog.json`
(115 docs) — each doc already carries an `_id` (`model:<model_id>` / `feature:<feature_id>`).
This is the single source of truth; the package no longer ships its own `seeds/` copy, and
`bootstrap.load_seeds` reads from here (override with `$TSFM_SEEDS_DIR`).

## 1. Register the two collections — `src/couchdb/collections.json`
Add (describes how each is keyed + indexed; `_id` is `id_prefix:primary_key`, which already
matches our docs so they're preserved verbatim):

```jsonc
"model_catalog": {
  "format": "json",
  "primary_key": ["model_id"],
  "id_prefix": "model",
  "indexes": [["task_ids"], ["status"], ["model_family"], ["domain"]]
},
"feature_catalog": {
  "format": "json",
  "primary_key": ["feature_id"],
  "id_prefix": "feature",
  "indexes": [["scenario_categories"], ["kind"], ["status"]]
}
```
(The `task_ids` / `scenario_categories` indexes back the `$elemMatch` filters that
`find_models` / `find_features` use.)

## 2. Point a manifest at the data files
Make the catalog JSONs reachable by the loader (it resolves a manifest path against the manifest
folder → its parent → the couchdb dir). Drop copies under the shared corpus, e.g.
`src/couchdb/shared/tsfm/model_catalog.json` and `feature_catalog.json`, then add to
`scenarios_data/default/manifest.json`:

```json
{
  "model_catalog":   "shared/tsfm/model_catalog.json",
  "feature_catalog": "shared/tsfm/feature_catalog.json"
}
```

(db name = the manifest key, so this creates the `model_catalog` and `feature_catalog`
databases.)

## 3. Load
Already wired into the CouchDB container (`couchdb_setup.sh` runs `init_data.py`). Manually:

```bash
COUCHDB_URL=http://localhost:5984 python -m couchdb.init_data          # default manifest
COUCHDB_URL=http://localhost:5984 python -m couchdb.init_data 304      # a scenario's manifest
```

## Per-scenario catalogs (the "models as a variable" requirement)
A scenario folder `scenarios_data/scenario_<id>/manifest.json` can list **only the catalog that
scenario should see** — a subset of models, a different feature set, or none:

```json
{ "iot": "...", "model_catalog": "scenario_42/models_subset.json" }
```
`init_data(42)` then loads exactly that catalog into CouchDB; the TSFM server reads it unchanged.
This is how the catalog varies scenario-to-scenario, with zero server changes.

## What the server does
`make_store()` → `CouchStore` (default) reads `model_catalog` / `feature_catalog`. The discovery
tools (`find_models`, `find_features`, `describe_candidates`, `get_component`) query those DBs;
write-backs (`register_model`, `register_feature`, evolve) persist there. (`fresh_store` also
self-seeds an empty DB from the packaged seeds as a dev convenience; once the loader has run it's
a no-op.)
```
