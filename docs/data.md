# CouchDB Data Layer

Config-driven loading of benchmark data into CouchDB. Every **collection** (work orders,
IoT sensor data, vibration, alerts, events, failure codes, …) is described once in
`collections.json`, and **manifests** bind data files to collections — `default.json`
for the baseline, `scenario_<id>.json` for scenario-specific data. One generic loader
(`loader.py`) handles all of them: **each manifest key becomes a CouchDB database of
the same name.**

No loader code changes are needed to add data or new collections — only config.

---

## Quick start

```bash
# 1. Start CouchDB — the container boots, then loads the default data automatically
docker compose -f src/couchdb/docker-compose.yaml up -d

# 2. Verify (Fauxton UI: http://localhost:5984/_utils — admin / password)
curl -s -u admin:password http://localhost:5984/_all_dbs
# → ["workorder","iot","vibration","failurecode", ...]

curl -s -u admin:password http://localhost:5984/workorder/wo:ABC:WO627398
```

To (re)load from your host at any time (no container restart needed):

```bash
pip install requests pandas python-dotenv     # one-time, or use `uv run`
python3 src/couchdb/init_data.py              # load the default data
python3 src/couchdb/init_data.py 7            # load scenario 7's data
python3 src/couchdb/init_data.py --reset-only # drop all user databases (clean slate)
```

---

## Directory layout

```
src/couchdb/
├── docker-compose.yaml        CouchDB container (publishes :5984)
├── couchdb_setup.sh           container entrypoint: boot CouchDB → init_data (default)
│
├── init_data.py               ENTRY POINT: resolve manifest → load collections
├── loader.py                  generic loader (CSV/JSON → docs → CouchDB)
├── collections.json           per-collection config (format, primary key, indexes, …)
├── transforms.py              optional per-collection doc transforms (escape hatch)
├── _design_workorders.json    work-order validation + views (installed on load)
│
├── sample_data/               the data files
│   ├── work_order/workorders.csv
│   ├── iot/*.json
│   └── failure_code/failure_code_sample.csv
│
└── scenarios_data/            manifests (what to load)
    ├── default.json           the DEFAULT configuration
    └── scenario_<id>.json     per-scenario overrides
```

---

## How loading works

1. `init_data.py` picks a **manifest**: `scenario_<id>.json` when an id is given and the
   file exists, otherwise `default.json`.
2. For each `key: source` entry in the manifest, `loader.py` looks up the key in
   `collections.json`, parses the source (CSV or JSON), and writes the documents to a
   database named `key` — **dropped and rebuilt from scratch** on every load.
3. Each document gets a deterministic `_id` built from the collection's
   `primary_key`, e.g. `wo:ABC:WO627398` or `fc:C001`. Reloads are therefore
   idempotent, and duplicate rows (same key) collapse to one document.

---

## Loading data (CLI)

```bash
python3 src/couchdb/init_data.py [SCENARIO_ID] [flags]
```

| Command | Effect |
| --- | --- |
| `init_data.py` | Load the default manifest. |
| `init_data.py 7` | Load `scenarios_data/scenario_7.json` (falls back to default if absent). |
| `init_data.py 7 --reset` | Drop **all** user databases first, then load scenario 7 (guaranteed clean start). |
| `init_data.py --reset-only` | Drop all user databases and exit. |
| `--managed-only` | With `--reset`/`--reset-only`: drop only the default-manifest collections. |
| `--reuse` | Skip the drop; keep existing docs and upsert on top. |

Programmatic use (e.g. from a benchmark harness, before each scenario run):

```python
from couchdb.init_data import init_data, reset
init_data(7, reset_first=True)   # clean slate, then scenario 7's data
init_data()                      # back to default
```

---

## Revising the default configuration

### 1. Change *what* the default loads — `scenarios_data/default.json`

```json
{
    "workorder":   "sample_data/work_order/workorders.csv",
    "iot":         ["sample_data/iot/chiller_6.json", "sample_data/iot/metro_pump_1.json"],
    "vibration":   "sample_data/iot/motor_01.json",
    "failurecode": "sample_data/failure_code/failure_code_sample.csv"
}
```

Each value may be any of (paths are relative to `src/couchdb/`):

| Value form | Meaning |
| --- | --- |
| `"path/to/file.csv"` or `.json` | Load that file. |
| `"path/to/dir"` | Load every matching file in the directory. |
| `["path1", "path2", …]` | Load several files, concatenated. |
| `[{...}, {...}]` | Inline document objects, written as-is. |
| `"default"` | Shorthand for `sample_data/<key>/`. |

Edit this file and re-run `init_data.py` — that's the whole change.

### 2. Change *how* a collection is parsed — `collections.json`

```json
"failurecode": {
    "format": "csv",
    "primary_key": ["code"],
    "id_prefix": "fc",
    "indexes": []
}
```

| Field | Required | Meaning |
| --- | --- | --- |
| `format` | yes | `csv` (pandas; dotted headers like `a.b.c` nest into objects; empty cells are dropped) or `json` (a file holding an array of docs, or one object). |
| `primary_key` | recommended | Fields whose values form the deterministic `_id`: `<id_prefix or key>:<v1>:<v2>`. If a doc is missing a key field, CouchDB assigns a random id (reloads then duplicate — so set this). |
| `id_prefix` | no | `_id` prefix; defaults to the collection key. |
| `design_doc` | no | A design-document JSON to install (validation + views). Only `workorder` uses one. |
| `indexes` | no | Mango indexes to create, one field-list per index. |

### 3. Add a new collection (no code)

Example — adding `alert`:

1. Add a block to `collections.json` (already present for `alert` and `event`):
   ```json
   "alert": { "format": "json", "primary_key": ["alert_id"],
              "indexes": [["equipment_id","start_time"], ["rule_id"]] }
   ```
2. Drop the data in: `sample_data/alert/alerts.json` (a JSON array of docs).
3. Reference it in a manifest: `"alert": "sample_data/alert/alerts.json"`.
4. `python3 src/couchdb/init_data.py` → database `alert` exists, docs keyed `alert:<alert_id>`.

### 4. Custom parsing — `transforms.py` (escape hatch)

CSV cells arrive as strings. If a collection needs typing or reshaping beyond the
generic rules, define a function named after the collection; the loader applies it to
every document before the `_id` is computed:

```python
def workorder(doc):                 # the only transform shipped
    doc.setdefault("type", "workorder")
    if isinstance(doc.get("wopriority"), str):
        doc["wopriority"] = int(float(doc["wopriority"]))
    ...
    return doc
```

Most collections need nothing here.

### 5. Work-order validation — `_design_workorders.json`

The `workorder` database installs a `validate_doc_update` that CouchDB runs on **every
write** (including writes made later by the WO MCP server's tools):

- **Required: `wonum` only.** Everything else is optional.
- Validated **only when present**: `status` ∈ `WAPPR/APPR/WMATL/WSCH/INPRG/WPCOND/COMP/CLOSE/CAN`,
  `worktype` ∈ `CM/PM/EM/PdM/CAL/INSP/GEN`, `wopriority` integer 1–5,
  `description` ≤ 100 chars, `aob_source.agent` / `trigger_type` enums.
- The doc also defines map/reduce views (`by_site`, `by_status`, `backlog`, …).

To change the rules, edit the `validate_doc_update` string. Note: CouchDB requires the
stored functions to be **anonymous** (`function (doc) { … }`) — a named function
(`function foo(doc)`) fails with `compilation_error` on install.

---

## Per-scenario data

A scenario binds to its data purely by **file naming**: create
`scenarios_data/scenario_<id>.json` with the same shape as `default.json`, listing only
the collections that scenario needs. Then, before running the scenario:

```bash
python3 src/couchdb/init_data.py <id> --reset
```

`--reset` guarantees collections *not* listed in the scenario's manifest end up empty
instead of carrying over from the previous load. Scenarios without their own manifest
automatically use the default.

---

## Environment variables

| Variable | Default | Used by |
| --- | --- | --- |
| `COUCHDB_URL` | `http://localhost:5984` | loader |
| `COUCHDB_USERNAME` / `COUCHDB_PASSWORD` | `admin` / `password` | loader |
| `COLLECTIONS_CONFIG` | `collections.json` | loader |
| `WO_SCENARIOS_DATA_DIR` | `scenarios_data/` | init_data |
| `WO_DEFAULT_MANIFEST` | `scenarios_data/default.json` | init_data |

---

## Gotchas

- **Database name = manifest key.** Consumers must match: the WO MCP server reads
  `WO_DBNAME=workorder`.
- **Duplicate primary keys dedupe.** Two CSV rows with the same `wonum` become one
  document (second is a conflict) — intended idempotency, not data loss.
- **Mango `sort` requires an index.** Query with selectors freely (CouchDB falls back
  to a scan), but a `sort` without a covering index returns `400 no_usable_index`.
  Consumers should sort client-side or add the index to `collections.json`.
- **`--reuse` keeps history; default reload wipes it.** Anything written at runtime
  (e.g. work orders created by agent tools) is deleted on the next default load —
  that's the deterministic-benchmark behavior.