# CouchDB Data Layer

Config-driven loading of benchmark data into CouchDB. Every **collection** (work orders,
IoT sensor data, vibration, alerts, events, failure codes, …) is described once in
`collections.json`, and **manifests** bind data files to collections. Each scenario is a
self-contained folder under `scenarios_data/` holding a `manifest.json` (what data to
load) and a `question.txt` (the scenario prompt); a `default/` folder provides the
baseline. One generic loader (`loader.py`) handles all of them: **each manifest key
becomes a CouchDB database of the same name.**

Data files common to many scenarios live once in `scenarios_data/shared/`. No loader code
changes are needed to add data or new collections — only config.

---

## Quick start

```bash
# 1. Start CouchDB — the container boots, then loads the default data automatically
docker compose -f src/couchdb/docker-compose.yaml up -d

# 2. Verify (Fauxton UI: http://localhost:5984/_utils — admin / password)
curl -s -u admin:password http://localhost:5984/_all_dbs
# → ["workorder","iot","vibration","failurecode", ...]

curl -s -u admin:password http://localhost:5984/workorder/wo:MAIN:1000045
```

To (re)load from your host at any time (no container restart needed):

```bash
python3 src/couchdb/init_data.py              # load the default data
python3 src/couchdb/init_data.py 1            # load scenario 1's data
python3 src/couchdb/init_data.py 1 --reset    # load scenario 1's data (clean slate)
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
└── scenarios_data/            scenarios + the shared corpus
    ├── default/               the DEFAULT configuration
    │   ├── manifest.json
    │   └── question.txt
    ├── scenario_<id>/         one self-contained folder per scenario
    │   ├── manifest.json      what to load (paths into shared/ or local files)
    │   └── question.txt       the scenario prompt (read by the benchmark harness)
    └── shared/                data common to many scenarios
        ├── work_order/workorders.csv
        ├── iot/*.json
        └── failure_code/failure_code_sample.csv
```

A scenario folder is fully self-contained: its `manifest.json`, its prompt, and any data
unique to that scenario. `question.txt` is consumed by the benchmark/agent harness, not
by the loader — `init_data.py` only reads `manifest.json`.

---

## How loading works

1. `init_data.py` picks a **manifest**: with no id, the default manifest
   `scenarios_data/default/manifest.json` (overridable via `DEFAULT_MANIFEST`). With an id,
   `scenarios_data/scenario_<id>/manifest.json` (or a legacy flat `scenario_<id>.json`) —
   and if that doesn't exist it **raises** rather than silently using the default. The
   manifest is resolved before any `--reset`, so an unknown id fails without dropping data.
2. For each `key: source` entry in the manifest, `loader.py` looks up the key in
   `collections.json`, parses the source (CSV or JSON), and writes the documents to a
   database named `key` — **dropped and rebuilt from scratch** on every load.
3. Each document gets a deterministic `_id` built from the collection's `primary_key`,
   e.g. `wo:ABC:WO627398` or `fc:C001`. Reloads are therefore idempotent, and duplicate
   rows (same key) collapse to one document.

### How data paths resolve

A relative path in a manifest is resolved in this order, first match wins:

1. The scenario's own folder (`scenarios_data/scenario_<id>/`) — for data unique to a
   scenario, referenced by a bare path like `iot/custom_reading.json`.
2. The scenario folder's **parent** (`scenarios_data/`) — which is why the shared corpus
   is referenced simply as `shared/iot/chiller_6.json`, with no `../` hops.
3. The couchdb dir (`src/couchdb/`) — preserves the behaviour of any legacy flat manifests.

So the convention is: **shared corpus → `shared/<folder>/<file>`; scenario-local data →
a bare path inside the scenario folder.** (A `../shared/...` form also works if you
prefer it.)

---

## Loading data (CLI)

```bash
python3 src/couchdb/init_data.py [SCENARIO_ID] [flags]
```

| Command | Effect |
| --- | --- |
| `init_data.py` | Load the default manifest. |
| `init_data.py 7` | Load `scenarios_data/scenario_7/manifest.json` (raises `FileNotFoundError` if it doesn't exist). |
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

### 1. Change *what* the default loads — `scenarios_data/default/manifest.json`

```json
{
    "workorder":   "shared/work_order/workorders.csv",
    "iot":         ["shared/iot/chiller_6.json", "shared/iot/metro_pump_1.json"],
    "vibration":   "shared/iot/motor_01.json",
    "failurecode": "shared/failure_code/failure_code_sample.csv"
}
```

Each value may be any of (paths resolve against the scenario folder, then its parent,
then `src/couchdb/` — see "How data paths resolve" above):

| Value form | Meaning |
| --- | --- |
| `"shared/.../file.csv"` or `.json` | Load that file from the shared corpus. |
| `"file.json"` (bare) | Load a file that lives in this scenario's own folder. |
| `"shared/<dir>"` | Load every matching file in the directory. |
| `["path1", "path2", …]` | Load several files, concatenated. |
| `[{...}, {...}]` | Inline document objects, written as-is. |
| `"default"` | Shorthand for the corpus folder of this collection (`<SAMPLE_DATA_DIR>/<key>/`). |

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
2. Drop the data in: `scenarios_data/shared/alert/alerts.json` (a JSON array of docs).
3. Reference it in a manifest: `"alert": "shared/alert/alerts.json"`.
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

A scenario is a folder named `scenario_<id>` under `scenarios_data/`:

```
scenarios_data/scenario_7/
├── manifest.json    # same shape as default/manifest.json; list only what this scenario needs
└── question.txt     # the scenario prompt (for the harness; the loader ignores it)
```

The manifest lists only the collections that scenario needs, pulling common files from
`shared/` and any bespoke files from the scenario folder itself. Then, before running it:

```bash
python3 src/couchdb/init_data.py 7 --reset
```

`--reset` guarantees collections *not* listed in the scenario's manifest end up empty
instead of carrying over from the previous load. An id with no `scenario_<id>/` folder
(or flat `scenario_<id>.json`) raises `FileNotFoundError` rather than loading the default,
and that check happens before the reset, so a wrong id won't wipe your databases.

### Private scenarios in another repo

The layout is portable: point `SCENARIOS_DATA_DIR` at a private folder that has the
same shape (`scenario_<id>/manifest.json` + a sibling `shared/`), and everything resolves
the same way:

```bash
SCENARIOS_DATA_DIR=/path/to/private python3 src/couchdb/init_data.py 7 --reset
```

A private scenario referencing `shared/...` reads its own repo's `shared/` (the scenario
folder's parent); data unique to a private scenario goes in that scenario's folder.

---

## Environment variables

| Variable | Default | Used by |
| --- | --- | --- |
| `COUCHDB_URL` | `http://localhost:5984` | loader |
| `COUCHDB_USERNAME` / `COUCHDB_PASSWORD` | `admin` / `password` | loader |
| `COLLECTIONS_CONFIG` | `collections.json` | loader |
| `SCENARIOS_DATA_DIR` | `scenarios_data/` | init_data — point at a private scenarios folder to load from it |
| `DEFAULT_MANIFEST` | `scenarios_data/default/manifest.json` | init_data — the default ("no id") manifest; override to point elsewhere |

---

## Gotchas

- **Database name = manifest key.** Consumers must match: the WO MCP server reads
  `WO_DBNAME=workorder`.
- **Shared paths resolve via the scenario folder's parent.** `shared/...` works because
  `shared/` sits beside the scenario folders under `scenarios_data/`. If you relocate the
  corpus, update the manifests (or use an explicit relative path).
- **The `"default"` keyword reads `SAMPLE_DATA_DIR`.** That loader constant still points at
  the old `sample_data/` location; if you use the `"default"` shorthand, point it at
  `shared/`. Most manifests use explicit `shared/...` paths and don't need it.
- **Duplicate primary keys dedupe.** Two CSV rows with the same `wonum` become one
  document (second is a conflict) — intended idempotency, not data loss.
- **Mango `sort` requires an index.** Query with selectors freely (CouchDB falls back
  to a scan), but a `sort` without a covering index returns `400 no_usable_index`.
  Consumers should sort client-side or add the index to `collections.json`.
- **`--reuse` keeps history; default reload wipes it.** Anything written at runtime
  (e.g. work orders created by agent tools) is deleted on the next default load —
  that's the deterministic-benchmark behavior.