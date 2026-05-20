# Smart Grid Data Provenance

*Created: 2026-05-01*

## Overview

The Smart Grid 7th-domain MCP servers in this package operate over **synthetic
data only**. No proprietary or course-restricted data is shipped with the AOB
codebase. Runtime data location is configured via the `SG_DATA_DIR`
environment variable.

The source project for this port is
[`HPML6998-S26-Team13/hpml-assetopsbench-smart-grid-mcp`](https://github.com/HPML6998-S26-Team13/hpml-assetopsbench-smart-grid-mcp).

## What `SG_DATA_DIR` is

`SG_DATA_DIR` is an environment variable pointing at the directory containing
the synthetic Smart Grid CSV datasets the servers read at runtime.

**Default path:** `./data/sg_processed/` relative to the current working
directory (wherever the server process is launched from).

Resolution order in [`src/servers/smart_grid/base.py`](../src/servers/smart_grid/base.py):

1. `SG_DATA_DIR` environment variable — absolute or cwd-relative path.
2. `./data/sg_processed/` relative to cwd (fallback if the variable is unset).

The path is not required to exist at import time; existence is enforced on the
first data-loading call, which raises a clear `FileNotFoundError` with
remediation instructions if the path is missing.

## What's in `SG_DATA_DIR`

Six synthetic CSV files, one per logical data slice:

| File | Server(s) | Description |
|---|---|---|
| `asset_metadata.csv` | IoT | Static nameplate data per transformer (`transformer_id`, `name`, `manufacturer`, `location`, `voltage_class`, `rating_kva`, `install_date`, `age_years`, `health_status`, `fdd_category`, `rul_days`, `in_service`) |
| `sensor_readings.csv` | IoT, TSFM | Time-series sensor readings (load current, winding temp, oil temp, voltage) |
| `failure_modes.csv` | FMSR | Failure mode catalogue with severity, IEC code, and recommended action |
| `dga_records.csv` | FMSR | Dissolved Gas Analysis (DGA) records per transformer, per sample date |
| `rul_labels.csv` | TSFM | Remaining-useful-life labels and health index per transformer |
| `fault_records.csv` | WO | Historical fault and maintenance event records |

All values are synthetic. Gas concentrations in `dga_records.csv` are derived
from the source project's data pipeline; the IEC 60599:2022 Rogers Ratio
fault-table boundaries used for DGA classification are encoded in that
project's `data/knowledge/transformer_standards.json`.

## No-CSV-port policy

The source project's processed CSVs (under `data/processed/`) are **not**
copied into AssetOpsBench. Reasons:

1. **Licensing** — three of the five Kaggle source datasets are CC0; two
   (Transformer Health Index — ODbL, and Current & Voltage Monitoring — author
   copyright) have redistribution restrictions and are treated as local-only in
   the source pipeline. No processed outputs derived from restricted sources are
   ported to AOB, which avoids a licensing audit for upstream reviewers.
2. **Reproducibility** — the synthetic data can be regenerated from
   `data/generate_synthetic.py` in the source project. Any downstream user with
   the generator and the IEC encoding can produce equivalent datasets without
   needing the source project's processed CSVs.
3. **AOB cleanliness** — no course-internal preprocessing outputs in the package
   simplifies upstream review scope.

**For a reviewer or downstream user needing Smart Grid data:**

```bash
git clone https://github.com/HPML6998-S26-Team13/hpml-assetopsbench-smart-grid-mcp.git
cd hpml-assetopsbench-smart-grid-mcp
pip install -r requirements.txt
python data/generate_synthetic.py          # produces data/processed/*.csv
export SG_DATA_DIR=$(pwd)/data/processed
```

## Scenario schema and identifiers

`src/scenarios/local/smart_grid.json` follows the AOB local scenario array
convention: each file is a JSON array and each record has an `id`, `type`,
`text`, `category`, and `characteristic_form`.

The main Smart Grid corpus contains 61 records: the original AOB-style
`AOB-FMSR-001` catalogue probe plus `SGT-001` through `SGT-060` from the
SmartGridBench source project. `SGT-036` through `SGT-050` fill domain-coverage
gaps across FMSR, IoT, TSFM, work-order, and multi-tool paths. `SGT-051`
through `SGT-060` are capability-targeted checks designed to distinguish
calibrated, evidence-grounded agents from models that fabricate data, ignore
tool-result conflicts, or miss strict output constraints.

Smart Grid records also carry evaluator-facing metadata:

| Field | Purpose |
|---|---|
| `asset_id` | Transformer identifier used by the scenario, when applicable. |
| `benchmark_design` | Optional capability-targeting metadata for discrimination-focused scenarios. |
| `difficulty` | Coarse difficulty label (`easy`, `medium`, or `hard`). |
| `domain_tags` | Smart Grid domains exercised by the prompt. |
| `expected_tools` | Intended tool path, using `iot.*`, `fmsr.*`, `tsfm.*`, and `wo.*` names. |
| `ground_truth` | Lightweight grading hints such as required concepts, thresholds, or intermediate values. |
| `ground_truth.must_NOT_include` | Optional negative rubric items that should not appear in a correct answer. |

These extended fields are advisory metadata for evaluators and are safe for
scenario consumers to ignore if they only need the core AOB prompt fields.

Identifier prefixes are intentional:

- `AOB-FMSR-*` records are domain-level catalogue probes that do not depend on a
  specific synthetic transformer.
- `SGT-*` records are transformer-grounded Smart Grid task scenarios.
- `SG-NEG-*` records are negative fixtures used to test validation behavior, not
  main benchmark prompts.

## Source datasets

The source pipeline draws from five Kaggle datasets; licensing varies:

| Dataset | License | Domain servers |
|---|---|---|
| Power Transformers FDD & RUL | CC0 | IoT, TSFM |
| DGA Fault Classification | CC0 | FMSR |
| Smart Grid Fault Records | CC0 | WO |
| Transformer Health Index | ODbL (redistribution restricted; local-only) | FMSR (supplemental) |
| Current & Voltage Monitoring | Author copyright (redistribution restricted; local-only) | IoT, TSFM (supplemental) |

Dataset licensing details and row counts are documented in
`docs/hpml_datasets.pdf` in the source project.

## IEC / IEEE standards encoding

DGA-related ground truth (fault codes, condition tiers, gas thresholds) is
encoded in `data/knowledge/transformer_standards.json` in the source project.
That artifact reflects:

- **IEC 60599:2022** (4th ed., publication 66491) — Rogers Ratio method,
  fault-table boundaries, representative gas profiles
- **IEEE C57.104-2019** — condition framework (C1–C4) and gas threshold values

The FMSR server's `analyze_dga` tool implements the Rogers Ratio method
using the fault-table boundaries from that artifact. Note: the AOB fork
server encodes the table directly in `src/servers/smart_grid/fmsr/main.py`
rather than reading the JSON at runtime. Downstream users regenerating DGA
records should verify that generated gas values round-trip correctly through
`analyze_dga` for their intended fault labels before using them as benchmark
ground truth.

## Citation

SmartGridBench: A Smart Grid transformer maintenance benchmark for MCP-enabled
LLM agents. Columbia University, 2026.
*Citation will be updated when the NeurIPS 2026 Datasets & Benchmarks
submission is finalized.*
