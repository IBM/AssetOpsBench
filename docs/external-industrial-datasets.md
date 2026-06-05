# External Industrial Dataset Guide

This page helps contributors discover public industrial datasets that may be useful for scenario design and benchmark extensions.

It does **not** change AssetOpsBench scoring or baseline definitions. It is a reference map for dataset discovery and adaptation planning.

## Quick selection criteria

Before using any external dataset:

1. Verify license terms and redistribution constraints.
2. Confirm no sensitive telemetry, secrets, or personally identifying data are included.
3. Keep provenance metadata (`source`, `version/date`, `transform script`) with every derived artifact.
4. Prefer datasets that can be mapped to one or more existing AssetOpsBench domains (`iot`, `wo`, `vibration`, `tsfm`, `fmsr`).

## Starter dataset references

| Dataset / index | Primary focus | AssetOpsBench fit | Notes |
| --- | --- | --- | --- |
| [awesome-industrial-datasets](https://github.com/jonathanwvd/awesome-industrial-datasets/tree/master) | Curated index (multiple domains) | Discovery for all domains | Useful first stop; each linked dataset has its own license and usage terms. |
| [SWaT (Secure Water Treatment)](https://www.kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system) | Water-treatment process telemetry, attack/anomaly traces | `iot`, `tsfm`, anomaly scenarios | Commonly used for anomaly-detection tasks; verify host terms and citation requirements. |
| [NASA C-MAPSS](https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq) | Turbofan degradation / RUL | `tsfm`, prognostics scenarios | Good candidate for PHM and RUL-style benchmark tasks. |
| [Case Western Reserve Bearing Data Center](https://engineering.case.edu/bearingdatacenter/welcome) | Bearing vibration fault data | `vibration` | Strong fit for spectral diagnosis and fault-classification tasks. |
| [Paderborn University Bearing Dataset](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter) | Rolling-bearing fault experiments | `vibration`, `tsfm` | Useful to cross-check bearing-fault robustness across machines/loads. |

## Mapping checklist to AssetOpsBench schema

When preparing scenarios from an external source, define these fields early:

- `asset_id` and `site_name` strategy (stable IDs, no ambiguous aliases)
- timestamp normalization (timezone, granularity, ISO format)
- sensor naming map (raw column names to scenario-facing names)
- expected outputs in `characteristic_form` that remain auditable from the data
- task-domain classification (`iot`, `wo`, `vibration`, `tsfm`, `fmsr`, multi-step)

## Suggested ingestion workflow

1. Keep raw source data outside committed benchmark artifacts unless license allows redistribution.
2. Build a deterministic transform script with clear input/output contracts.
3. Store transformed fixtures under domain-specific folders with a compact README.
4. Add unit checks for schema and timestamp consistency before creating scenarios.
5. Open a PR with:
   - data provenance note,
   - sample scenario IDs,
   - before/after validation evidence.

## Privacy and safety guardrails

- Remove direct identifiers, facility names, and any customer-linked metadata.
- Never include production secrets, API keys, or internal endpoint information.
- If uncertainty exists, treat the dataset as restricted until maintainers confirm usage policy.

## Related contribution entry points

- Main scenario contribution section: `README.md` -> "Call for Scenario Contribution"
- Scenario design guidelines:
  - `docs/guideline/utterance_design_guideline.md`
  - `docs/guideline/ground_truth_design_guideline.md`
