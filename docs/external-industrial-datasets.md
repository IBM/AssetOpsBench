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
| [Paderborn University Bearing Dataset](https://groups.uni-paderborn.de/kat/BearingDataCenter/) | Rolling-bearing fault experiments | `vibration`, `tsfm` | Useful to cross-check bearing-fault robustness across machines/loads. |

## Mapping checklist to AssetOpsBench schema

When preparing scenarios from an external source, define these fields early:

- `asset_id` and `site_name` strategy (stable IDs, no ambiguous aliases)
- timestamp normalization (timezone, granularity, ISO format)
- sensor naming map (raw column names to scenario-facing names)
- expected outputs in `characteristic_form` that remain auditable from the data
- task-domain classification (`iot`, `wo`, `vibration`, `tsfm`, `fmsr`, multi-step)

## Concrete starter mappings

The mappings below are documentation-only starting points. They do not imply
that the raw datasets are redistributed in this repository or that executable
benchmark scenarios already exist for each source.

### Vibration diagnostics

Public bearing datasets are a strong fit for the existing `vibration` domain.
AssetOpsBench already includes a
[vibration MCP server](mcp-servers.md#vibration--vibration-diagnostics) with
FFT analysis, envelope analysis, bearing characteristic frequency calculation,
ISO 10816 severity assessment, and full vibration diagnosis capabilities.
Existing local utterances in `src/scenarios/local/vibration_utterance.json` can
be used as a style reference for future scenario PRs.

| Source | Asset class | Candidate AssetOpsBench domain | Candidate scenario shape | Notes |
| --- | --- | --- | --- | --- |
| [Case Western Reserve Bearing Data Center](https://engineering.case.edu/bearingdatacenter/welcome) | Bearings, rotating machinery, motors | `vibration` | Fault classification from FFT/envelope evidence; bearing-frequency reasoning; maintenance prioritization after suspected bearing fault | Verify dataset terms and citation requirements before deriving fixtures. |
| [Paderborn University Bearing Dataset](https://groups.uni-paderborn.de/kat/BearingDataCenter/) | Rolling bearings under varying operating conditions | `vibration`, optional `tsfm` | Cross-load bearing diagnosis; robustness checks across machine/load conditions; time-series condition comparison | Useful follow-up source once the first bearing mapping is agreed. |

Candidate vibration prompts for a future executable scenario PR:

- Diagnose whether a motor vibration signal suggests an outer race bearing fault
  using envelope-spectrum evidence.
- Calculate BPFO, BPFI, BSF, and FTF for a bearing geometry and shaft speed,
  then explain which observed peaks match the expected fault frequencies.
- Compare two bearing signals and prioritize maintenance based on spectral
  evidence and severity.
- Explain whether dominant FFT peaks are more consistent with unbalance,
  misalignment, looseness, or a bearing defect.

### SWaT / water-treatment telemetry

SWaT is a useful starting point for water-treatment anomaly scenarios. It maps
most naturally to `iot` for sensor and actuator history, and to `tsfm` for
forecasting or anomaly-detection tasks. More advanced scenarios may combine
sensor lookup, time-series analysis, process-stage interpretation, and operator
recommendations.

| Source | Asset class | Candidate AssetOpsBench domain | Candidate scenario shape | Notes |
| --- | --- | --- | --- | --- |
| [SWaT (Secure Water Treatment)](https://www.kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system) | Water-treatment process, sensors, actuators | `iot`, `tsfm`, multi-step | Retrieve process telemetry for a time window; identify abnormal process state; forecast threshold breach; explain affected treatment stage | Verify source terms before use. Do not commit raw Kaggle data unless redistribution is explicitly allowed. |

Candidate SWaT prompts for a future executable scenario PR:

- Retrieve sensor readings for a water-treatment stage over a specific time
  window and summarize abnormal behavior.
- Forecast whether a tank-level, flow, or pressure variable is likely to breach
  an operating threshold in the next window.
- Determine whether anomalous readings are consistent with a process fault or a
  cyber-physical attack pattern.
- Recommend operator checks after an anomaly is detected in a treatment stage,
  citing the sensor evidence used.

## From mapping to executable scenarios

Before turning either mapping into benchmark scenarios, a follow-up PR should
make the source-to-scenario contract explicit:

1. Confirm license, citation, and redistribution constraints.
2. Keep raw data outside the repository unless redistribution is allowed.
3. Define stable `asset_id` and `site_name` values.
4. Normalize timestamps to a documented ISO 8601 convention.
5. Map raw sensor or actuator columns to scenario-facing names.
6. Document the transform script input/output contract and provenance metadata.
7. Add candidate utterances following `docs/guideline/utterance_design_guideline.md`.
8. Define expected behavior and ground-truth criteria following `docs/guideline/ground_truth_design_guideline.md`.
9. Validate scenario files against the evaluation expectations in `docs/evaluation.md`.

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
