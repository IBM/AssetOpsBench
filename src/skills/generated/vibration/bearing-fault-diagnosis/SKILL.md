---
name: bearing-fault-diagnosis
description: >-
  End-to-end bearing fault diagnosis from raw vibration data.
  Use when user asks to "diagnose vibration", "check bearing health",
  "analyse bearing fault", or "investigate vibration alarm".
  Covers sensor listing, FFT spectrum, envelope spectrum, bearing
  frequency calculation, ISO 10816 severity, and full diagnosis report.
domain: vibration
level: mid
required_mcp_tools:
  - list_vibration_sensors
  - get_vibration_data
  - compute_fft_spectrum
  - compute_envelope_spectrum
  - calculate_bearing_frequencies
  - assess_vibration_severity
  - diagnose_vibration
trigger_keywords:
  - vibration
  - bearing
  - fault
  - diagnosis
  - FFT
  - envelope
  - ISO 10816
metadata:
  author: AssetOpsBench
  version: 1.0.0
  mcp-server: vibration-mcp-server
---

# Bearing Fault Diagnosis

## When to Use

When a vibration alarm is raised on rotating machinery, or a user asks to
diagnose bearing health for a specific asset.  This skill applies to single-asset
vibration analysis with known or discoverable bearing geometry.

## Prerequisites

- Vibration MCP server connected (`vibration-mcp-server`)
- Site name and asset ID are known or can be obtained from the user
- Bearing designation or geometry (number of balls, ball diameter,
  pitch diameter) is known or can be looked up via `list_known_bearings`

## Procedure

### Step 1 — Discover sensors

```
call list_vibration_sensors(site_name=SITE, asset_id=ASSET)
```

Select the sensor closest to the bearing of interest.

### Step 2 — Retrieve raw vibration data

```
call get_vibration_data(site_name=SITE, asset_id=ASSET, sensor_name=SENSOR, start=START_ISO)
```

Record the returned `data_id` — it is required for all subsequent analysis calls.

### Step 3 — Compute FFT spectrum

```
call compute_fft_spectrum(data_id=DATA_ID)
```

Inspect the top peaks for dominant frequencies.

### Step 4 — Compute envelope spectrum

```
call compute_envelope_spectrum(data_id=DATA_ID)
```

Envelope analysis highlights repetitive impact signatures typical of bearing
defects (inner race, outer race, rolling element).

### Step 5 — Calculate bearing characteristic frequencies

If bearing geometry is known:

```
call calculate_bearing_frequencies(rpm=RPM, n_balls=N, ball_diameter_mm=D_BALL, pitch_diameter_mm=D_PITCH)
```

If not, look up the designation:

```
call list_known_bearings()
```

### Step 6 — Run full diagnosis

```
call diagnose_vibration(data_id=DATA_ID, rpm=RPM, bearing_designation=DESIGNATION, machine_group="group2")
```

This produces a comprehensive report including ISO 10816 severity zone,
shaft features, bearing analysis, and a Markdown diagnostic report.

## Decision Logic

- IF `iso_10816.zone` is "D" (danger) THEN recommend immediate shutdown and inspection.
- IF `iso_10816.zone` is "C" (alert) THEN schedule corrective maintenance within 7 days.
- IF bearing analysis shows BPFO or BPFI peaks above 3× noise floor THEN flag bearing defect.
- IF no bearing frequencies are prominent AND RMS velocity is low THEN report healthy.

## Expected Outputs

```json
{
  "asset_id": "string",
  "sensor_name": "string",
  "iso_severity_zone": "A | B | C | D",
  "dominant_fault": "outer_race | inner_race | rolling_element | cage | none",
  "recommendation": "string",
  "report_markdown": "string"
}
```

## Domain References

- ISO 10816 — Mechanical vibration severity classification
- BPFO / BPFI / BSF / FTF — bearing defect frequency formulas
- SKF bearing designation catalog
