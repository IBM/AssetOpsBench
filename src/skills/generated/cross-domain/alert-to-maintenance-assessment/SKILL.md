---
name: alert-to-maintenance-assessment
description: >-
  Cross-domain assessment that starts from an alert on equipment and
  produces a maintenance recommendation by combining work-order history,
  failure mode analysis, and vibration diagnostics.  Use when user says
  "investigate alert", "alert to work order", "what maintenance is needed
  for this alert", or "assess equipment health after alert".
domain: cross-domain
level: high
required_mcp_tools:
  - analyze_alert_to_failure
  - get_work_orders
  - get_failure_codes
  - get_failure_modes
  - get_failure_mode_sensor_mapping
  - list_vibration_sensors
  - diagnose_vibration
  - get_vibration_data
trigger_keywords:
  - alert
  - maintenance
  - failure
  - work order
  - corrective
  - assessment
  - equipment health
metadata:
  author: AssetOpsBench
  version: 1.0.0
  mcp-server: wo-mcp-server, fmsr-mcp-server, vibration-mcp-server
---

# Alert-to-Maintenance Assessment

## When to Use

When an alert (rule violation) has been raised on a piece of equipment and a
user needs to understand the root cause, assess severity, and decide on the
appropriate maintenance action.  This skill orchestrates across the Work Order,
FMSR, and Vibration servers to produce an integrated assessment.

## Prerequisites

- Work Order MCP server connected (`wo-mcp-server`)
- FMSR MCP server connected (`fmsr-mcp-server`)
- Vibration MCP server connected (`vibration-mcp-server`)
- Equipment ID and alert rule ID are known

## Procedure

### Phase 1 — Alert context (Work Order server)

#### Step 1.1 — Analyse alert-to-failure transitions

```
call analyze_alert_to_failure(equipment_id=EQ_ID, rule_id=RULE_ID)
```

This returns transition probabilities from the alert to downstream failure
codes and average hours to maintenance.

#### Step 1.2 — Review work-order history

```
call get_work_orders(equipment_id=EQ_ID)
```

Check for recent corrective or preventive work orders on the same equipment.

#### Step 1.3 — Get failure code reference

```
call get_failure_codes()
```

Map the failure codes from step 1.1 to human-readable descriptions.

### Phase 2 — Failure mode analysis (FMSR server)

#### Step 2.1 — Get failure modes for the asset

```
call get_failure_modes(asset_name=ASSET_NAME)
```

#### Step 2.2 — Map failure modes to sensors

```
call get_failure_mode_sensor_mapping(asset_name=ASSET_NAME, failure_modes=FM_LIST, sensors=SENSOR_LIST)
```

Identify which sensors are most relevant to the suspected failure modes from
Phase 1.

### Phase 3 — Vibration confirmation (Vibration server)

#### Step 3.1 — Retrieve vibration data for the relevant sensor

```
call list_vibration_sensors(site_name=SITE, asset_id=ASSET_ID)
call get_vibration_data(site_name=SITE, asset_id=ASSET_ID, sensor_name=RELEVANT_SENSOR, start=ALERT_TIME)
```

#### Step 3.2 — Run vibration diagnosis

```
call diagnose_vibration(data_id=DATA_ID, machine_group="group2")
```

Use the ISO 10816 severity and bearing analysis to confirm or refute the
suspected failure mode.

## Decision Logic

- IF `analyze_alert_to_failure` transition probability to a critical failure code
  exceeds 0.5 AND vibration ISO zone is "C" or "D":
  → Recommend **immediate corrective maintenance** with high confidence.

- IF transition probability exceeds 0.3 BUT vibration is in zone "A" or "B":
  → Recommend **scheduled inspection** within 14 days — alert pattern matches
  historical failures but current vibration levels are acceptable.

- IF transition probability is below 0.1 AND vibration is healthy:
  → Classify as **false alarm** or transient event.  No maintenance action.

- IF vibration data is unavailable for the relevant sensor:
  → Recommend a **manual inspection** — cannot confirm condition remotely.

## Expected Outputs

```json
{
  "equipment_id": "string",
  "rule_id": "string",
  "alert_transition_probability": "float",
  "most_likely_failure_code": "string",
  "vibration_iso_zone": "A | B | C | D | unavailable",
  "dominant_fault": "string | none",
  "recommendation": "immediate_corrective | scheduled_inspection | false_alarm | manual_inspection",
  "confidence": "high | medium | low",
  "supporting_evidence": ["string"]
}
```

## Domain References

- FMEA (Failure Mode and Effects Analysis) — IEC 60812
- ISO 10816 — Vibration severity classification
- ISO 14224 — Reliability and maintenance data for equipment
- Alert-to-failure Markov transition model (AssetOpsBench WO server)
