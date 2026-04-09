---
name: asset-sensor-inventory
description: >-
  Retrieve the full sensor inventory for an IoT-monitored asset.
  Use when user asks "list sensors", "what sensors does this asset have",
  "show me the sensor inventory", or "check available IoT data".
  Returns site, asset, and sensor metadata in a single call.
domain: iot
level: low
required_mcp_tools:
  - sites
  - assets
  - sensors
trigger_keywords:
  - sensor
  - inventory
  - IoT
  - asset
  - list sensors
metadata:
  author: AssetOpsBench
  version: 1.0.0
  mcp-server: iot-mcp-server
---

# Asset Sensor Inventory

## When to Use

When a user needs to know which sensors are available for a given asset, or
wants an overview of the IoT infrastructure at a site.  This is typically the
first step before querying sensor history or running anomaly detection.

## Prerequisites

- IoT MCP server connected (`iot-mcp-server`)
- Site name is known; if not, discover it first

## Procedure

### Step 1 — Discover available sites

```
call sites()
```

If only one site is returned, use it.  Otherwise ask the user which site.

### Step 2 — List assets at the site

```
call assets(site_name=SITE)
```

Present the asset list to the user or use the asset ID they already provided.

### Step 3 — List sensors for the target asset

```
call sensors(site_name=SITE, asset_id=ASSET)
```

Return the full sensor list including names and types.

## Decision Logic

- IF `sites()` returns no sites THEN report that no IoT data is available.
- IF `assets(site_name)` returns no assets THEN report that the site has no
  monitored assets.
- IF `sensors()` returns an empty list THEN report that the asset exists but
  has no sensors configured.

## Expected Outputs

```json
{
  "site_name": "string",
  "asset_id": "string",
  "total_sensors": "integer",
  "sensors": [
    { "name": "string", "type": "string" }
  ]
}
```

## Domain References

- ISA-95 asset hierarchy (site → area → asset → sensor)
