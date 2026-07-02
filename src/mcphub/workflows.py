"""
Composed workflows — plain functions that chain tool calls.

Each takes a ``ToolUniverse`` instance and returns a dict. Names listed in
``REGISTERED`` are auto-registered so they run through the same entrypoint::

    tu.run({"name": "chiller_triage", "arguments": {"asset_id": "Chiller 6"}})

To add one: write a function ``fn(tu, **arguments)`` and add its name to
``REGISTERED``.
"""

REGISTERED = ["chiller_triage", "sensor_inventory_gap"]


def chiller_triage(tu, asset_id, site="MAIN", raise_work_order=True, priority="2"):
    """Sensors -> failure modes -> mode/sensor mapping -> (optional) work order."""
    sensors = tu.run("iot.sensors", {"site_name": site, "asset_id": asset_id})
    failure_modes = tu.run("fmsr.get_failure_modes", {"asset_name": asset_id})
    mapping = tu.run("fmsr.get_failure_mode_sensor_mapping", {
        "asset_name": asset_id,
        "failure_modes": failure_modes,
        "sensors": sensors,
    })

    out = {
        "asset_id": asset_id,
        "site_name": site,
        "sensors": sensors,
        "failure_modes": failure_modes,
        "failure_mode_sensor_mapping": mapping,
    }
    if raise_work_order:
        out["work_order"] = tu.run("wo.generate_work_order", {
            "description": f"Investigate potential failure modes on {asset_id}",
            "asset_num": asset_id,
            "site_id": site,
            "priority": priority,
            "aob_source": "mcphub:chiller_triage",
        })
    return out


def sensor_inventory_gap(tu, asset_id, site="MAIN"):
    """Installed (registry) vs measured (telemetry) sensors for an asset."""
    installed = _names(tu.run("iot.asset_sensors",
                              {"site_name": site, "asset_id": asset_id}))
    measured = _names(tu.run("iot.sensors",
                             {"site_name": site, "asset_id": asset_id}))
    return {
        "asset_id": asset_id,
        "site_name": site,
        "installed_not_streaming": sorted(set(installed) - set(measured)),
        "streaming_not_in_registry": sorted(set(measured) - set(installed)),
    }


def _names(value):
    if isinstance(value, dict):
        for key in ("sensors", "names", "data", "items"):
            if key in value:
                return _names(value[key])
        return list(value.keys())
    if isinstance(value, list):
        return [v if isinstance(v, str) else v.get("name", str(v)) for v in value]
    return [value]