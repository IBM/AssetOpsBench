"""Tool-level JSON-safety smoke for all Smart Grid MCP servers.

Every ``@mcp.tool()``-decorated callable returns over MCP's JSON-RPC
transport, so its output must serialize under
``json.dumps(..., allow_nan=False)``. This test exercises every tool against
a hermetic ``SG_DATA_DIR`` fixture and asserts strict JSON serialization
succeeds. Catches the class of bug fixed in ``fmsr.get_dga_record`` (pandas
``Timestamp`` leaking through ``to_dict()``) across all current and future
Smart Grid tools.
"""

from __future__ import annotations

import json

import pytest

from servers.smart_grid.fmsr import main as fmsr
from servers.smart_grid.iot import main as iot
from servers.smart_grid.tsfm import main as tsfm
from servers.smart_grid.wo import main as wo

_ASSET_METADATA_CSV = "\n".join(
    [
        "transformer_id,name,manufacturer,location,voltage_class,rating_kva,install_date,age_years,health_status,fdd_category,rul_days,in_service",
        "T-001,Unit 1,Acme,Site A,138kV,50000,2018-03-15,8,healthy,normal,2400,True",
        "T-002,Unit 2,Acme,Site B,138kV,50000,2017-06-01,9,degraded,attention,1200,True",
    ]
)

_SENSOR_READINGS_CSV = "\n".join(
    [
        "transformer_id,timestamp,sensor_id,value,unit,source",
        "T-001,2026-01-01T00:00:00,winding_temp_c,55.2,celsius,sim",
        "T-001,2026-01-02T00:00:00,winding_temp_c,56.1,celsius,sim",
        "T-001,2026-01-03T00:00:00,winding_temp_c,57.0,celsius,sim",
    ]
)

_FAILURE_MODES_CSV = "\n".join(
    [
        "failure_mode_id,name,dga_label,description,severity,iec_code,key_gases,recommended_action",
        "FM-001,Thermal Fault T1,T1,Low temperature thermal fault,low,IEC-60599-T1,CH4|C2H6,monitor",
        "FM-002,Arc Discharge,D2,High energy arc discharge,critical,IEC-60599-D2,C2H2|H2,immediate inspection",
    ]
)

_DGA_RECORDS_CSV = "\n".join(
    [
        "transformer_id,sample_date,dissolved_h2_ppm,dissolved_ch4_ppm,dissolved_c2h2_ppm,dissolved_c2h4_ppm,dissolved_c2h6_ppm,dissolved_co_ppm,dissolved_co2_ppm,fault_label,source_dataset",
        "T-001,2026-01-02,10,20,1,30,40,100,200,T1,unit-test",
    ]
)

_RUL_LABELS_CSV = "\n".join(
    [
        "transformer_id,timestamp,rul_days,health_index,fdd_category",
        "T-001,2026-01-01T00:00:00,2400,0.92,0",
        "T-001,2026-01-15T00:00:00,2390,0.91,0",
    ]
)

_FAULT_RECORDS_CSV = "\n".join(
    [
        "transformer_id,fault_id,fault_type,location,voltage_v,current_a,power_load_mw,temperature_c,wind_speed_kmh,weather_condition,maintenance_status,component_health,duration_hrs,downtime_hrs",
        "T-001,F001,Thermal Fault,Site A,138000,200,30,55,5,clear,Pending,degraded,4,2",
    ]
)


@pytest.fixture
def sg_data_dir(tmp_path, monkeypatch):
    """Hermetic SG_DATA_DIR with minimal CSVs + module cache reset across all four servers."""
    (tmp_path / "asset_metadata.csv").write_text(_ASSET_METADATA_CSV, encoding="utf-8")
    (tmp_path / "sensor_readings.csv").write_text(
        _SENSOR_READINGS_CSV, encoding="utf-8"
    )
    (tmp_path / "failure_modes.csv").write_text(_FAILURE_MODES_CSV, encoding="utf-8")
    (tmp_path / "dga_records.csv").write_text(_DGA_RECORDS_CSV, encoding="utf-8")
    (tmp_path / "rul_labels.csv").write_text(_RUL_LABELS_CSV, encoding="utf-8")
    (tmp_path / "fault_records.csv").write_text(_FAULT_RECORDS_CSV, encoding="utf-8")
    monkeypatch.setenv("SG_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(iot, "_metadata", None)
    monkeypatch.setattr(iot, "_readings", None)
    monkeypatch.setattr(fmsr, "_failure_modes", None)
    monkeypatch.setattr(fmsr, "_dga_records", None)
    monkeypatch.setattr(tsfm, "_rul", None)
    monkeypatch.setattr(tsfm, "_readings", None)
    monkeypatch.setattr(wo, "_fault_records", None)
    monkeypatch.setattr(wo, "_asset_metadata", None)
    return tmp_path


@pytest.mark.parametrize(
    "label, call",
    [
        ("iot.list_assets", lambda: iot.list_assets()),
        ("iot.get_asset_metadata", lambda: iot.get_asset_metadata("T-001")),
        ("iot.list_sensors", lambda: iot.list_sensors("T-001")),
        (
            "iot.get_sensor_readings",
            lambda: iot.get_sensor_readings("T-001", "winding_temp_c", limit=10),
        ),
        ("fmsr.list_failure_modes", lambda: fmsr.list_failure_modes()),
        ("fmsr.search_failure_modes", lambda: fmsr.search_failure_modes("thermal")),
        (
            "fmsr.get_sensor_correlation",
            lambda: fmsr.get_sensor_correlation("FM-001"),
        ),
        ("fmsr.get_dga_record", lambda: fmsr.get_dga_record("T-001")),
        (
            "fmsr.analyze_dga",
            lambda: fmsr.analyze_dga(
                h2=10, ch4=20, c2h2=1, c2h4=30, c2h6=40, transformer_id="T-001"
            ),
        ),
        ("tsfm.get_rul", lambda: tsfm.get_rul("T-001")),
        (
            "tsfm.forecast_rul",
            lambda: tsfm.forecast_rul("T-001", horizon_days=30),
        ),
        (
            "tsfm.detect_anomalies",
            lambda: tsfm.detect_anomalies("T-001", "winding_temp_c"),
        ),
        (
            "tsfm.trend_analysis",
            lambda: tsfm.trend_analysis("T-001", "winding_temp_c"),
        ),
        ("wo.list_fault_records", lambda: wo.list_fault_records(limit=5)),
        ("wo.get_fault_record", lambda: wo.get_fault_record("F001")),
        (
            "wo.create_work_order",
            lambda: wo.create_work_order("T-001", "test issue"),
        ),
    ],
)
def test_tool_output_is_strict_json_safe(sg_data_dir, label, call):
    """Every Smart Grid tool's return value must pass strict JSON serialization.

    ``json.dumps(..., allow_nan=False)`` is the contract the FastMCP JSON-RPC
    transport enforces on tool responses. This test catches the class of bug
    fixed in ``fmsr.get_dga_record`` (pandas ``Timestamp`` leaking through
    ``to_dict()``) for any tool, current or future.
    """
    result = call()
    json.dumps(result, allow_nan=False)
