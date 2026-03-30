import json
import os
from unittest.mock import patch

import pytest
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

# --- Custom markers ---


def _couchdb_reachable() -> bool:
    url = os.environ.get("COUCHDB_URL")
    if not url:
        return False
    try:
        import requests
        requests.get(url, timeout=2)
        return True
    except Exception:
        return False


requires_couchdb = pytest.mark.skipif(
    not _couchdb_reachable(),
    reason="CouchDB not reachable (set COUCHDB_URL and ensure CouchDB is running)",
)


# --- Fixture DataFrames ---


def _make_wo_df() -> pd.DataFrame:
    data = {
        "wo_id": ["WO001", "WO002", "WO003", "WO004"],
        "wo_description": ["Oil Analysis", "Routine Maintenance", "Corrective Repair", "Emergency Fix"],
        "collection": ["compressor", "compressor", "motor", "motor"],
        "primary_code": ["MT010", "MT001", "MT013", "MT013"],
        "primary_code_description": ["Oil Analysis", "Routine Maintenance", "Corrective", "Corrective"],
        "secondary_code": ["MT010b", "MT001a", "MT013a", "MT013b"],
        "secondary_code_description": ["Routine Oil Analysis", "Basic Maint", "Repair", "Emergency"],
        "equipment_id": ["CWC04013", "CWC04013", "CWC04013", "CWC04007"],
        "equipment_name": ["Chiller 13", "Chiller 13", "Chiller 13", "Chiller 7"],
        "preventive": ["TRUE", "TRUE", "FALSE", "FALSE"],
        "work_priority": ["5", "5", "3", "1"],
        "actual_finish": [
            pd.Timestamp("2017-06-01"),
            pd.Timestamp("2017-08-15"),
            pd.Timestamp("2017-11-20"),
            pd.Timestamp("2018-03-10"),
        ],
        "duration": ["3:00", "2:00", "4:00", "6:00"],
        "actual_labor_hours": ["1:00", "1:00", "2:00", "3:00"],
    }
    return pd.DataFrame(data)


def _make_events_df() -> pd.DataFrame:
    data = {
        "event_id": ["E001", "E002", "E003"],
        "event_group": ["WORK_ORDER", "ALERT", "ANOMALY"],
        "event_category": ["PM", "ALERT", "ANOMALY"],
        "event_type": ["MT001", "CR00002", None],
        "description": ["Routine Maintenance", "Temperature Alert", "Anomaly Detected"],
        "equipment_id": ["CWC04013", "CWC04013", "CWC04013"],
        "equipment_name": ["Chiller 13", "Chiller 13", "Chiller 13"],
        "event_time": [
            pd.Timestamp("2017-06-01"),
            pd.Timestamp("2017-07-01"),
            pd.Timestamp("2017-08-01"),
        ],
        "note": [None, "High temp", None],
    }
    return pd.DataFrame(data)


def _make_failure_codes_df() -> pd.DataFrame:
    data = {
        "category": ["Maintenance and Routine Checks", "Maintenance and Routine Checks", "Corrective"],
        "primary_code": ["MT010", "MT001", "MT013"],
        "primary_code_description": ["Oil Analysis", "Routine Maintenance", "Corrective"],
        "secondary_code": ["MT010b", "MT001a", "MT013a"],
        "secondary_code_description": ["Routine Oil Analysis", "Basic Maint", "Repair"],
    }
    return pd.DataFrame(data)


def _make_primary_failure_codes_df() -> pd.DataFrame:
    data = {
        "category": ["Maintenance and Routine Checks", "Maintenance and Routine Checks", "Corrective"],
        "primary_code": ["MT010", "MT001", "MT013"],
        "primary_code_description": ["Oil Analysis", "Routine Maintenance", "Corrective"],
    }
    return pd.DataFrame(data)


def _make_alert_events_df() -> pd.DataFrame:
    data = {
        "equipment_id": ["CWC04013", "CWC04013", "CWC04013"],
        "equipment_name": ["Chiller 13", "Chiller 13", "Chiller 13"],
        "rule_id": ["CR00002", "CR00002", "CR00002"],
        "start_time": [
            pd.Timestamp("2017-01-01"),
            pd.Timestamp("2017-03-01"),
            pd.Timestamp("2017-06-01"),
        ],
        "end_time": [
            pd.Timestamp("2017-01-02"),
            pd.Timestamp("2017-03-02"),
            pd.Timestamp("2017-06-02"),
        ],
        "event_group": ["ALERT", "ALERT", "WORK_ORDER"],
    }
    return pd.DataFrame(data)


_FIXTURE_DATA = {
    "wo_events": _make_wo_df,
    "events": _make_events_df,
    "failure_codes": _make_failure_codes_df,
    "primary_failure_codes": _make_primary_failure_codes_df,
    "alert_events": _make_alert_events_df,
}


# --- Fixtures ---


@pytest.fixture
def mock_data():
    """Patch load() in tools namespace to return fixture DataFrames without CouchDB."""
    def _fake_load(key: str):
        factory = _FIXTURE_DATA.get(key)
        return factory() if factory else None

    with patch("servers.wo.tools.load", side_effect=_fake_load):
        yield


async def call_tool(mcp_instance, tool_name: str, args: dict) -> dict:
    """Helper: call an MCP tool and return parsed JSON response."""
    contents, _ = await mcp_instance.call_tool(tool_name, args)
    return json.loads(contents[0].text)
