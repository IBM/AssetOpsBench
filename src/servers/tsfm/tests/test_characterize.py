"""P4: characterize_series through the MCP boundary — file pointer in → pattern evidence out."""

import asyncio, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from ..main import mcp
from ..io import refs

N = 168


def call(name, args):
    content, _ = asyncio.run(mcp.call_tool(name, args))
    return json.loads(content[0].text)


def _write(frame: dict) -> str:
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=N, freq="h"), **frame})
    refs._ensure_workdir()
    import os
    p = os.path.join(refs.WORKDIR, "characterize_in.csv")
    df.to_csv(p, index=False)
    return p


def test_generic_default_per_channel():
    p = _write({"sensor_A": np.linspace(0, 6, N), "flow": 0.1 * np.random.RandomState(0).randn(N)})
    r = call("characterize_series", {"dataset_path": p, "timestamp_column": "timestamp"})
    assert r["status"] == "success" and r["evidence_file"].startswith("file://")
    assert set(r["groups"]) == {"sensor_A", "flow"}
    pg = r["phases"][0]["per_group"]
    assert pg["sensor_A"]["state"] == "RISE" and pg["flow"]["state"] == "STABLE"


def test_sentsr_decoupled_with_preset_grouping():
    # vibration rises, temperature stable — the SenTSR row-0 archetype, via the opt-in preset
    ramp = np.linspace(0, 6, N)
    rng = np.random.RandomState(1)
    p = _write({"Acceleration": ramp + 0.1 * rng.randn(N), "Velocity": ramp + 0.1 * rng.randn(N),
                "Temperature": 0.1 * rng.randn(N)})
    r = call("characterize_series", {"dataset_path": p, "timestamp_column": "timestamp",
                                     "group_rules": "vibration_temperature"})
    assert set(r["groups"]) == {"vibration", "temperature"}
    pg = r["phases"][0]["per_group"]
    assert pg["vibration"]["state"] == "RISE" and pg["temperature"]["state"] == "STABLE"
    assert any(rel["type"] == "DECOUPLED" for rel in r["phases"][0]["relations"])


def test_summary_is_fault_free():
    ramp = np.linspace(0, 6, N)
    p = _write({"Acceleration": ramp, "Velocity": ramp, "Temperature": 0.1 * np.random.RandomState(2).randn(N)})
    r = call("characterize_series", {"dataset_path": p, "timestamp_column": "timestamp",
                                     "group_rules": "vibration_temperature"})
    banned = ["alignment", "bearing", "lubric", "wear", "imbalance", "friction", "fault", "gear"]
    assert not any(b in r["summary"].lower() for b in banned), r["summary"]


def test_validation_error():
    assert "error" in call("characterize_series", {"dataset_path": ""})
