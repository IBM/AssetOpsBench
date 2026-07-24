import json
from pathlib import Path

from couchdb import loader


def test_failure_code_collection_parses_shared_csv() -> None:
    cfg = loader.collection_config("failure_code")
    base_dir = Path(loader.__file__).parent / "scenarios_data" / "default"

    docs = loader._collect_docs(
        "failure_code",
        "shared/failure_code/failure_code_sample.csv",
        cfg,
        base_dir=str(base_dir),
    )

    assert cfg["format"] == "csv"
    assert len(docs) == 10
    assert docs[0]["code"] == "FC001"
    assert docs[4]["description"] == "excessive vibration, shaking, or instability"


def test_default_manifest_uses_failure_code_collection_key() -> None:
    manifest_path = (
        Path(loader.__file__).parent / "scenarios_data" / "default" / "manifest.json"
    )

    with manifest_path.open() as f:
        manifest = json.load(f)

    assert "failure_code" in manifest
    assert "failurecode" not in manifest
    assert manifest["catalog"] == [
        "shared/catalog/assets.csv",
        "shared/catalog/failure_modes.csv",
        "shared/catalog/sensors.csv",
    ]
    assert manifest["failure_mode"] == "shared/fmea/failure_modes_sample.json"


def test_failure_mode_collection_parses_shared_json() -> None:
    cfg = loader.collection_config("failure_mode")
    base_dir = Path(loader.__file__).parent / "scenarios_data" / "default"

    docs = loader._collect_docs(
        "failure_mode",
        "shared/fmea/failure_modes_sample.json",
        cfg,
        base_dir=str(base_dir),
    )

    assert cfg["format"] == "json"
    assert cfg["primary_key"] == ["asset_class"]
    assert len(docs) == 1
    assert docs[0]["asset_class"] == "pump"
    assert docs[0]["failure_modes"] == ["seal leakage", "impeller wear"]


def test_iot_summary_doc_materializes_full_stream_metadata() -> None:
    summaries = loader._make_iot_summary_docs(
        [
            {
                "asset_id": "Pump-1",
                "timestamp": "2024-01-01T00:00:00",
                "Temp": 1.0,
                "Pressure": None,
            },
            {
                "asset_id": "Pump-1",
                "timestamp": "2024-01-01T00:02:00",
                "Temp": "3",
                "Pressure": 5.0,
            },
            {
                "asset_id": "Pump-1",
                "timestamp": "2024-01-01T00:01:00",
                "Temp": True,
                "Pressure": 7.0,
            },
        ]
    )

    assert len(summaries) == 2
    summaries_by_id = {summary["_id"]: summary for summary in summaries}
    summary = summaries_by_id["iot_summary:Pump-1"]
    daily = summaries_by_id["iot_summary_day:Pump-1:2024-01-01"]

    assert summary["_id"] == "iot_summary:Pump-1"
    assert summary["doctype"] == "iot_asset_summary"
    assert summary["summary_asset_id"] == "Pump-1"
    assert "asset_id" not in summary
    assert "daily" not in summary
    assert summary["timestamped_records"] == 3
    assert summary["start_time"] == "2024-01-01T00:00:00"
    assert summary["end_time"] == "2024-01-01T00:02:00"
    assert summary["sensors"] == ["Pressure", "Temp"]
    assert summary["latest"] == {
        "timestamp": "2024-01-01T00:02:00",
        "values": {"Temp": "3", "Pressure": 5.0},
    }

    assert summary["coverage"]["Temp"] == {
        "non_null_count": 3,
        "first_timestamp": "2024-01-01T00:00:00",
        "last_timestamp": "2024-01-01T00:02:00",
        "latest_timestamp": "2024-01-01T00:02:00",
        "latest_value": "3",
    }
    assert summary["stats"]["Temp"] == {
        "count": 2,
        "null_count": 1,
        "min": 1.0,
        "max": 3.0,
        "mean": 2.0,
        "stddev": 1.0,
        "first_timestamp": "2024-01-01T00:00:00",
        "last_timestamp": "2024-01-01T00:02:00",
    }
    assert daily["doctype"] == "iot_asset_daily_summary"
    assert daily["summary_asset_id"] == "Pump-1"
    assert daily["day"] == "2024-01-01"
    assert daily["timestamped_records"] == 3
    assert daily["start_time"] == "2024-01-01T00:00:00"
    assert daily["end_time"] == "2024-01-01T00:02:00"
    assert daily["sensors"] == ["Pressure", "Temp"]
    assert daily["stats"]["Pressure"]["mean"] == 6.0


def test_iot_load_appends_summary_doc_but_returns_source_count(monkeypatch) -> None:
    source_docs = [
        {
            "asset_id": "Pump-1",
            "timestamp": "2024-01-01T00:00:00",
            "Temp": 1.0,
        }
    ]
    inserted = {}

    monkeypatch.setattr(
        loader,
        "collection_config",
        lambda key: {"format": "json", "primary_key": ["asset_id", "timestamp"]},
    )
    monkeypatch.setattr(loader, "_collect_docs", lambda *args, **kwargs: source_docs)
    monkeypatch.setattr(loader, "_ensure_db", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader, "_install_design", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader, "_create_indexes", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        loader,
        "_bulk_insert",
        lambda db, docs: inserted.setdefault("docs", docs),
    )

    db, count = loader.load_collection("iot", "unused")

    assert db == "iot"
    assert count == 1
    assert [doc["_id"] for doc in inserted["docs"]] == [
        "iot:Pump-1:2024-01-01T00:00:00",
        "iot_summary:Pump-1",
        "iot_summary_day:Pump-1:2024-01-01",
    ]
