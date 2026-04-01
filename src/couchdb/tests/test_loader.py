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
    assert manifest["failure_mode"] == [
        "shared/fmea/failure_modes_sample.json",
        "shared/fmea/transformer_failure_modes.json",
    ]


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


def test_default_manifest_loads_transformer_registry_telemetry_and_failure_modes():
    base_dir = Path(loader.__file__).parent / "scenarios_data" / "default"
    manifest = json.loads((base_dir / "manifest.json").read_text())
    docs = {
        name: loader._collect_docs(
            name, manifest[name], loader.collection_config(name), base_dir=str(base_dir)
        )
        for name in ("asset", "iot", "failure_mode")
    }
    transformer = next(d for d in docs["asset"] if d["assetnum"] == "Transformer 1")
    assert transformer["siteid"] == "MAIN"
    telemetry = [d for d in docs["iot"] if d["asset_id"] == transformer["assetnum"]]
    assert telemetry
    assert "hydrogen" in transformer["sensors"]
    assert all("hydrogen" in d for d in telemetry)
    modes = next(d for d in docs["failure_mode"] if d["asset_class"] == "transformer")
    assert len(modes["failure_modes"]) == 9
    assert "Partial Discharge in Oil" in modes["failure_modes"]
