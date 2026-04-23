# src/tests/test_static_fact_extractor.py
import json
import os
import tempfile
import pandas as pd
from src.utils.static_fact_extractor import extract_facts_from_csv

def make_dummy_csv(path):
    df = pd.DataFrame({
        "f1": [1.2, 2.3, 3.4],
        "f2": [0.1, 0.2, 0.3],
        "class": ["healthy", "waxing", "gas_injection"]
    })
    df.to_csv(path, index=False)

def test_extract_and_jsonl(tmp_path):
    csvp = tmp_path / "dummy.csv"
    make_dummy_csv(str(csvp))
    outp = str(tmp_path / "dummy_facts.jsonl")
    # call extractor
    result_path = extract_facts_from_csv(str(csvp), asset_id="meter_dummy", dataset_name="usm_test", out_path=outp)
    assert os.path.exists(result_path)
    with open(result_path, "r") as f:
        lines = [json.loads(l) for l in f.readlines()]
    assert len(lines) == 3
    # check fields
    first = lines[0]
    assert first["dataset"] == "usm_test"
    assert first["asset_id"] == "meter_dummy"
    assert "features" in first and isinstance(first["features"], list)
