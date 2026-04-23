# src/tests/test_episodic_store.py
import os, json, tempfile
from src.utils.episodic_store import EpisodicStore

def make_dummy_facts(path):
    lines = [
        {"fact_id":"c_0","dataset":"usm_c","source_file":"c.csv","asset_id":"c","row_index":0,
         "features":[{"name":"f1","value":1.2},{"name":"f2","value":0.3}],"label":"waxing","provenance":{"file":"c.csv","row":0},"confidence":1.0},
        {"fact_id":"c_1","dataset":"usm_c","source_file":"c.csv","asset_id":"c","row_index":1,
         "features":[{"name":"f1","value":2.1},{"name":"f2","value":0.5}],"label":"healthy","provenance":{"file":"c.csv","row":1},"confidence":1.0},
    ]
    with open(path,"w") as f:
        for l in lines:
            f.write(json.dumps(l)+"\n")

def test_store_ingest_and_query(tmp_path):
    jsonl = tmp_path / "facts.jsonl"
    make_dummy_facts(str(jsonl))
    dbpath = str(tmp_path / "store.db")
    store = EpisodicStore(db_path=dbpath)
    n = store.ingest_jsonl(str(jsonl))
    assert n == 2
    facts = store.query_by_asset("c")
    assert len(facts) == 2
    res = store.search_by_feature_threshold("f1", ">", 1.5)
    assert len(res) == 1
    store.close()
