# src/utils/episodic_store.py
"""
Episodic Store:
- Ingest JSONL facts and store them in a small SQLite DB (facts table + feature_index).
- Provide query API used by prompt builder / verifier.
"""

import json, sqlite3, os, typing
from typing import Dict, Any, List, Optional, Tuple

DEFAULT_DB_PATH = "data/episodic_store.db"

CREATE_FACTS_TABLE = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id TEXT PRIMARY KEY,
    dataset TEXT,
    source_file TEXT,
    asset_id TEXT,
    row_index INTEGER,
    label TEXT,
    fact_json TEXT
);
"""

CREATE_FEATURES_TABLE = """
CREATE TABLE IF NOT EXISTS features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id TEXT,
    feature_name TEXT,
    feature_value REAL,
    feature_text TEXT,
    FOREIGN KEY(fact_id) REFERENCES facts(fact_id)
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_facts_asset ON facts(asset_id);",
    "CREATE INDEX IF NOT EXISTS idx_facts_label ON facts(label);",
    "CREATE INDEX IF NOT EXISTS idx_features_name ON features(feature_name);",
    "CREATE INDEX IF NOT EXISTS idx_features_value ON features(feature_value);"
]


class EpisodicStore:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        c = self.conn.cursor()
        c.execute(CREATE_FACTS_TABLE)
        c.execute(CREATE_FEATURES_TABLE)
        for s in CREATE_INDEXES:
            c.execute(s)
        self.conn.commit()

    def ingest_jsonl(self, jsonl_path: str, overwrite: bool = False) -> int:
        """Ingest facts from JSONL file. Returns number of facts ingested."""
        count = 0
        with open(jsonl_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                fact = json.loads(line)
                fid = fact.get("fact_id")
                if not fid:
                    # create deterministic id if missing
                    fid = f"fact_{fact.get('source_file','unk')}_{fact.get('row_index',count)}"
                    fact["fact_id"] = fid
                # skip if exists and not overwrite
                if not overwrite and self.get_fact(fid) is not None:
                    continue
                self._upsert_fact(fact)
                count += 1
        return count

    def _upsert_fact(self, fact: Dict[str, Any]):
        c = self.conn.cursor()
        # prepare values
        fact_id = fact["fact_id"]
        dataset = fact.get("dataset")
        source_file = fact.get("source_file")
        asset_id = fact.get("asset_id")
        row_index = fact.get("row_index")
        label = fact.get("label")
        fact_json = json.dumps(fact)
        # upsert: delete then insert for simplicity
        c.execute("DELETE FROM features WHERE fact_id = ?", (fact_id,))
        c.execute("REPLACE INTO facts(fact_id,dataset,source_file,asset_id,row_index,label,fact_json) VALUES (?,?,?,?,?,?,?)",
                  (fact_id,dataset,source_file,asset_id,row_index,label,fact_json))
        # insert features
        features = fact.get("features", [])
        for feat in features:
            name = feat.get("name")
            val = feat.get("value")
            if isinstance(val, (int,float)):
                c.execute("INSERT INTO features(fact_id,feature_name,feature_value,feature_text) VALUES (?,?,?,?)",
                          (fact_id,name,float(val),None))
            else:
                # try numeric conversion
                try:
                    fv = float(val)
                    c.execute("INSERT INTO features(fact_id,feature_name,feature_value,feature_text) VALUES (?,?,?,?)",
                              (fact_id,name,fv,None))
                except Exception:
                    c.execute("INSERT INTO features(fact_id,feature_name,feature_value,feature_text) VALUES (?,?,?,?)",
                              (fact_id,name,None,str(val)))
        self.conn.commit()

    def get_fact(self, fact_id: str) -> Optional[Dict[str,Any]]:
        c = self.conn.cursor()
        r = c.execute("SELECT fact_json FROM facts WHERE fact_id = ?", (fact_id,)).fetchone()
        if r is None:
            return None
        return json.loads(r[0])

    def list_assets(self) -> List[str]:
        c = self.conn.cursor()
        rows = c.execute("SELECT DISTINCT asset_id FROM facts WHERE asset_id IS NOT NULL").fetchall()
        return [r[0] for r in rows]

    def query_by_asset(self, asset_id: str, limit:int=100) -> List[Dict[str,Any]]:
        c = self.conn.cursor()
        rows = c.execute("SELECT fact_json FROM facts WHERE asset_id = ? LIMIT ?", (asset_id,limit)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def query_by_label(self, label: str, limit:int=100) -> List[Dict[str,Any]]:
        c = self.conn.cursor()
        rows = c.execute("SELECT fact_json FROM facts WHERE label = ? LIMIT ?", (label,limit)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def search_by_feature_threshold(self, feature_name: str, op: str, value: float, limit:int=200) -> List[Dict[str,Any]]:
        """
        Search facts where numeric feature meets threshold, op in ('<','>','<=','>=','=')
        """
        if op not in ("<",">","<=",">=","="):
            raise ValueError("op must be one of <,>,<=,>=,=")
        sql = f"SELECT DISTINCT f.fact_json FROM features AS ft JOIN facts AS f ON ft.fact_id=f.fact_id WHERE ft.feature_name=? AND ft.feature_value {op} ? LIMIT ?"
        c = self.conn.cursor()
        rows = c.execute(sql, (feature_name, float(value), limit)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def sample_balanced(self, labels: List[str], per_label:int=10) -> List[Dict[str,Any]]:
        out = []
        for lab in labels:
            out.extend(self.query_by_label(lab, limit=per_label))
        return out

    def close(self):
        self.conn.close()

# CLI helper
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    parser.add_argument("--ingest", help="path to facts jsonl")
    parser.add_argument("--list-assets", action="store_true")
    parser.add_argument("--get-fact", help="fact_id to fetch")
    args = parser.parse_args()
    store = EpisodicStore(db_path=args.db)
    if args.ingest:
        n = store.ingest_jsonl(args.ingest)
        print(f"Ingested {n} facts to {args.db}")
    if args.list_assets:
        print(store.list_assets())
    if args.get_fact:
        print(json.dumps(store.get_fact(args.get_fact), indent=2))
    store.close()
