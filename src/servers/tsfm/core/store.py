"""Core storage abstraction — the single registry layer everything else sits on.

Every catalog and result table is a *collection* of JSON docs. `Store` gives them a uniform
get / put / find / delete / list, with two interchangeable backends:

  - MemoryStore : in-process dict; deterministic, no dependencies — used for tests, the demo,
                  and the deterministic benchmark.
  - CouchStore  : the AssetOpsBench CouchDB (same wire format as the existing loader); used in
                  production and participates in export_state() (#394 / PR #400).

Switch via `make_store()` reading TSFM_STORE=memory|couch. A tiny Mango-subset selector
({field: value} and {field: {"$gte"/"$lte"/"$elemMatch": ...}}) keeps query code identical
across backends.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional


# --------------------------------------------------------------------------- #
# selector matching (Mango subset, used by MemoryStore and as a fallback)
# --------------------------------------------------------------------------- #
def _match(doc: dict, selector: dict) -> bool:
    for field, cond in (selector or {}).items():
        val = doc.get(field)
        if isinstance(cond, dict):
            for op, operand in cond.items():
                if op == "$gte" and not (val is not None and val >= operand):
                    return False
                elif op == "$lte" and not (val is not None and val <= operand):
                    return False
                elif op == "$gt" and not (val is not None and val > operand):
                    return False
                elif op == "$eq" and not (val == operand):
                    return False
                elif op == "$ne" and not (val != operand):
                    return False
                elif op == "$in" and not (val in operand):
                    return False
                elif op == "$elemMatch":
                    if not isinstance(val, list) or not any(_match({"_": v}, {"_": operand})
                                                            for v in val):
                        # $elemMatch with a scalar equality operand
                        if not (isinstance(val, list) and operand in val):
                            return False
        else:
            if val != cond:
                return False
    return True


def _strip(doc: dict) -> dict:
    d = dict(doc)
    d.pop("_rev", None)
    return d


class Store:
    """Backend-agnostic interface."""
    def get(self, collection: str, doc_id: str) -> Optional[dict]: ...
    def put(self, collection: str, doc: dict) -> dict: ...
    def find(self, collection: str, selector: Optional[dict] = None,
             limit: int = 1000) -> List[dict]: ...
    def delete(self, collection: str, doc_id: str) -> bool: ...
    def list_collections(self) -> List[str]: ...
    def export_state(self, collections: Optional[Iterable[str]] = None) -> Dict[str, List[dict]]:
        cols = list(collections) if collections else self.list_collections()
        return {c: self.find(c) for c in cols}


# --------------------------------------------------------------------------- #
class MemoryStore(Store):
    def __init__(self):
        self._db: Dict[str, Dict[str, dict]] = {}

    def _col(self, c): return self._db.setdefault(c, {})

    def get(self, collection, doc_id):
        d = self._col(collection).get(doc_id)
        return _strip(d) if d else None

    def put(self, collection, doc):
        if "_id" not in doc:
            raise ValueError("doc requires _id")
        self._col(collection)[doc["_id"]] = dict(doc)
        return _strip(doc)

    def find(self, collection, selector=None, limit=1000):
        docs = [_strip(d) for d in self._col(collection).values() if _match(d, selector or {})]
        return docs[:limit]

    def delete(self, collection, doc_id):
        return self._col(collection).pop(doc_id, None) is not None

    def list_collections(self):
        return [c for c, docs in self._db.items() if docs]


# --------------------------------------------------------------------------- #
class CouchStore(Store):
    """CouchDB backend — same wire format as the AssetOpsBench loader."""
    def __init__(self, url=None, auth=None):
        import requests  # lazy
        self._requests = requests
        self.url = (url or os.environ.get("COUCHDB_URL", "http://localhost:5984")).rstrip("/")
        self.auth = auth or (os.environ.get("COUCHDB_USERNAME", "admin"),
                             os.environ.get("COUCHDB_PASSWORD", "password"))

    def _u(self, *p): return "/".join([self.url, *p])

    def _ensure(self, collection):
        r = self._requests.head(self._u(collection), auth=self.auth, timeout=10)
        if r.status_code == 404:
            self._requests.put(self._u(collection), auth=self.auth, timeout=10)

    def get(self, collection, doc_id):
        r = self._requests.get(self._u(collection, doc_id), auth=self.auth, timeout=10)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return _strip(r.json())

    def put(self, collection, doc):
        self._ensure(collection)
        ex = self._requests.get(self._u(collection, doc["_id"]), auth=self.auth, timeout=10)
        body = dict(doc)
        if ex.status_code == 200:
            body["_rev"] = ex.json()["_rev"]
        r = self._requests.put(self._u(collection, doc["_id"]), json=body, auth=self.auth, timeout=15)
        r.raise_for_status()
        return _strip(doc)

    def find(self, collection, selector=None, limit=1000):
        r = self._requests.post(self._u(collection, "_find"),
                                json={"selector": selector or {"_id": {"$gt": None}}, "limit": limit},
                                auth=self.auth, timeout=20)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return [_strip(d) for d in r.json().get("docs", [])]

    def delete(self, collection, doc_id):
        ex = self._requests.get(self._u(collection, doc_id), auth=self.auth, timeout=10)
        if ex.status_code != 200:
            return False
        self._requests.delete(self._u(collection, doc_id),
                              params={"rev": ex.json()["_rev"]}, auth=self.auth, timeout=10)
        return True

    def list_collections(self):
        r = self._requests.get(self._u("_all_dbs"), auth=self.auth, timeout=10)
        r.raise_for_status()
        return [d for d in r.json() if not d.startswith("_")]


def make_store() -> Store:
    return CouchStore() if os.environ.get("TSFM_STORE", "memory") == "couch" else MemoryStore()
