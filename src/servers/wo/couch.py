"""Minimal async CouchDB client used by the WO tools.

Only the handful of operations the tools need: get / put / delete a document,
Mango `_find`, design-doc views, and a deterministic work-order-number counter.

The tool functions in `workorders.py` depend only on this small interface (duck
typed), so they can be unit-tested against an in-memory fake (see test_workorders.py)
without a running CouchDB.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import httpx
except Exception:  # httpx optional at import time so the fake-backed tests still run
    httpx = None  # type: ignore


class CouchError(Exception):
    pass


class CouchClient:
    def __init__(self, base_url: str, db: str, *, username: Optional[str] = None,
                 password: Optional[str] = None, timeout: float = 10.0):
        if httpx is None:
            raise CouchError("httpx is required for the real CouchClient (pip install httpx)")
        self.db = db
        auth = (username, password) if username else None
        self._c = httpx.AsyncClient(base_url=base_url.rstrip("/"), auth=auth, timeout=timeout)

    async def aclose(self) -> None:
        await self._c.aclose()

    # ---- document CRUD ----
    async def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        r = await self._c.get(f"/{self.db}/{doc_id}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    async def put(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        if "_id" not in doc:
            raise CouchError("document must have _id")
        r = await self._c.put(f"/{self.db}/{doc['_id']}", json=doc)
        if r.status_code == 409:
            raise CouchError(f"conflict updating {doc['_id']} (stale _rev)")
        r.raise_for_status()
        return r.json()

    async def delete(self, doc_id: str, rev: str) -> Dict[str, Any]:
        r = await self._c.delete(f"/{self.db}/{doc_id}", params={"rev": rev})
        r.raise_for_status()
        return r.json()

    # ---- queries ----
    async def find(self, selector: Dict[str, Any], *, fields: Optional[List[str]] = None,
                   sort: Optional[List[Dict[str, str]]] = None, limit: int = 200,
                   skip: int = 0) -> List[Dict[str, Any]]:
        body: Dict[str, Any] = {"selector": selector, "limit": limit, "skip": skip}
        if fields:
            body["fields"] = fields
        if sort:
            body["sort"] = sort
        r = await self._c.post(f"/{self.db}/_find", json=body)
        r.raise_for_status()
        return r.json().get("docs", [])

    async def view(self, ddoc: str, view: str, **params: Any) -> Dict[str, Any]:
        # CouchDB expects JSON-encoded key/startkey/endkey params.
        import json as _json
        q = {k: (_json.dumps(v) if k in ("key", "startkey", "endkey") else v)
             for k, v in params.items()}
        r = await self._c.get(f"/{self.db}/_design/{ddoc}/_view/{view}", params=q)
        r.raise_for_status()
        return r.json()

    # ---- deterministic WO number allocation ----
    async def next_wonum(self, site_id: str) -> str:
        """Allocate the next WONUM for a site from a counter doc.

        Reproducible across a benchmark run because allocation is sequential and
        seeded by `reset`. For fully fixed ids, callers may pass an explicit wonum
        to create_workorder instead.
        """
        cid = f"counter:{site_id.upper()}"
        for _ in range(5):  # retry on write conflict
            doc = await self.get(cid) or {"_id": cid, "type": "counter", "value": 1000}
            doc["value"] = int(doc["value"]) + 1
            try:
                await self.put(doc)
                return str(doc["value"])
            except CouchError:
                continue
        raise CouchError("could not allocate wonum (counter contention)")