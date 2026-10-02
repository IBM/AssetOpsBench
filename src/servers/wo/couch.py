"""Minimal async CouchDB client used by the WO tools.

Only the handful of operations the tools need: get / put / delete a document,
Mango `_find`, design-doc views, and a deterministic work-order-number counter.

The tool functions in `workorders.py` depend only on this small interface (duck
typed), so they can be unit-tested against an in-memory fake (see test_workorders.py)
without a running CouchDB.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from servers.db_errors import DATA_UNAVAILABLE

try:
    import httpx
except Exception:  # httpx optional at import time so the fake-backed tests still run
    httpx = None  # type: ignore


class CouchError(Exception):
    pass


class DatabaseUnavailable(CouchError):
    """The database does not exist or CouchDB is unreachable."""

    def __init__(self) -> None:
        super().__init__(DATA_UNAVAILABLE)


class CouchClient:
    def __init__(
        self,
        base_url: str,
        db: str,
        *,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 10.0,
    ):
        if httpx is None:
            raise CouchError(
                "httpx is required for the real CouchClient (pip install httpx)"
            )
        self.db = db
        auth = (username, password) if username else None
        self._c = httpx.AsyncClient(
            base_url=base_url.rstrip("/"), auth=auth, timeout=timeout
        )

    async def aclose(self) -> None:
        await self._c.aclose()

    def _raise_if_missing_db(self, r: "httpx.Response") -> None:
        """CouchDB answers 404 both for a missing doc and a missing database;
        only the body's reason tells them apart."""
        if r.status_code != 404:
            return
        try:
            reason = r.json().get("reason")
        except ValueError:
            return
        if reason == "Database does not exist.":
            raise DatabaseUnavailable()

    async def _request(self, method: str, path: str, **kwargs: Any) -> "httpx.Response":
        """Send one request. An unreachable server and a missing database raise the
        data-unavailable error; the URL (database name) never reaches the message."""
        try:
            r = await self._c.request(method, path, **kwargs)
        except httpx.TransportError as exc:
            raise DatabaseUnavailable() from exc
        self._raise_if_missing_db(r)
        return r

    @staticmethod
    def _raise_for_status(r: "httpx.Response") -> None:
        if r.is_error:
            raise CouchError(f"database request failed (HTTP {r.status_code})")

    # ---- document CRUD ----
    async def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        r = await self._request("GET", f"/{self.db}/{doc_id}")
        if r.status_code == 404:
            return None
        self._raise_for_status(r)
        return r.json()

    async def put(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        if "_id" not in doc:
            raise CouchError("document must have _id")
        r = await self._request("PUT", f"/{self.db}/{doc['_id']}", json=doc)
        if r.status_code == 409:
            raise CouchError(f"conflict updating {doc['_id']} (stale _rev)")
        self._raise_for_status(r)
        return r.json()

    async def delete(self, doc_id: str, rev: str) -> Dict[str, Any]:
        r = await self._request("DELETE", f"/{self.db}/{doc_id}", params={"rev": rev})
        self._raise_for_status(r)
        return r.json()

    # ---- queries ----
    async def find(
        self,
        selector: Dict[str, Any],
        *,
        fields: Optional[List[str]] = None,
        sort: Optional[List[Dict[str, str]]] = None,
        limit: int = 200,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        body: Dict[str, Any] = {"selector": selector, "limit": limit, "skip": skip}
        if fields:
            body["fields"] = fields
        if sort:
            body["sort"] = sort
        r = await self._request("POST", f"/{self.db}/_find", json=body)
        self._raise_for_status(r)
        return r.json().get("docs", [])

    async def view(self, ddoc: str, view: str, **params: Any) -> Dict[str, Any]:
        # CouchDB expects JSON-encoded key/startkey/endkey params.
        import json as _json

        q = {
            k: (_json.dumps(v) if k in ("key", "startkey", "endkey") else v)
            for k, v in params.items()
        }
        r = await self._request(
            "GET", f"/{self.db}/_design/{ddoc}/_view/{view}", params=q
        )
        self._raise_for_status(r)
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
