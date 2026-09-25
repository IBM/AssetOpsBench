"""A missing document and a missing database produce different outcomes."""

import httpx
import pytest

from servers.wo.couch import CouchClient, CouchError


def _client(reason: str) -> CouchClient:
    client = CouchClient("http://couch.test", "workorder")
    client._c = httpx.AsyncClient(
        base_url="http://couch.test",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                404, json={"error": "not_found", "reason": reason}
            )
        ),
    )
    return client


@pytest.mark.anyio
async def test_missing_document_returns_none() -> None:
    assert await _client("missing").get("wo:MAIN:1") is None


@pytest.mark.anyio
async def test_missing_database_raises_distinct_message() -> None:
    client = _client("Database does not exist.")

    with pytest.raises(CouchError, match="database 'workorder' does not exist"):
        await client.get("wo:MAIN:1")
    with pytest.raises(CouchError, match="database 'workorder' does not exist"):
        await client.find({"type": "workorder"})
