"""A missing document and a missing database produce different outcomes."""

import httpx
import pytest

from servers.db_errors import DATA_UNAVAILABLE
from servers.wo.couch import CouchClient, CouchError, DatabaseUnavailable


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

    with pytest.raises(CouchError, match="data source does not exist"):
        await client.get("wo:MAIN:1")
    with pytest.raises(CouchError, match="data source does not exist"):
        await client.find({"type": "workorder"})


@pytest.mark.anyio
async def test_missing_database_message_hides_database_name() -> None:
    with pytest.raises(CouchError) as exc_info:
        await _client("Database does not exist.").get("wo:MAIN:1")

    assert "workorder" not in str(exc_info.value)


def _transport_client(handler) -> CouchClient:
    client = CouchClient("http://couch.test", "secret_wo")
    client._c = httpx.AsyncClient(
        base_url="http://couch.test", transport=httpx.MockTransport(handler)
    )
    return client


@pytest.mark.anyio
async def test_unreachable_couchdb_raises_unavailable() -> None:
    def refuse(request):
        raise httpx.ConnectError("connection refused", request=request)

    client = _transport_client(refuse)

    with pytest.raises(DatabaseUnavailable, match="does not exist or is unreachable"):
        await client.find({"type": "workorder"})


@pytest.mark.anyio
async def test_http_error_hides_url_and_database_name() -> None:
    client = _transport_client(lambda request: httpx.Response(500, json={}))

    with pytest.raises(CouchError) as exc_info:
        await client.find({"type": "workorder"})

    assert str(exc_info.value) == "database request failed (HTTP 500)"


@pytest.mark.anyio
async def test_failure_codes_report_unavailable_database() -> None:
    from servers.wo import workorders

    result = await workorders.get_failure_codes(_client("Database does not exist."))

    assert result["error"] == DATA_UNAVAILABLE


@pytest.mark.anyio
async def test_wonum_allocation_does_not_retry_http_errors() -> None:
    def handler(request):
        if request.method == "GET":
            return httpx.Response(404, json={"error": "not_found", "reason": "missing"})
        return httpx.Response(500, json={})

    with pytest.raises(CouchError, match="HTTP 500"):
        await _transport_client(handler).next_wonum("MAIN")


@pytest.mark.anyio
async def test_wonum_allocation_retries_conflicts() -> None:
    puts = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(404, json={"error": "not_found", "reason": "missing"})
        puts.append(request)
        if len(puts) == 1:
            return httpx.Response(409, json={"error": "conflict"})
        return httpx.Response(201, json={"ok": True})

    assert await _transport_client(handler).next_wonum("MAIN") == "1001"
    assert len(puts) == 2
