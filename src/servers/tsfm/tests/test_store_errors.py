"""CouchStore failures: a missing catalog or an unreachable server is unavailable
data, not an empty catalog, and no error carries the URL or database name."""

import pytest
import requests

from servers.db_errors import DATA_UNAVAILABLE

from ..core.store import CouchStore, StoreError, StoreUnavailable
from ..stores import feature_store, model_store, results


class _Response:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class _Requests:
    """Stands in for `requests`: every call gets the same response."""

    exceptions = requests.exceptions

    def __init__(self, status_code, body):
        self.response = _Response(status_code, body)

    def request(self, method, url, **kwargs):
        return self.response


def _store(status_code, body):
    store = CouchStore(url="http://couch.test")
    store._requests = _Requests(status_code, body)
    return store


def _no_databases():
    return _store(404, {"error": "not_found", "reason": "Database does not exist."})


def test_unreachable_server_raises_unavailable_without_url():
    store = CouchStore(url="http://127.0.0.1:9")

    with pytest.raises(StoreUnavailable) as exc_info:
        store.find("secret_collection")

    assert str(exc_info.value) == DATA_UNAVAILABLE


def test_missing_model_catalog_is_unavailable_not_empty():
    with pytest.raises(StoreUnavailable):
        model_store.list_models(_no_databases())
    with pytest.raises(StoreUnavailable):
        model_store.get_model(_no_databases(), "ttm_r2")


def test_missing_feature_catalog_is_unavailable_not_empty():
    with pytest.raises(StoreUnavailable):
        feature_store.get_feature(_no_databases(), "mean")


def test_collections_created_on_write_read_as_empty():
    store = _no_databases()

    assert results.list_results(store, "tsfm_forecasting") == []
    assert store.get("tsfm_runs", "r1") is None


def test_http_error_hides_url_and_database_name():
    with pytest.raises(StoreError) as exc_info:
        _store(500, {}).find("secret_collection")

    assert str(exc_info.value) == "database request failed (HTTP 500)"
