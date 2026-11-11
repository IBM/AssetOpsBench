"""CouchDB identifiers cannot change the endpoint of an authenticated request."""

from urllib.parse import unquote, urlsplit

import pytest
import requests

from servers.tsfm.core.store import CouchStore
from servers.tsfm.stores.results import get_result


@pytest.fixture
def sent_requests(monkeypatch):
    sent = []

    def send(session, request, **kwargs):
        sent.append(request)
        response = requests.Response()
        response.status_code = 404
        response._content = b'{"error":"not_found"}'
        return response

    monkeypatch.setattr(requests.Session, "send", send)
    return sent


@pytest.mark.parametrize(
    "doc_id",
    [
        "../iot/secret_doc",
        "../_node/_local/_config/admins",
        "../_node/_local/_config",
        "../_all_dbs",
        "..%2Fiot%2Fsecret_doc",
        "%2e%2e/%2e%2e/_all_dbs",
        "..\\iot\\secret_doc",
        "doc?rev=other#fragment",
        "fr:012345abcdef",
        "run:012345abcdef",
        "model:org/model",
    ],
)
def test_result_id_stays_in_result_collection(doc_id, sent_requests):
    store = CouchStore(url="http://couch.test", auth=("test", "test"))

    assert get_result(store, "tsfm_forecasting", doc_id) is None

    assert len(sent_requests) == 1
    url = urlsplit(sent_requests[0].url)
    assert url.path.split("/")[:2] == ["", "forecast_result"]
    assert len(url.path.split("/")) == 3
    assert unquote(url.path.split("/")[2]) == doc_id
    assert url.query == url.fragment == ""


@pytest.mark.parametrize("doc_id", [".", ".."])
def test_dot_only_ids_are_rejected_before_request(doc_id, sent_requests):
    store = CouchStore(url="http://couch.test")

    with pytest.raises(ValueError, match="dot segments"):
        store.get("tsfm_runs", doc_id)

    assert sent_requests == []


def test_run_id_stays_in_runs_collection(sent_requests):
    store = CouchStore(url="http://couch.test")

    assert store.get("tsfm_runs", "../iot/secret_doc") is None

    assert urlsplit(sent_requests[0].url).path == "/tsfm_runs/..%2Fiot%2Fsecret_doc"
