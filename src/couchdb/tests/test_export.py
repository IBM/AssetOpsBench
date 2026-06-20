"""Export tests — no live CouchDB; requests is mocked."""
import json, os, sys, tempfile, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import couchdb.loader as loader
import couchdb.init_data as init_data


class _Resp:
    def __init__(self, payload, code=200):
        self._p, self.status_code = payload, code
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")
    def json(self):
        return self._p


def _patch(get):
    loader.requests = types.SimpleNamespace(get=get)
    init_data.loader = loader


def test_strips_rev_and_design_docs():
    _patch(lambda url, **k: _Resp({"rows": [
        {"doc": {"_id": "_design/x", "views": {}}},
        {"doc": {"_id": "wo:1", "_rev": "1-a", "status": "WAPPR"}},
    ]}))
    docs = loader.export_database("workorder")
    assert docs == [{"_id": "wo:1", "status": "WAPPR"}]


def test_include_design_keeps_design_doc():
    _patch(lambda url, **k: _Resp({"rows": [{"doc": {"_id": "_design/x"}}]}))
    assert loader.export_database("workorder", include_design=True)


def test_missing_db_returns_empty_not_error():
    _patch(lambda url, **k: _Resp({}, 404))
    assert loader.export_database("ghost") == []


def test_export_state_excludes_system_dbs_and_writes_file():
    def get(url, **k):
        if url.endswith("/_all_dbs"):
            return _Resp(["workorder", "_users"])
        return _Resp({"rows": [{"doc": {"_id": "a:1", "_rev": "1-z", "v": 1}}]})
    _patch(get)
    fp = os.path.join(tempfile.mkdtemp(), "state.json")
    state = init_data.export_state(dest=fp, managed_only=False)
    assert "_users" not in state and list(state) == ["workorder"]
    assert json.load(open(fp)) == state