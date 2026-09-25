import json

import pytest
from couchdb3.exceptions import ConflictError
from unittest.mock import patch


async def call_tool(mcp_instance, tool_name: str, args: dict) -> dict:
    """Helper: call an MCP tool and return parsed JSON response."""
    contents, _ = await mcp_instance.call_tool(tool_name, args)
    return json.loads(contents[0].text)


class FakeDatabase:
    """In-memory stand-in for couchdb3.Database, including _rev conflict checks."""

    def __init__(self, docs=None):
        self.docs = {doc["_id"]: {"_rev": "1-fake", **doc} for doc in docs or []}

    def find(self, selector, fields=None, limit=None):
        docs = [dict(doc) for doc in self.docs.values() if self._matches(doc, selector)]
        if limit is not None:
            docs = docs[:limit]
        if fields is not None:
            docs = [
                {field: doc[field] for field in fields if field in doc} for doc in docs
            ]
        return {"docs": docs}

    def get(self, doc_id, *, check=False, default_value=None):
        if doc_id not in self.docs:
            if check:
                raise KeyError(doc_id)
            return default_value
        return dict(self.docs[doc_id])

    def save(self, doc):
        current = self.docs.get(doc["_id"])
        if doc.get("_rev") != (current or {}).get("_rev"):
            raise ConflictError("Document update conflict.")
        generation = int(current["_rev"].split("-")[0]) + 1 if current else 1
        self.docs[doc["_id"]] = {**doc, "_rev": f"{generation}-fake"}
        return doc["_id"], True, self.docs[doc["_id"]]["_rev"]

    @staticmethod
    def _matches(doc, selector):
        for key, expected in selector.items():
            if isinstance(expected, dict) and "$exists" in expected:
                exists = key in doc
                if exists != expected["$exists"]:
                    return False
            elif doc.get(key) != expected:
                return False
        return True


class BrokenDatabase(FakeDatabase):
    def find(self, selector, fields=None, limit=None):
        raise RuntimeError("database read failed")

    def get(self, doc_id, *, check=False, default_value=None):
        raise RuntimeError("database read failed")

    def save(self, doc):
        raise RuntimeError("database write failed")


class WriteFailingDatabase(FakeDatabase):
    def save(self, doc):
        raise RuntimeError("database write failed")


class RacingDatabase(FakeDatabase):
    """Another writer updates each doc between this client's read and write."""

    def find(self, selector, fields=None, limit=None):
        res = super().find(selector, fields=fields, limit=limit)
        for doc_id, doc in self.docs.items():
            self.docs[doc_id] = {
                **doc,
                "failure_modes": [*doc.get("failure_modes", []), "concurrent edit"],
                "_rev": "2-other-writer",
            }
        return res


@pytest.fixture
def fake_fm_db():
    db = FakeDatabase(
        [
            {
                "_id": "fm:pump",
                "asset_class": "pump",
                "failure_modes": ["seal leakage", "impeller wear"],
                "exhaustive": False,
                "source": "synthetic sample",
            },
        ]
    )
    with patch("servers.fmsr.main.fm_db", db):
        yield db


@pytest.fixture
def messy_fm_db():
    """Stored names are not normalised and _ids do not follow the fm:<class> form."""
    db = FakeDatabase(
        [
            {
                "_id": "fm:Hydraulic Pump",
                "asset_class": "Hydraulic Pump",
                "failure_modes": ["cavitation"],
            },
            {
                "_id": "a1b2c3",
                "asset_class": "CO2 Compressor",
                "failure_modes": ["valve leakage"],
            },
            {
                "_id": "fm:co compressor",
                "asset_class": "CO Compressor",
                "failure_modes": ["carbon fouling"],
            },
            {
                "_id": "fm:Chiller",
                "asset_class": "Chiller",
                "failure_modes": ["refrigerant leak"],
            },
            {
                "_id": "fm:fan",
                "asset_class": "Blower",
                "failure_modes": ["blade imbalance"],
            },
        ]
    )
    with patch("servers.fmsr.main.fm_db", db):
        yield db


@pytest.fixture
def duplicate_fm_db():
    """Two stored names that normalise to the same class."""
    db = FakeDatabase(
        [
            {"_id": "b", "asset_class": "Pump", "failure_modes": ["from Pump"]},
            {"_id": "a", "asset_class": "pump", "failure_modes": ["from pump"]},
        ]
    )
    with patch("servers.fmsr.main.fm_db", db):
        yield db


@pytest.fixture
def write_failing_fm_db():
    db = WriteFailingDatabase(
        [{"_id": "fm:pump", "asset_class": "pump", "failure_modes": ["seal leakage"]}]
    )
    with patch("servers.fmsr.main.fm_db", db):
        yield db


@pytest.fixture
def racing_fm_db():
    db = RacingDatabase(
        [{"_id": "fm:pump", "asset_class": "pump", "failure_modes": ["seal leakage"]}]
    )
    with patch("servers.fmsr.main.fm_db", db):
        yield db


@pytest.fixture
def empty_fm_db():
    db = FakeDatabase()
    with patch("servers.fmsr.main.fm_db", db):
        yield db


@pytest.fixture
def broken_fm_db():
    db = BrokenDatabase()
    with patch("servers.fmsr.main.fm_db", db):
        yield db
