import json
import os

import pytest
from unittest.mock import MagicMock, patch

requires_watsonx = pytest.mark.skipif(
    os.environ.get("WATSONX_APIKEY") is None,
    reason="WatsonX not available (set WATSONX_APIKEY)",
)


async def call_tool(mcp_instance, tool_name: str, args: dict) -> dict:
    """Helper: call an MCP tool and return parsed JSON response."""
    contents, _ = await mcp_instance.call_tool(tool_name, args)
    return json.loads(contents[0].text)


class FakeCouchDB:
    def __init__(self, docs=None):
        self.docs = {doc["_id"]: dict(doc) for doc in docs or []}

    def find(self, selector, fields=None, limit=None):
        docs = [doc for doc in self.docs.values() if self._matches(doc, selector)]
        if limit is not None:
            docs = docs[:limit]
        if fields is not None:
            docs = [
                {field: doc[field] for field in fields if field in doc} for doc in docs
            ]
        return {"docs": docs}

    def get(self, doc_id):
        if doc_id not in self.docs:
            raise KeyError(doc_id)
        return dict(self.docs[doc_id])

    def save(self, doc):
        self.docs[doc["_id"]] = dict(doc)

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


@pytest.fixture
def no_llm():
    """Simulate missing WatsonX credentials."""
    with patch("servers.fmsr.main._llm_available", False):
        yield


@pytest.fixture
def fake_fm_db():
    db = FakeCouchDB(
        [
            {
                "_id": "fm:chiller",
                "doctype": "failure_mode",
                "asset_class": "chiller",
                "failure_modes": ["Compressor overheating", "Condenser fouling"],
                "exhaustive": False,
                "source": "test",
            },
            {
                "_id": "fm:pump",
                "doctype": "failure_mode",
                "asset_class": "pump",
                "failure_modes": ["seal leakage"],
                "exhaustive": False,
                "source": "test",
            },
        ]
    )
    with patch("servers.fmsr.main.fm_db", db):
        yield db


@pytest.fixture
def empty_fm_db():
    db = FakeCouchDB()
    with patch("servers.fmsr.main.fm_db", db):
        yield db


@pytest.fixture
def fake_catalog_db():
    db = FakeCouchDB(
        [
            {
                "_id": "catalog:sensor:dp",
                "sensor": "differential pressure",
                "description": "pressure difference",
            },
            {
                "_id": "catalog:sensor:temp",
                "sensor": "temperature",
                "description": "temperature",
            },
            {
                "_id": "catalog:fm:seal",
                "category": "rotating equipment",
                "failure_mode": "seal leakage",
                "description": "seal loses containment",
            },
        ]
    )
    with patch("servers.fmsr.main.catalog_db", db):
        yield db


@pytest.fixture
def mock_relevancy_chain():
    """Patch _call_relevancy so it always returns 'Yes' without calling the LLM."""
    mock = MagicMock(
        return_value={
            "answer": "Yes",
            "reason": "Relevant sensor",
            "temporal_behavior": "Increases",
        }
    )
    with patch("servers.fmsr.main._call_relevancy", mock):
        with patch("servers.fmsr.main._llm_available", True):
            yield mock


@pytest.fixture
def mock_asset2fm_chain():
    """Patch _call_asset2fm to return a fixed failure mode list."""
    mock = MagicMock(return_value=["Fan Failure", "Belt Wear"])
    with patch("servers.fmsr.main._call_asset2fm", mock):
        with patch("servers.fmsr.main._llm_available", True):
            yield mock


@pytest.fixture
def mock_asset2fm_extend_chain():
    """Patch _call_asset2fm_extend to return duplicate and new failure modes."""
    mock = MagicMock(return_value=["Seal Leakage", "Bearing Wear"])
    with patch("servers.fmsr.main._call_asset2fm_extend", mock):
        with patch("servers.fmsr.main._llm_available", True):
            yield mock
