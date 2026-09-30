import json

import pytest
from unittest.mock import MagicMock, patch


def _fmsr_llm_available() -> bool:
    """True when FMSR_MODEL_ID names a router whose credentials are present."""
    try:
        from servers.fmsr.main import _llm_available
    except Exception:  # noqa: BLE001 - collection must not fail on import
        return False
    return bool(_llm_available)


requires_fmsr_llm = pytest.mark.skipif(
    not _fmsr_llm_available(),
    reason=(
        "FMSR LLM not configured: set FMSR_MODEL_ID to a tokenrouter/ or "
        "litellm_proxy/ model and that router's credentials"
    ),
)


async def call_tool(mcp_instance, tool_name: str, args: dict) -> dict:
    """Helper: call an MCP tool and return parsed JSON response."""
    contents, _ = await mcp_instance.call_tool(tool_name, args)
    return json.loads(contents[0].text)


class FakeDatabase:
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

    def get(self, doc_id, *, check=False, default_value=None):
        if doc_id not in self.docs:
            if check:
                raise KeyError(doc_id)
            return default_value
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


class BrokenDatabase(FakeDatabase):
    def find(self, selector, fields=None, limit=None):
        raise RuntimeError("database read failed")

    def get(self, doc_id, *, check=False, default_value=None):
        raise RuntimeError("database read failed")

    def save(self, doc):
        raise RuntimeError("database write failed")


@pytest.fixture
def no_llm():
    """Simulate an unconfigured or unreachable FMSR LLM."""
    with patch("servers.fmsr.main._llm_available", False):
        yield


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
def empty_fm_db():
    db = FakeDatabase()
    with patch("servers.fmsr.main.fm_db", db):
        yield db


@pytest.fixture
def broken_fm_db():
    db = BrokenDatabase()
    with patch("servers.fmsr.main.fm_db", db):
        yield db


@pytest.fixture
def mock_failure_mode_generation():
    """Patch failure-mode generation so tests do not call the LLM."""
    mock = MagicMock(
        return_value=[
            "bearing wear",
            "seal leakage",
            "motor overheating",
        ]
    )
    with patch("servers.fmsr.main._call_failure_mode_generation", mock):
        with patch("servers.fmsr.main._llm_available", True):
            yield mock
