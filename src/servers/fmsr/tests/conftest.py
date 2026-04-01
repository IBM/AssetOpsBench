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
    """Simulate missing WatsonX credentials."""
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


@pytest.fixture
def mock_dga_chain():
    """Patch _call_dga to return a fixed DGAInterpretationResult."""
    mock = MagicMock(
        return_value={
            "fault_type": "Partial Discharge",
            "r1": 0.1,
            "r2": 0.2,
            "r3": 0.3,
            "code": "0.1,0.2,0.3",
            "confidence": "High",
            "reasoning": "Based on gas ratios",
            "recommended_action": "Inspect insulation",
        }
    )
    with patch("servers.fmsr.main._call_dga", mock):
        with patch("servers.fmsr.main._llm_available", True):
            yield mock


@pytest.fixture
def mock_winding_chain():
    """Patch _call_winding to return a fixed WindingTemperatureResult."""
    mock = MagicMock(
        return_value={
            "thermal_status": "Normal",
            "hot_spot_rise_c": 45.0,
            "ageing_rate": 1.0,
            "alarm_active": False,
            "trip_active": False,
            "risk_level": "Low",
            "reasoning": "Within limits",
            "recommended_action": "None",
        }
    )
    with patch("servers.fmsr.main._call_winding", mock):
        with patch("servers.fmsr.main._llm_available", True):
            yield mock


@pytest.fixture
def mock_load_chain():
    """Patch _call_load to return a fixed LoadProfileResult."""
    mock = MagicMock(
        return_value={
            "load_mva": 50.0,
            "load_factor_pct": 80.0,
            "loading_status": "Normal",
            "current_imbalance_pct": 3.0,
            "neutral_current_flag": False,
            "reasoning": "Balanced load",
            "recommended_action": "None",
        }
    )
    with patch("servers.fmsr.main._call_load", mock):
        with patch("servers.fmsr.main._llm_available", True):
            yield mock