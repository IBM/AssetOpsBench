"""Shared fixtures and helpers for robot MCP server tests."""

import json
import os
import random

from dotenv import load_dotenv
import pytest
from unittest.mock import patch

load_dotenv()


# --- CouchDB availability ---


def _couchdb_reachable() -> bool:
    url = os.environ.get("COUCHDB_URL")
    if not url:
        return False
    try:
        import requests
        requests.get(url, timeout=2)
        return True
    except Exception:
        return False


requires_couchdb = pytest.mark.skipif(
    not _couchdb_reachable(),
    reason="CouchDB not reachable (set COUCHDB_URL and ensure CouchDB is running)",
)


# --- Fixtures ---


@pytest.fixture
def simulator():
    """Fresh PhysicalStateSimulator with seed=42."""
    from servers.robot.simulator import PhysicalStateSimulator
    return PhysicalStateSimulator(seed=42)


@pytest.fixture(autouse=True)
def reset_simulator_rng():
    """Reset module-level simulator RNG before every test.

    open_panel() and simulate_read_gauge() consume RNG state; without a reset
    the outcome of tests depends on execution order.
    """
    import servers.robot.main as robot_main
    robot_main._simulator._rng = random.Random(42)
    yield
    robot_main._simulator._rng = random.Random(42)


@pytest.fixture
def mock_db():
    """Patch module-level `db` in robot main with a Mock."""
    import servers.robot.main as robot_main
    with patch("servers.robot.main.db") as mock:
        yield mock


@pytest.fixture
def no_db():
    """Patch module-level `db` to None (simulate disconnected IoT CouchDB)."""
    with patch("servers.robot.main.db", None):
        yield


@pytest.fixture
def no_wo_db():
    """Patch _get_wo_db() to return None (simulate disconnected WO CouchDB)."""
    with patch("servers.robot.main._get_wo_db", return_value=None):
        yield


# --- Tool call helper ---


async def call_tool(mcp_instance, tool_name: str, args: dict) -> dict:
    """Call an MCP tool and return the parsed JSON response."""
    contents, _ = await mcp_instance.call_tool(tool_name, args)
    return json.loads(contents[0].text)
