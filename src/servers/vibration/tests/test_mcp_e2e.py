"""MCP stdio transport E2E tests for the vibration server.

Environment-agnostic and CI-safe. Each test spawns a real server subprocess
and communicates over stdin/stdout via MCP JSON-RPC (stdio transport).
No CouchDB, API keys, or external services required.

Place: src/servers/vibration/tests/test_mcp_e2e.py

Usage:
    pytest src/servers/vibration/tests/test_mcp_e2e.py -v
    pytest src/servers/vibration/tests/test_mcp_e2e.py -v -k "sc04"
    pytest -n auto src/servers/vibration/tests/test_mcp_e2e.py

Design decisions
----------------
1. sys.executable     -- env-agnostic subprocess spawn; no PATH dependency
2. anyio.fail_after   -- propagates cancellation through anyio task groups,
                         terminating the server subprocess on deadlock instead
                         of leaving a zombie process (unlike asyncio.wait_for)
3. _SENSITIVE_KEYS    -- LLM credentials stripped before passing env to server;
                         prevents billable API calls if server logic changes
4. OTEL_SDK_DISABLED  -- disables OTEL in the subprocess; avoids concurrent
                         JSONL file writes under pytest-xdist parallel runs
5. vibration_session  -- function-scoped pytest fixture (scope="function");
                         each test gets its own subprocess, eliminating
                         cross-test state pollution
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import anyio
import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, get_default_environment, stdio_client

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def _find_repo_root(start: Path) -> Path:
    """Locate AssetOpsBench root whether this file is at root or in a suite folder."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "src" / "servers" / "vibration" / "main.py"
        ).is_file():
            return candidate
    raise RuntimeError(
        "Could not locate AssetOpsBench repo root. Place this test under the "
        "repository root or a child directory containing pyproject.toml and src/."
    )


_REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
_SRC = _REPO_ROOT / "src"
_DEADLINE = 20  # seconds per operation; anyio enforces this at subprocess level

# LLM credentials that must not reach the test subprocess.
# Prevents accidental billable API calls if server-side logic is ever changed.
_SENSITIVE_KEYS: frozenset[str] = frozenset({
    "WATSONX_APIKEY", "WATSONX_PROJECT_ID", "WATSONX_URL",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LITELLM_API_KEY",
    "LITELLM_BASE_URL", "COHERE_API_KEY", "AZURE_API_KEY",
    "AZURE_API_BASE", "HUGGINGFACE_API_KEY",
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _server_params(
    module: str,
    *,
    otel_dir: Path | None = None,
) -> StdioServerParameters:
    """Build StdioServerParameters for a given server module.

    Uses sys.executable (env-agnostic), strips LLM credentials (air-gap),
    and disables OTEL SDK by default to prevent JSONL file collisions under
    pytest-xdist. Pass otel_dir to enable per-process OTEL tracing instead.
    """
    env = get_default_environment()
    env["PYTHONPATH"] = str(_SRC)

    for key in _SENSITIVE_KEYS:
        env.pop(key, None)

    if otel_dir is not None:
        env.pop("OTEL_SDK_DISABLED", None)
        env["OTEL_TRACES_FILE"] = str(otel_dir / "traces.jsonl")
    else:
        env["OTEL_SDK_DISABLED"] = "true"

    return StdioServerParameters(
        command=sys.executable,
        args=["-m", module],
        env=env,
    )


@asynccontextmanager
async def _mcp_session(
    module: str,
    *,
    otel_dir: Path | None = None,
) -> AsyncGenerator[ClientSession, None]:
    """Open an MCP session with anyio deadline enforcement.

    anyio.fail_after is used instead of asyncio.wait_for because anyio's
    cancel scopes propagate through anyio task groups -- which is how
    stdio_client manages its read/write streams internally. This ensures
    the server subprocess is terminated rather than orphaned on timeout.

    The deadline covers initialization only. Per-call timeouts are applied
    at the call site via asyncio.wait_for.
    """
    params = _server_params(module, otel_dir=otel_dir)
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            with anyio.fail_after(_DEADLINE):
                await session.initialize()
            yield session


def _parse_result(result) -> dict:
    """Extract the first JSON dict from an MCP tool call result."""
    for content in result.content:
        text = getattr(content, "text", None) or str(content)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return {"_raw": text}
    return {}


# ---------------------------------------------------------------------------
# Pytest fixture -- function-scoped (fresh subprocess per test)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
async def vibration_session() -> AsyncGenerator[ClientSession, None]:
    """Each test gets a fresh vibration server process.

    Isolation guarantee: no in-memory data store state leaks between tests.
    Teardown: stdio_client.__aexit__ terminates the subprocess via anyio
    cancel scope when the fixture goes out of scope.
    """
    async with _mcp_session("servers.vibration.main") as session:
        yield session


# ---------------------------------------------------------------------------
# vibration server -- no CouchDB required
# ---------------------------------------------------------------------------


class TestVibrationMCPProtocol:
    """E2E tests over real MCP stdio transport to the vibration server.

    SC-01 to SC-05 use the vibration_session fixture (function-scoped).
    SC-06 exercises the full lifecycle directly to validate teardown.
    """

    @pytest.mark.anyio
    async def test_sc01_tool_listing(self, vibration_session: ClientSession) -> None:
        """SC-01: Server starts and exposes expected tools over stdio."""
        tools = await asyncio.wait_for(vibration_session.list_tools(), timeout=_DEADLINE)
        names = {t.name for t in tools.tools}
        expected = {
            "get_vibration_data",
            "list_vibration_sensors",
            "compute_fft_spectrum",
            "compute_envelope_spectrum",
            "assess_vibration_severity",
            "calculate_bearing_frequencies",
            "list_known_bearings",
            "diagnose_vibration",
        }
        assert expected <= names, f"Missing tools: {expected - names}"

    @pytest.mark.anyio
    async def test_sc02_static_tool_happy_path(self, vibration_session: ClientSession) -> None:
        """SC-02: list_known_bearings returns static database without CouchDB."""
        result = await asyncio.wait_for(
            vibration_session.call_tool("list_known_bearings", {}),
            timeout=_DEADLINE,
        )
        data = _parse_result(result)
        assert "bearings" in data, f"Expected 'bearings' key, got: {data}"
        assert len(data["bearings"]) >= 5
        names = [b["name"] for b in data["bearings"]]
        assert any("6205" in n for n in names), f"6205 not found in {names}"

    @pytest.mark.anyio
    async def test_sc03_iso_severity_zone_classification(self, vibration_session: ClientSession) -> None:
        """SC-03: assess_vibration_severity classifies ISO 10816 zones correctly."""
        zone_d = _parse_result(await asyncio.wait_for(
            vibration_session.call_tool("assess_vibration_severity", {"rms_velocity_mm_s": 50.0}),
            timeout=_DEADLINE,
        ))
        zone_a = _parse_result(await asyncio.wait_for(
            vibration_session.call_tool("assess_vibration_severity", {"rms_velocity_mm_s": 0.5}),
            timeout=_DEADLINE,
        ))
        assert zone_d.get("iso_zone") == "D", f"Expected D, got: {zone_d}"
        assert zone_a.get("iso_zone") == "A", f"Expected A, got: {zone_a}"

    @pytest.mark.anyio
    async def test_sc04_pydantic_boundary_missing_required_field(
        self, vibration_session: ClientSession
    ) -> None:
        """SC-04: Pydantic rejects diagnose_vibration when data_id is absent.

        Primary MCP schema validation regression test. The FastMCP Pydantic
        layer must catch the missing field and return a structured validation
        error rather than crashing the subprocess.

        Note: accesses result.content[0].text directly (not _parse_result) because
        the assertion needs raw keyword search on the error string, not a parsed
        dict -- the validation error response is not guaranteed to be valid JSON.
        """
        result = await asyncio.wait_for(
            vibration_session.call_tool(
                "diagnose_vibration",
                {
                    "velocity_rms": 15.5,
                    "machine_class": "Class I",
                    "envelope_peak_freqs": [50.0, 100.0, 150.0],
                    "bpfo": 50.0,
                    "bpfi": 70.0,
                    "ftf": 10.0,
                    "bsf": 30.0,
                },
            ),
            timeout=_DEADLINE,
        )
        raw_text = result.content[0].text if result.content else ""
        assert "data_id" in raw_text, f"Expected 'data_id' in error, got: {raw_text!r}"
        assert any(
            kw in raw_text.lower()
            for kw in ("field required", "validation error", "missing")
        ), f"Expected validation error message, got: {raw_text!r}"

    @pytest.mark.anyio
    async def test_sc05_data_not_found_returns_error_not_crash(
        self, vibration_session: ClientSession
    ) -> None:
        """SC-05: compute_fft_spectrum with unknown data_id returns error dict.

        Verifies application-level errors (data not found) are returned as
        structured JSON error responses, not unhandled subprocess crashes.
        """
        result = await asyncio.wait_for(
            vibration_session.call_tool(
                "compute_fft_spectrum", {"data_id": "__nonexistent__"}
            ),
            timeout=_DEADLINE,
        )
        data = _parse_result(result)
        assert "error" in data, f"Expected error key, got: {data}"

    @pytest.mark.anyio
    async def test_sc06_session_teardown_exits_cleanly(self) -> None:
        """SC-06: Full session lifecycle completes within 2x_DEADLINE seconds.

        Regression test for FIND-010 (async teardown deadlock). Uses the
        context manager directly (not the fixture) to test the complete
        lifecycle. anyio.fail_after wraps the entire block -- if teardown
        hangs, anyio cancels it via the stdio_client task group rather
        than leaving a zombie process behind.
        """
        exited = False
        with anyio.fail_after(_DEADLINE * 2):
            async with _mcp_session("servers.vibration.main") as session:
                await asyncio.wait_for(session.list_tools(), timeout=_DEADLINE)
        exited = True
        assert exited, "Session teardown caused a deadlock"
