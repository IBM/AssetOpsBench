# MCP Protocol E2E Test Suite

**Contribution by Flamehaven | CAS-2026-05-ASSET-004 | 2026-05-06**

---

## What This Is

A pytest-native, environment-agnostic end-to-end test suite for the **MCP stdio transport layer** of AssetOpsBench.

### The Coverage Gap This Fills

All existing tests in `src/servers/vibration/tests/test_tools.py` use the following pattern:

```python
result = await call_tool(mcp, "compute_fft_spectrum", {"data_id": data_id})
```

This calls `mcp.call_tool()` directly in-process. It tests **tool logic** but never touches the MCP wire protocol — no subprocess is spawned, no JSON-RPC frames are exchanged, no stdio streams are opened.

This suite tests the actual transport stack:

```
pytest process
  -> spawns servers.vibration.main subprocess via sys.executable
  -> communicates over stdin/stdout via MCP JSON-RPC (stdio transport)
  -> validates Pydantic schema enforcement, error responses, clean teardown
```

---

## Requirements

- Python 3.12+
- `mcp[cli]` and `fastmcp` installed (included in `pyproject.toml` core deps)
- `pytest`, `pytest-anyio`, `anyio` installed (included in `[dependency-groups].dev`)
- **No CouchDB, no API keys, no external services required**
- **Subprocess spawning required** — each test spawns a real server process via `sys.executable`. Sandboxed environments that block subprocess creation or Windows named pipe access (e.g., restricted container runtimes, some in-browser execution environments) will fail. Standard CI runners (GitHub Actions ubuntu-latest, Windows Server) are confirmed compatible.

---

## Running

```bash
# All 6 scenarios
pytest src/servers/vibration/tests/test_mcp_e2e.py -v

# Single scenario by keyword
pytest src/servers/vibration/tests/test_mcp_e2e.py -v -k "sc04"

# Parallel (xdist-safe)
pytest -n auto src/servers/vibration/tests/test_mcp_e2e.py
```

Expected output:

```
src/servers/vibration/tests/test_mcp_e2e.py::TestVibrationMCPProtocol::test_sc01_tool_listing[asyncio]                              PASSED
src/servers/vibration/tests/test_mcp_e2e.py::TestVibrationMCPProtocol::test_sc02_static_tool_happy_path[asyncio]                    PASSED
src/servers/vibration/tests/test_mcp_e2e.py::TestVibrationMCPProtocol::test_sc03_iso_severity_zone_classification[asyncio]           PASSED
src/servers/vibration/tests/test_mcp_e2e.py::TestVibrationMCPProtocol::test_sc04_pydantic_boundary_missing_required_field[asyncio]   PASSED
src/servers/vibration/tests/test_mcp_e2e.py::TestVibrationMCPProtocol::test_sc05_data_not_found_returns_error_not_crash[asyncio]     PASSED
src/servers/vibration/tests/test_mcp_e2e.py::TestVibrationMCPProtocol::test_sc06_session_teardown_exits_cleanly[asyncio]             PASSED

6 passed; runtime varies by local subprocess startup cost
```

---

## Engineering Design Decisions

### 1. `sys.executable` — Environment Agnostic

The original script used `command="python"`, which resolves from OS `PATH` and may point to a system Python without `mcp` installed, causing `ModuleNotFoundError` in the subprocess.

```python
# Before (fragile)
StdioServerParameters(command="python", args=["-m", "servers.vibration.main"])

# After (environment-agnostic)
StdioServerParameters(command=sys.executable, args=["-m", "servers.vibration.main"])
```

`sys.executable` always resolves to the Python running pytest — venv, conda, or pyenv.

### 2. `anyio.fail_after` — Zombie Process Prevention

`asyncio.wait_for` raises `TimeoutError` but does not propagate cancellation through anyio task groups. `stdio_client` uses anyio internally to manage its read/write streams and subprocess lifecycle. Using `anyio.fail_after` ensures the cancellation signal travels through the full anyio task group stack, terminating the server subprocess rather than leaving a zombie process.

```python
@asynccontextmanager
async def _mcp_session(module: str):
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            with anyio.fail_after(_DEADLINE):   # anyio cancel scope
                await session.initialize()
            yield session

# SC-06 wraps the full lifecycle including teardown:
with anyio.fail_after(_DEADLINE * 2):
    async with _mcp_session("servers.vibration.main") as session:
        ...
```

### 3. `_SENSITIVE_KEYS` — Air-Gapped Sandbox

`get_default_environment()` inherits the full OS environment including `WATSONX_APIKEY`, `OPENAI_API_KEY`, etc. If server logic is ever modified to call an LLM, tests would make billable API calls silently. All LLM credentials are stripped before passing the environment to the subprocess.

```python
_SENSITIVE_KEYS = frozenset({
    "WATSONX_APIKEY", "WATSONX_PROJECT_ID", "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY", "LITELLM_API_KEY", "LITELLM_BASE_URL", ...
})
for key in _SENSITIVE_KEYS:
    env.pop(key, None)
```

### 4. `OTEL_SDK_DISABLED=true` — xdist Parallel Safe

Under `pytest -n auto` (xdist), multiple test processes may write to the same `OTEL_TRACES_FILE` simultaneously. Setting `OTEL_SDK_DISABLED=true` in each subprocess's environment disables the OTEL SDK entirely, preventing concurrent JSONL file write collisions.

For tests that specifically need OTEL tracing, pass an `otel_dir` path to `_server_params` to get a per-process trace file:

```python
async with _mcp_session("servers.vibration.main", otel_dir=tmp_path) as session:
    ...
```

### 5. `vibration_session` Fixture — Per-Test Isolation

The function-scoped `vibration_session` fixture spawns a fresh server subprocess for each test. This eliminates in-memory data store state leaks between tests.

SC-06 deliberately does **not** use the fixture — it tests the complete lifecycle including teardown within a single `anyio.fail_after` scope.

---

## Test Scenarios

| ID | Name | What It Tests |
|---|---|---|
| SC-01 | Tool Listing | Server starts over stdio; all 8 expected tools are exposed |
| SC-02 | Static Tool Happy Path | `list_known_bearings` returns bearing DB without CouchDB |
| SC-03 | ISO Severity Classification | `assess_vibration_severity` returns correct ISO 10816 zones (A and D) |
| SC-04 | Pydantic Boundary | `diagnose_vibration` with missing `data_id` → `Field required`, no crash |
| SC-05 | Data Not Found | `compute_fft_spectrum` with unknown `data_id` → structured error dict |
| SC-06 | Teardown Regression | Full lifecycle within `anyio.fail_after(40s)` — FIND-010 deadlock guard |

---

## Extending to Other Servers

To add E2E coverage for another MCP server (e.g., `servers.utilities.main`), add a new class:

```python
class TestUtilitiesMCPProtocol:

    @pytest.fixture
    async def utilities_session(self):
        async with _mcp_session("servers.utilities.main") as session:
            yield session

    @pytest.mark.anyio
    async def test_sc01_tool_listing(self, utilities_session):
        tools = await asyncio.wait_for(utilities_session.list_tools(), timeout=_DEADLINE)
        names = {t.name for t in tools.tools}
        assert "convert_units" in names   # adjust to actual tool names
```

Servers requiring CouchDB (`servers.iot`, `servers.wo`) can be added with a skip guard:

```python
import os
requires_couchdb = pytest.mark.skipif(
    os.environ.get("COUCHDB_URL") is None,
    reason="CouchDB not available (set COUCHDB_URL)"
)

@requires_couchdb
@pytest.mark.anyio
async def test_sc01_tool_listing(self, iot_session): ...
```

---

## Relationship to CAS Audit Findings

| Finding | Scenario |
|---|---|
| FIND-006: Pydantic strict validation (PASS) | SC-04 |
| FIND-010: Async teardown deadlock (static risk) | SC-06 |
| FIND-002: DSP tools functional (confirmed) | SC-03 via ISO classification |
| FIND-005: MCP tools accessible over protocol | SC-01, SC-02 |

---

## Verified On

| Field | Value |
|---|---|
| Commit | `a928284b06411ceb9e31663d936cf8b342b15ca2` |
| OS | Windows 11 |
| Python | 3.12.10 |
| Initial result | 6 passed in 29.12s on 2026-05-06 |
| Latest local re-check | 6 passed in 18.54s on 2026-05-07 after repo-root locator hardening |

---

*Flamehaven Contribution | AssetOpsBench MCP E2E Suite v2 | 2026-05-06*
