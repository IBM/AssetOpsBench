"""Where each MCP server lives, and how to swap one for a remote service.

AssetOpsBench normally launches its six servers as local stdio processes. A
benchmark that asks "what happens if this domain is served by someone else's
MCP server instead of ours" needs one slot to point somewhere else without
forking the runner, so a server spec is either:

* a local entry point (``"tsfm-mcp-server"``), launched as
  ``uv run --directory <repo> <entry-point>``; or
* a :class:`RemoteMCPServer`, reached over Streamable HTTP with an optional
  bearer token.

Swapping is an environment variable, not a code change, so the same build runs
both arms of the comparison::

    ASSETOPS_MCP_URL_TSFM=https://api.tsfm.ai/mcp \\
    ASSETOPS_MCP_TOKEN_TSFM=$TSFM_API_KEY \\
        uv run python -m benchmark.scenario_suite_runner ...

The token is read from the environment and never written to a trajectory, a
report or a span: :func:`describe_servers` reports the endpoint and whether a
token was supplied, never its value.

A remote server is not a drop-in for a local one. Tool names, schemas and
coverage all differ, so scenarios calling tools the replacement does not expose
will fail. That is the measurement, not a defect, but read
:func:`describe_servers` output before a sweep so the failures are expected
rather than discovered afterwards.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DEFAULT_SERVER_PATHS",
    "RemoteMCPServer",
    "ServerSpec",
    "apply_env_overrides",
    "describe_servers",
    "is_remote",
]

_URL_ENV_PREFIX = "ASSETOPS_MCP_URL_"
_TOKEN_ENV_PREFIX = "ASSETOPS_MCP_TOKEN_"
# A remote domain server answers real queries, so it gets a request budget in
# the same range as the local ones rather than an HTTP client's short default.
_DEFAULT_TIMEOUT_S = float(os.environ.get("ASSETOPS_MCP_HTTP_TIMEOUT", 120))
_DEFAULT_SSE_READ_TIMEOUT_S = float(
    os.environ.get("ASSETOPS_MCP_SSE_READ_TIMEOUT", 300)
)


@dataclass(frozen=True)
class RemoteMCPServer:
    """An MCP server reached over Streamable HTTP instead of stdio.

    ``token`` is a bearer credential. It is held here only to build the request
    header; never log, serialise or persist this object's ``token``.
    """

    url: str
    token: str | None = None
    timeout_s: float = _DEFAULT_TIMEOUT_S
    sse_read_timeout_s: float = _DEFAULT_SSE_READ_TIMEOUT_S

    def headers(self) -> dict[str, str] | None:
        """Auth headers for this endpoint, or None when it is unauthenticated."""
        if not self.token:
            return None
        return {"Authorization": f"Bearer {self.token}"}

    def redacted(self) -> str:
        """A log-safe description: endpoint plus whether a token was supplied."""
        return f"{self.url} (token: {'set' if self.token else 'none'})"


ServerSpec = "Path | str | RemoteMCPServer"

# Maps MCP-server names to either a uv entry-point name (str), a script Path,
# or a RemoteMCPServer. Entry-point names are invoked as ``uv run <name>``.
DEFAULT_SERVER_PATHS: dict[str, "Path | str | RemoteMCPServer"] = {
    "iot": "iot-mcp-server",
    "utilities": "utilities-mcp-server",
    "fmsr": "fmsr-mcp-server",
    "tsfm": "tsfm-mcp-server",
    "wo": "wo-mcp-server",
    "vibration": "vibration-mcp-server",
}


def is_remote(spec: object) -> bool:
    """True when *spec* names a remote endpoint rather than a local process."""
    return isinstance(spec, RemoteMCPServer)


def apply_env_overrides(
    servers: dict[str, "Path | str | RemoteMCPServer"],
    env: "os._Environ[str] | dict[str, str] | None" = None,
) -> dict[str, "Path | str | RemoteMCPServer"]:
    """Return *servers* with any ``ASSETOPS_MCP_URL_<NAME>`` slots redirected.

    The name is matched case-insensitively, so ``ASSETOPS_MCP_URL_TSFM``
    redirects the ``tsfm`` slot. A URL for a server that is not in *servers* is
    ignored rather than added: the benchmark's server set is fixed, and a typo
    should not silently introduce a seventh domain.
    """
    source = os.environ if env is None else env
    resolved = dict(servers)
    for name in list(resolved):
        url = source.get(f"{_URL_ENV_PREFIX}{name.upper()}")
        if not url:
            continue
        resolved[name] = RemoteMCPServer(
            url=url.strip(),
            token=(source.get(f"{_TOKEN_ENV_PREFIX}{name.upper()}") or "").strip()
            or None,
        )
    return resolved


def describe_servers(
    servers: dict[str, "Path | str | RemoteMCPServer"],
) -> dict[str, str]:
    """A log-safe ``{name: description}`` map, with no credentials in it."""
    return {
        name: (spec.redacted() if is_remote(spec) else f"local: {spec}")
        for name, spec in servers.items()
    }
