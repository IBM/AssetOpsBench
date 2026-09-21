"""Tests for swapping one MCP server slot for a remote service.

The swap is an environment variable rather than a code change so both arms of
the comparison run from one build. These pin the two things that would quietly
invalidate such a run: a credential leaking into an artifact, and a typo
silently changing the server set.
"""

from __future__ import annotations

from agent.mcp_servers import (
    DEFAULT_SERVER_PATHS,
    RemoteMCPServer,
    apply_env_overrides,
    describe_servers,
    is_remote,
)


def test_no_override_leaves_every_slot_local():
    resolved = apply_env_overrides(dict(DEFAULT_SERVER_PATHS), env={})

    assert resolved == DEFAULT_SERVER_PATHS
    assert not any(is_remote(spec) for spec in resolved.values())


def test_url_env_redirects_only_the_named_slot():
    resolved = apply_env_overrides(
        dict(DEFAULT_SERVER_PATHS),
        env={
            "ASSETOPS_MCP_URL_TSFM": "https://api.tsfm.ai/mcp",
            "ASSETOPS_MCP_TOKEN_TSFM": "secret-token",
        },
    )

    assert is_remote(resolved["tsfm"])
    assert resolved["tsfm"].url == "https://api.tsfm.ai/mcp"
    # Every other domain must stay local, or the comparison has more than one
    # variable in it.
    for name in ("iot", "utilities", "fmsr", "wo", "vibration"):
        assert not is_remote(resolved[name])


def test_token_becomes_a_bearer_header():
    spec = RemoteMCPServer(url="https://api.tsfm.ai/mcp", token="secret-token")

    assert spec.headers() == {"Authorization": "Bearer secret-token"}


def test_no_token_sends_no_auth_header():
    spec = RemoteMCPServer(url="https://example.invalid/mcp")

    assert spec.headers() is None


def test_describe_servers_never_reveals_the_token():
    resolved = apply_env_overrides(
        dict(DEFAULT_SERVER_PATHS),
        env={
            "ASSETOPS_MCP_URL_TSFM": "https://api.tsfm.ai/mcp",
            "ASSETOPS_MCP_TOKEN_TSFM": "super-secret-value",
        },
    )

    described = describe_servers(resolved)
    blob = repr(described)

    assert "super-secret-value" not in blob
    assert "https://api.tsfm.ai/mcp" in described["tsfm"]
    assert "token: set" in described["tsfm"]


def test_unknown_server_name_is_ignored_not_added():
    # The benchmark's server set is fixed. A typo must not introduce a seventh
    # domain that silently changes the tool surface under test.
    resolved = apply_env_overrides(
        dict(DEFAULT_SERVER_PATHS),
        env={"ASSETOPS_MCP_URL_TSMF": "https://typo.invalid/mcp"},
    )

    assert set(resolved) == set(DEFAULT_SERVER_PATHS)
    assert not any(is_remote(spec) for spec in resolved.values())


def test_blank_token_is_treated_as_absent():
    resolved = apply_env_overrides(
        dict(DEFAULT_SERVER_PATHS),
        env={
            "ASSETOPS_MCP_URL_TSFM": "https://api.tsfm.ai/mcp",
            "ASSETOPS_MCP_TOKEN_TSFM": "   ",
        },
    )

    assert resolved["tsfm"].token is None
    assert resolved["tsfm"].headers() is None
