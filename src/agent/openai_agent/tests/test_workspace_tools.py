"""Tests for OpenAI-agent local workspace and web function tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.openai_agent.workspace_tools import (
    WorkspaceToolFactory,
    _public_http_url,
)


def test_build_tools_default_is_empty() -> None:
    assert WorkspaceToolFactory(None).build_tools() == []


def test_build_tools_matches_permissions(tmp_path: Path) -> None:
    tools = WorkspaceToolFactory(tmp_path).build_tools(
        allow_files=True,
        allow_bash=True,
        allow_web=True,
    )

    assert {tool.name for tool in tools} == {
        "delete_file",
        "list_files",
        "read_file",
        "replace_in_file",
        "run_bash",
        "search_files",
        "web_fetch",
        "web_search",
        "write_file",
    }


def test_build_tools_requires_workspace_for_local_capabilities() -> None:
    factory = WorkspaceToolFactory(None)

    with pytest.raises(ValueError, match="workspace_dir is required"):
        factory.build_tools(allow_files=True)
    with pytest.raises(ValueError, match="workspace_dir is required"):
        factory.build_tools(allow_bash=True)
    with pytest.raises(ValueError, match="workspace_dir is required"):
        factory.build_tools(allow_edit=True)


def test_file_tools_are_scoped_to_workspace(tmp_path: Path) -> None:
    factory = WorkspaceToolFactory(tmp_path)
    factory.write_file("notes/example.txt", "alpha\nbeta\n")

    assert "notes/example.txt" in factory.list_files()
    assert "alpha" in factory.read_file("notes/example.txt")
    assert "notes/example.txt:2:beta" in factory.search_files("beta")

    factory.replace_in_file("notes/example.txt", "beta", "gamma")
    assert "gamma" in factory.read_file("notes/example.txt")
    factory.delete_file("notes/example.txt")
    assert not (tmp_path / "notes" / "example.txt").exists()


def test_file_tools_reject_workspace_escape(tmp_path: Path) -> None:
    factory = WorkspaceToolFactory(tmp_path / "workspace")
    factory.workspace_dir.mkdir()

    with pytest.raises(ValueError, match="escapes the workspace"):
        factory.read_file("../outside.txt")
    with pytest.raises(ValueError, match="escapes the workspace"):
        factory.write_file("../outside.txt", "no")


def test_delete_file_unlinks_symlink_without_deleting_target(tmp_path: Path) -> None:
    factory = WorkspaceToolFactory(tmp_path)
    target = tmp_path / "target.txt"
    target.write_text("keep", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(target)

    factory.delete_file("link.txt")

    assert not link.exists()
    assert target.read_text(encoding="utf-8") == "keep"


def test_delete_file_rejects_symlink_parent_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("keep", encoding="utf-8")
    (workspace / "outside-link").symlink_to(outside, target_is_directory=True)
    factory = WorkspaceToolFactory(workspace)

    with pytest.raises(ValueError, match="escapes the workspace"):
        factory.delete_file("outside-link/secret.txt")

    assert (outside / "secret.txt").read_text(encoding="utf-8") == "keep"


def test_run_bash_uses_workspace_and_scrubs_router_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TOKENROUTER_API_KEY", "secret-value")
    factory = WorkspaceToolFactory(tmp_path)

    output = factory.run_bash(
        'pwd; printf "token=%s" "${TOKENROUTER_API_KEY-unset}"; touch created.txt'
    )

    assert str(tmp_path) in output
    assert "token=unset" in output
    assert "secret-value" not in output
    assert (tmp_path / "created.txt").exists()


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/example",
        "http://127.0.0.1/",
        "http://localhost/",
        "http://user:password@example.com/",
    ],
)
def test_public_http_url_rejects_unsafe_destinations(url: str) -> None:
    with pytest.raises(ValueError):
        _public_http_url(url)
