"""The skill library must land where the agent reads it, not beside it.

The failure this file exists to prevent: the library was copied into
``temp_base_dir`` while the sandbox exposed a *child* of that directory as
``/workspace``, so ``/workspace/skills`` never existed and every K1 run scored
as an unaided K0 run while being labelled K1.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from agent.stirrup_agent.skills_mount import (
    copy_skills_into,
    mount_path,
    resolve_skills_source,
    skills_prompt,
)


@pytest.fixture
def library(tmp_path: Path) -> Path:
    root = tmp_path / "library"
    (root / "repo-skills-router").mkdir(parents=True)
    (root / "repo-skills-router" / "SKILL.md").write_text("router\n")
    (root / "repo-skills" / "demo").mkdir(parents=True)
    (root / "repo-skills" / "demo" / "SKILL.md").write_text("demo\n")
    (root / "repo-skills" / "demo" / "__pycache__").mkdir()
    (root / "repo-skills" / "demo" / "__pycache__" / "x.pyc").write_text("junk")
    return root


def test_k1_without_a_library_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires a skill library"):
        resolve_skills_source(None, k_level="k1")


def test_k1_with_a_non_library_directory_raises(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="repo-skills-router"):
        resolve_skills_source(tmp_path / "empty", k_level="k1")


def test_k0_mounts_nothing_and_appends_nothing(library: Path) -> None:
    assert resolve_skills_source(None, k_level="k0") is None
    assert resolve_skills_source(library, k_level="k0") is None
    assert skills_prompt(None, k_level="k0") is None


def test_copy_lands_inside_the_exec_dir(library: Path, tmp_path: Path) -> None:
    base = tmp_path / "ws"
    exec_dir = base / "stirrup_agent" / "run-1" / "exec-1"
    exec_dir.mkdir(parents=True)

    copy_skills_into(library, exec_dir)

    # What the prompt promises the agent, relative to the exec dir.
    assert (exec_dir / "skills" / "repo-skills-router" / "SKILL.md").is_file()
    # And nothing beside it, which is where the old code put the library.
    assert not (base / "skills").exists()


def test_copy_drops_junk_and_reports_the_count(library: Path, tmp_path: Path) -> None:
    exec_dir = tmp_path / "exec"
    exec_dir.mkdir()

    assert copy_skills_into(library, exec_dir) == 2
    assert not (exec_dir / "skills" / "repo-skills" / "demo" / "__pycache__").exists()


def test_copy_is_idempotent(library: Path, tmp_path: Path) -> None:
    exec_dir = tmp_path / "exec"
    exec_dir.mkdir()
    copy_skills_into(library, exec_dir)
    stale = exec_dir / "skills" / "stale.md"
    stale.write_text("from a previous run")

    copy_skills_into(library, exec_dir)

    assert not stale.exists()


@pytest.mark.parametrize(
    ("backend", "expected"), [("docker", "/workspace/skills"), ("local", "skills")]
)
def test_prompt_names_the_router_at_the_mount(
    library: Path, backend: str, expected: str
) -> None:
    block = skills_prompt(library, k_level="k1", code_backend=backend)
    assert f"{expected}/repo-skills-router/SKILL.md" in block
    assert mount_path(backend) == expected


def test_recovery_prompt_defers_the_library(library: Path) -> None:
    block = skills_prompt(library, k_level="k1-recovery", code_backend="docker")
    assert "on your own first" in block
    assert "/workspace/skills/repo-skills-router/SKILL.md" in block


def test_provider_wrapper_copies_after_entry(library: Path, tmp_path: Path) -> None:
    """The wrapper must copy into temp_dir, which only exists after entry."""
    from agent.stirrup_agent.runner import _mounting_provider_class

    base = tmp_path / "ws"
    base.mkdir()

    class _FakeProvider:
        """Mimics the Stirrup contract: temp_dir is a child, made on entry."""

        def __init__(self, *, temp_base_dir: Path) -> None:
            self._base = Path(temp_base_dir)
            self.temp_dir = None

        async def __aenter__(self):
            self.temp_dir = self._base / "exec-abc"
            self.temp_dir.mkdir()
            return self

        async def __aexit__(self, *exc) -> None:
            return None

    wrapped = _mounting_provider_class(_FakeProvider)

    async def _run() -> Path:
        async with wrapped(temp_base_dir=base, mounts=[(library, "skills")]) as provider:
            return provider.temp_dir

    exec_dir = asyncio.run(_run())

    assert (exec_dir / "skills" / "repo-skills-router" / "SKILL.md").is_file()
    assert not (base / "skills").exists()


def test_wrapper_refuses_a_provider_without_a_temp_dir(
    library: Path, tmp_path: Path
) -> None:
    from agent.stirrup_agent.runner import _mounting_provider_class

    class _NoTempDirProvider:
        def __init__(self, **kwargs) -> None:
            self.temp_dir = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc) -> None:
            return None

    wrapped = _mounting_provider_class(_NoTempDirProvider)

    async def _run() -> None:
        async with wrapped(mounts=[(library, "skills")]):
            pass

    with pytest.raises(RuntimeError, match="no temp_dir"):
        asyncio.run(_run())


# -- preserve_workspace, and its interaction with the mount -----------------


class _FakeStirrupProvider:
    """The Stirrup contract both wrappers depend on.

    ``temp_dir`` is a child of ``temp_base_dir``, created on entry and removed
    on exit. ``_fix_file_ownership`` exists because the sandbox writes as a
    different uid.
    """

    def __init__(self, *, temp_base_dir: Path) -> None:
        self._base = Path(temp_base_dir)
        self.temp_dir: Path | None = None
        self.ownership_fixed = False
        self.cleaned_up = False

    async def __aenter__(self):
        self.temp_dir = self._base / "stirrup_agent" / "run-1" / "exec-1"
        self.temp_dir.mkdir(parents=True)
        return self

    async def _fix_file_ownership(self) -> None:
        self.ownership_fixed = True

    async def __aexit__(self, *exc) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.cleaned_up = True

    def agent_writes(self, name: str, text: str) -> None:
        (self.temp_dir / name).write_text(text)


def _compose(preserve: bool, skills: bool):
    """Mirror _build_code_provider's wrapper order."""
    from agent.stirrup_agent.runner import (
        _mounting_provider_class,
        _preserving_provider_class,
    )

    cls = _FakeStirrupProvider
    if preserve:
        cls = _preserving_provider_class(cls)
    if skills:
        cls = _mounting_provider_class(cls)
    return cls


def _run(cls, *, base: Path, writes: dict[str, str], **kwargs) -> _FakeStirrupProvider:
    async def _go():
        async with cls(temp_base_dir=base, **kwargs) as provider:
            for name, text in writes.items():
                provider.agent_writes(name, text)
            return provider

    return asyncio.run(_go())


def test_preserve_alone_keeps_agent_output_after_cleanup(tmp_path: Path) -> None:
    base = tmp_path / "ws-k0"
    base.mkdir()

    provider = _run(
        _compose(preserve=True, skills=False),
        base=base,
        writes={"answer.txt": "42"},
        preserve_dir=base,
    )

    assert provider.cleaned_up
    assert not provider.temp_dir.exists()
    assert (base / "answer.txt").read_text() == "42"
    assert provider.ownership_fixed


def test_preserve_and_skills_compose(tmp_path: Path, library: Path) -> None:
    """Both wrappers on one provider: mount on entry, preserve on exit."""
    base = tmp_path / "ws-k1"
    base.mkdir()

    provider = _run(
        _compose(preserve=True, skills=True),
        base=base,
        writes={"answer.txt": "42"},
        preserve_dir=base,
        mounts=[(library, "skills")],
    )

    # The agent's own output survives.
    assert (base / "answer.txt").read_text() == "42"
    # And the exec dir is gone, so anything left is what preserve copied.
    assert not provider.temp_dir.exists()


def test_preserve_captures_files_written_after_the_mount(
    tmp_path: Path, library: Path
) -> None:
    """The mount happens on entry; preserve must still catch later writes."""
    base = tmp_path / "ws-k1"
    base.mkdir()

    _run(
        _compose(preserve=True, skills=True),
        base=base,
        writes={"late.txt": "written after the library was mounted"},
        preserve_dir=base,
        mounts=[(library, "skills")],
    )

    assert (base / "late.txt").is_file()


def test_preserve_copies_the_mounted_library_too(
    tmp_path: Path, library: Path
) -> None:
    """Documents current behaviour: the library lands in the preserved dir.

    This is what makes `ls <ws>/skills` evidence that the mount reached the
    agent. It also means the preserved workspace mixes a mounted *input* with
    the agent's *outputs*, and that the library is duplicated once per
    preserved run.
    """
    base = tmp_path / "ws-k1"
    base.mkdir()

    _run(
        _compose(preserve=True, skills=True),
        base=base,
        writes={},
        preserve_dir=base,
        mounts=[(library, "skills")],
    )

    assert (base / "skills" / "repo-skills-router" / "SKILL.md").is_file()


def test_k0_preserve_leaves_no_skills_behind(tmp_path: Path) -> None:
    """The contamination check the docs rely on, as a test."""
    base = tmp_path / "ws-k0"
    base.mkdir()

    _run(
        _compose(preserve=True, skills=False),
        base=base,
        writes={"answer.txt": "42"},
        preserve_dir=base,
    )

    assert not (base / "skills").exists()


def test_preserve_does_not_recurse_into_itself(tmp_path: Path, library: Path) -> None:
    """preserve_dir is the parent of temp_dir, so the copy walks into itself."""
    base = tmp_path / "ws-k1"
    base.mkdir()

    _run(
        _compose(preserve=True, skills=True),
        base=base,
        writes={"answer.txt": "42"},
        preserve_dir=base,
        mounts=[(library, "skills")],
    )

    depth = max(len(p.relative_to(base).parts) for p in base.rglob("*"))
    assert depth < 8
