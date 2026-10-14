"""Skill mounting for the Stirrup runner (Plug A).

Stirrup has no skill mechanism: `StirrupAgentRunner` builds its system prompt
from `AGENT_SYSTEM_PROMPT` plus the code-execution blocks, and serves tools
through the workspace-bridged MCP provider. This module adds the smallest thing
that makes a skill collection usable there.

The mechanism is deliberately plain. The skill tree is copied into the
code-execution workspace base, so the agent sees it at `/workspace/skills` under
the Docker backend and at `skills/` under the local backend, and a short block
is appended to the system prompt telling it the entry point and the routing
discipline. Progressive disclosure then comes free, because the agent chooses
which file to read with the shell it already has.

Install this file at `src/agent/stirrup_agent/skills_mount.py` and apply
`patches/stirrup_runner.diff`.

Design notes
------------
The prompt block names the router and nothing else. Listing the skills in the
prompt would defeat the purpose: the whole point of a routed collection is that
the up-front context cost is one paragraph rather than the library.

`K_LEVEL` is the benchmark control. `k0` mounts nothing and appends nothing, so
the unaided baseline stays exactly what it was before this module existed.
`k1` mounts the collection. `k1-recovery` mounts it but instructs the agent to
attempt the task unaided first and consult the collection only after a concrete
failure, which preserves unaided difficulty measurement.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

_log = logging.getLogger(__name__)

K_LEVELS = ("k0", "k1", "k1-recovery")

_SKILLS_PROMPT = """\
A skill collection is mounted at {mount}. It holds operating knowledge for this
environment: which server owns which capability, the order of operations that
avoids the common failure patterns, and the preconditions a claim needs before
it is defensible.

Route before you act. Read {mount}/repo-skills-router/SKILL.md, follow it to the
repository skill, then open that skill's sub-skill for the step you are on. Read
one sub-skill at a time and open a reference file only when the sub-skill points
at it. Do not read the whole collection.

The skills describe this environment's tools and conventions. They do not
contain answers to your task.
"""

_RECOVERY_PROMPT = """\
Attempt the task on your own first. Consult the skill collection at {mount} only
after a concrete failure: a tool error you cannot resolve, an identifier that
will not resolve, or a result you cannot defend. When that happens, route
through {mount}/repo-skills-router/SKILL.md rather than browsing.
"""


_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", ".git", "tests", "reports", "test-cases"
)


def resolve_skills_source(
    skills_source: Path | str | None, k_level: str = "k1"
) -> Path | None:
    """Validate the requested library and return it, or None when unused.

    Returns None for ``k0``. Raises when ``k1``/``k1-recovery`` is requested
    without a usable library, so a run can never be labelled K1 while silently
    behaving as K0.
    """
    if k_level not in K_LEVELS:
        raise ValueError(f"k_level must be one of {K_LEVELS}, got {k_level!r}")
    if k_level == "k0":
        if skills_source is not None:
            _log.warning("k_level=k0 ignores --skills-dir %s", skills_source)
        return None
    if skills_source is None:
        raise ValueError(
            f"k_level={k_level} requires a skill library; pass --skills-dir "
            "at the directory holding repo-skills/ and repo-skills-router/"
        )
    source = Path(skills_source).expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"skills source is not a directory: {source}")
    if not (source / "repo-skills-router" / "SKILL.md").is_file():
        raise ValueError(
            f"no repo-skills-router/SKILL.md under {source}; --skills-dir must "
            "point at the directory holding repo-skills/ and repo-skills-router/"
        )
    return source


def skills_prompt(
    skills_source: Path | None,
    k_level: str = "k1",
    code_backend: str = "docker",
) -> str | None:
    """Return the system-prompt block for the mount, or None for ``k0``."""
    if k_level not in K_LEVELS:
        raise ValueError(f"k_level must be one of {K_LEVELS}, got {k_level!r}")
    if k_level == "k0" or skills_source is None:
        return None
    mount = mount_path(code_backend)
    template = _RECOVERY_PROMPT if k_level == "k1-recovery" else _SKILLS_PROMPT
    return template.format(mount=mount)


def mount_path(code_backend: str = "docker") -> str:
    """The path the agent sees, which is the exec directory, not its parent."""
    return "/workspace/skills" if code_backend == "docker" else "skills"


def copy_skills_into(skills_source: Path | str, exec_dir: Path | str) -> int:
    """Copy the library into the live code-execution directory.

    ``exec_dir`` is the directory the sandbox exposes as ``/workspace``. It is
    the provider's ``temp_dir``, a child of ``temp_base_dir``, and it does not
    exist until the provider is entered. Copying into ``temp_base_dir`` instead
    puts the library one level above the mount, where the agent cannot see it.
    """
    source = Path(skills_source).expanduser().resolve()
    destination = Path(exec_dir).expanduser().resolve() / "skills"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=_IGNORE)
    # The sandbox may run as a different uid than the process doing the copy.
    for path in destination.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    destination.chmod(0o755)
    n = sum(1 for _ in destination.rglob("SKILL.md"))
    _log.info("mounted %d skills from %s into %s", n, source, destination)
    return n


def mount_skills(
    skills_source: Path | str | None,
    workspace_dir: Path | None,
    k_level: str = "k1",
    code_backend: str = "docker",
) -> str | None:
    """Deprecated. Copies beside the exec directory, so the agent never sees it.

    Kept only so out-of-tree callers fail loudly rather than silently mounting
    into the wrong directory. Use :func:`resolve_skills_source`,
    :func:`skills_prompt` and :func:`copy_skills_into`.
    """
    raise NotImplementedError(
        "mount_skills copied the library into temp_base_dir, which is the "
        "parent of the directory exposed as /workspace. Use "
        "resolve_skills_source() + skills_prompt() at construction time and "
        "copy_skills_into(source, provider.temp_dir) after the provider is "
        "entered."
    )
