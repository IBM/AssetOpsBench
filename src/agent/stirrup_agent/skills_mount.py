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


def mount_skills(
    skills_source: Path | str | None,
    workspace_dir: Path | None,
    k_level: str = "k1",
    code_backend: str = "docker",
) -> str | None:
    """Copy the skill tree into the workspace and return the prompt block.

    Returns None when nothing should be appended to the system prompt, which is
    the case for ``k0`` and whenever the source is absent.
    """
    if k_level not in K_LEVELS:
        raise ValueError(f"k_level must be one of {K_LEVELS}, got {k_level!r}")
    if k_level == "k0" or skills_source is None:
        return None

    source = Path(skills_source).expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"skills source is not a directory: {source}")
    if workspace_dir is None:
        raise ValueError("workspace_dir is required when skills are mounted")

    destination = Path(workspace_dir).expanduser().resolve() / "skills"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".git", "tests", "reports", "test-cases"))

    mount = "/workspace/skills" if code_backend == "docker" else "skills"
    n = sum(1 for _ in destination.rglob("SKILL.md"))
    _log.info("mounted %d skills from %s at %s (k_level=%s)", n, source, mount, k_level)

    template = _RECOVERY_PROMPT if k_level == "k1-recovery" else _SKILLS_PROMPT
    return template.format(mount=mount)
