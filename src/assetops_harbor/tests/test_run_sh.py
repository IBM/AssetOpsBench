"""Keep benchmarks/harbor/run.sh in step with Harbor and llm.routers.

run.sh names Harbor exception classes as strings for `harbor jobs resume
--filter-error-type`, which matches the exact class name and ignores a name
that matches nothing. A typo, a rename, or a new subclass would silently stop
those trials from running again, so these tests read the names out of the
script and check them against the installed Harbor.
"""

from __future__ import annotations

import asyncio
import inspect
import re
from pathlib import Path

import pytest

pytest.importorskip(
    "harbor.agents.installed.base",
    reason="Harbor is an optional extra: uv sync --dev --extra harbor",
)

from harbor.agents.installed import base as installed_base
from harbor.environments import base as environments_base
from harbor.trial import errors as trial_errors
from harbor.verifier import verifier

from assetops_harbor.stirrup import ROUTER_CREDENTIALS

RUN_SH = Path(__file__).resolve().parents[3] / "benchmarks/harbor/run.sh"

# Failures that are the model's own work, so a resume keeps them as results.
KEPT = {
    "AgentTimeoutError",
    "ContextWindowExceededError",
    "OutputTokenExceededError",
    "AgentSafetyRefusalError",
}


def _retry_error_types() -> list[str]:
    text = RUN_SH.read_text(encoding="utf-8")
    block = re.search(r"^retry_error_types=\((.*?)^\)", text, re.MULTILINE | re.DOTALL)
    assert block, "retry_error_types=( ... ) not found in run.sh"
    return block.group(1).split()


def _exception_classes() -> dict[str, type]:
    classes = {"CancelledError": asyncio.CancelledError}
    for module in (installed_base, environments_base, trial_errors, verifier):
        for name, obj in vars(module).items():
            if inspect.isclass(obj) and issubclass(obj, BaseException):
                classes[name] = obj
    return classes


def test_every_retried_error_type_exists_in_harbor() -> None:
    known = _exception_classes()
    unknown = [name for name in _retry_error_types() if name not in known]
    assert not unknown, f"not Harbor exception classes: {unknown}"


def test_every_agent_error_is_retried_or_kept() -> None:
    """A new NonZeroAgentExitCodeError subclass must be placed on one side."""
    listed = set(_retry_error_types())
    base = installed_base.NonZeroAgentExitCodeError
    subclasses = {
        name
        for name, obj in vars(installed_base).items()
        if inspect.isclass(obj) and issubclass(obj, base)
    }
    unplaced = subclasses - listed - KEPT
    assert not unplaced, f"add to retry_error_types in run.sh or to KEPT: {unplaced}"
    assert not listed & KEPT


def test_the_router_probe_matches_the_agent() -> None:
    """check_model's ROUTERS is a copy; ROUTER_CREDENTIALS is tested against llm."""
    text = RUN_SH.read_text(encoding="utf-8")
    block = re.search(r"^ROUTERS = \{(.*?)^\}", text, re.MULTILINE | re.DOTALL)
    assert block, "ROUTERS = { ... } not found in run.sh"
    routers = {
        prefix: (base, key)
        for prefix, base, key in re.findall(
            r'"([^"]+)": \("([^"]+)", "([^"]+)"\)', block.group(1)
        )
    }
    assert routers == ROUTER_CREDENTIALS
