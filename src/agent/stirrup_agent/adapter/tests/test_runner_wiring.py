"""The runner accepts an adapter and defaults to the unadapted surface.

Stirrup is not installable here, so this tests the wiring rather than a live
run: that the constructor takes the argument, that an absent adapter loads the
empty spec, and that an empty spec leaves the provider class untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.stirrup_agent.adapter.provider import build_mcp_provider_class, load_spec
from agent.stirrup_agent.adapter.spec import AdapterError


class Base:
    pass


def test_no_adapter_dir_loads_the_empty_spec() -> None:
    assert load_spec(None).is_empty


def test_the_empty_spec_leaves_the_provider_class_identical() -> None:
    """The baseline guarantee, asserted at the wiring level."""
    assert build_mcp_provider_class(Base, load_spec(None)) is Base


def test_a_real_adapter_dir_loads_and_wraps(tmp_path: Path) -> None:
    (tmp_path / "adapter.json").write_text(
        json.dumps({"tools": {"wo__query": {"strip_empty": True}}})
    )

    spec = load_spec(tmp_path)

    assert not spec.is_empty
    assert build_mcp_provider_class(Base, spec) is not Base


def test_a_directory_without_a_spec_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(AdapterError, match="no adapter.json"):
        load_spec(tmp_path)


def test_the_runner_takes_an_adapter_dir_argument() -> None:
    import inspect

    from agent.stirrup_agent.runner import StirrupAgentRunner

    params = inspect.signature(StirrupAgentRunner.__init__).parameters
    assert "adapter_dir" in params
    assert params["adapter_dir"].default is None


def test_the_cli_exposes_adapter_dir() -> None:
    from agent.stirrup_agent.cli import _build_parser

    options = {a.dest for a in _build_parser()._actions}
    assert "adapter_dir" in options
