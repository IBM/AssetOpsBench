"""Tests for the Stirrup runner's system prompt assembly.

The prompt is built from the shared AGENT_SYSTEM_PROMPT, the Stirrup-only
finish contract (formatted with max_turns and the finish tool's name), and,
on the code track, the code-execution and backend sections. These tests pin
that the placeholders are filled, the right sections appear per track, and
retired wording stays out.
"""

from __future__ import annotations

import pytest

from agent._prompts import AGENT_SYSTEM_PROMPT
from agent.stirrup_agent.finish_tool import ASSETOPS_FINISH_TOOL
from agent.stirrup_agent.runner import StirrupAgentRunner


def test_finish_contract_is_appended_when_code_is_disabled():
    runner = StirrupAgentRunner(code_enabled=False, max_turns=17)

    prompt = runner._build_system_prompt()

    assert prompt.startswith(AGENT_SYSTEM_PROMPT)
    assert "within 17 steps" in prompt
    assert f"`{ASSETOPS_FINISH_TOOL.name}`" in prompt
    assert "`answer`" in prompt
    assert "Code execution:" not in prompt


@pytest.mark.parametrize("max_turns", [1, 30, 250])
def test_step_budget_matches_runner_max_turns(max_turns: int):
    runner = StirrupAgentRunner(code_enabled=False, max_turns=max_turns)

    prompt = runner._build_system_prompt()

    assert f"within {max_turns} steps" in prompt


@pytest.mark.parametrize(
    ("code_enabled", "code_backend"),
    [(False, "docker"), (True, "docker"), (True, "local")],
)
def test_prompt_has_no_unformatted_placeholders(
    code_enabled: bool, code_backend: str
):
    runner = StirrupAgentRunner(code_enabled=code_enabled, code_backend=code_backend)

    prompt = runner._build_system_prompt()

    assert "{" not in prompt
    assert "}" not in prompt


def test_docker_backend_adds_code_and_docker_sections():
    runner = StirrupAgentRunner(code_enabled=True, code_backend="docker")

    prompt = runner._build_system_prompt()

    assert "Code execution:" in prompt
    assert "/workspace" in prompt
    assert "local execution workspace" not in prompt


def test_local_backend_adds_code_and_local_sections():
    runner = StirrupAgentRunner(code_enabled=True, code_backend="local")

    prompt = runner._build_system_prompt()

    assert "Code execution:" in prompt
    assert "local execution workspace" in prompt
    assert "/workspace" not in prompt


@pytest.mark.parametrize("code_backend", ["docker", "local"])
def test_code_backends_keep_short_library_and_installation_guidance(
    code_backend: str,
):
    prompt = StirrupAgentRunner(
        code_enabled=True, code_backend=code_backend
    )._build_system_prompt()

    assert "Python and common data libraries are already installed:" in prompt
    for module in (
        "numpy", "pandas", "scipy", "sklearn", "statsmodels", "matplotlib",
        "seaborn", "joblib", "threadpoolctl", "pywt",
    ):
        assert f"`{module}`" in prompt
    assert '`python -c "import <pkg>"`' in prompt
    assert "Do not install packages" in " ".join(prompt.split())


def test_tools_only_prompt_has_no_code_capability_or_installation_claims():
    prompt = StirrupAgentRunner(code_enabled=False)._build_system_prompt()

    assert "Python and common data libraries" not in prompt
    assert "scikit-learn" not in prompt
    assert "install packages" not in prompt


def test_backend_guidance_does_not_claim_network_isolation_or_host_execution():
    docker = StirrupAgentRunner(
        code_enabled=True, code_backend="docker"
    )._build_system_prompt()
    local = StirrupAgentRunner(
        code_enabled=True, code_backend="local"
    )._build_system_prompt()

    assert "no route to the asset databases" not in docker
    assert "same environment as the agent process" in local
    assert "task container in Harbor" in local
    assert "commands run on the\nhost" not in local


def test_sections_appear_in_order():
    runner = StirrupAgentRunner(code_enabled=True, code_backend="docker")

    prompt = runner._build_system_prompt()

    shared = prompt.index(AGENT_SYSTEM_PROMPT)
    finish = prompt.index(f"`{ASSETOPS_FINISH_TOOL.name}`")
    code = prompt.index("Code execution:")
    docker = prompt.index("/workspace")
    assert shared < finish < code < docker


@pytest.mark.parametrize(
    "retired",
    [
        "Record any assumptions",
        "not graded",
        "does not hold the task's data",
        "Do not overuse code_exec",
        "Prefer one complete script",
    ],
)
def test_retired_wording_is_absent(retired: str):
    runner = StirrupAgentRunner(code_enabled=True, code_backend="docker")

    prompt = runner._build_system_prompt()

    assert retired not in prompt


def test_shared_prompt_has_no_placeholders():
    # Four other runners use AGENT_SYSTEM_PROMPT without formatting it.
    assert "{" not in AGENT_SYSTEM_PROMPT
    assert "}" not in AGENT_SYSTEM_PROMPT


def test_finish_tool_exposes_the_answer_field_the_prompt_names():
    assert "answer" in ASSETOPS_FINISH_TOOL.parameters.model_fields
