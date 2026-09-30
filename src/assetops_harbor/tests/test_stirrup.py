"""Unit tests for the Harbor agent adapter.

Harbor is an optional extra (``uv sync --extra harbor``), so everything here
skips cleanly when it is absent.

The trajectory tests are written against the SERIALIZED record that
``observability.persistence`` writes, rather than against the dataclasses that
produce it. That keeps them independent of the agent runner dependency tree,
which ``agent/__init__`` pulls in eagerly. One test below does import the real
dataclasses to catch a field rename, and skips when those deps are missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip(
    "harbor.models.trajectories",
    reason="Harbor is an optional extra: uv sync --dev --extra harbor",
)

from harbor.models.trajectories import Trajectory

from assetops_harbor.stirrup import (
    FMSR_MODEL_ENV,
    ROUTER_CREDENTIALS,
    SETTING_ENV_VARS,
    SHARED_WORKSPACE,
    StirrupAgent,
)

RUN_ID = "wosr-1__abc1234"


@pytest.fixture(autouse=True)
def _isolated_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's own .env out of these tests.

    StirrupAgent loads the nearest .env above the cwd, so running from the repo
    root would otherwise feed real credentials into every agent built here.
    """
    monkeypatch.chdir(tmp_path)


def _agent(tmp_path: Path) -> tuple[StirrupAgent, Path]:
    logs_dir = tmp_path / RUN_ID / "agent"
    logs_dir.mkdir(parents=True)
    return StirrupAgent(logs_dir=logs_dir, model_name="watsonx/test"), logs_dir


def _write_record(logs_dir: Path, trajectory: object) -> None:
    (logs_dir / f"{RUN_ID}.json").write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "scenario_id": "1",
                "runner": "stirrup",
                "model": "watsonx/test",
                "question": "How many work orders?",
                "answer": "42",
                "trajectory": trajectory,
            }
        ),
        encoding="utf-8",
    )


# agent.models.Trajectory -> dataclasses.asdict
SDK_TRAJECTORY = {
    "started_at": "2026-09-26T10:00:00+00:00",
    "turns": [
        {
            "index": 1,
            "text": "Querying work orders.",
            "tool_calls": [
                {
                    "name": "get_work_orders",
                    "input": {"site": "MAIN"},
                    "id": "",  # ToolCall.id defaults to "" on main
                    "output": {"count": 42},
                    "duration_ms": None,
                }
            ],
            "input_tokens": 1200,
            "output_tokens": 80,
            "duration_ms": None,
        },
        {
            "index": 2,
            "text": "42",
            "tool_calls": [],
            "input_tokens": 1400,
            "output_tokens": 12,
            "duration_ms": None,
        },
    ],
}

# plan_execute.models.StepResult -> list[dataclasses.asdict]
PLAN_EXECUTE_TRAJECTORY = [
    {
        "step_number": 1,
        "task": "count work orders",
        "server": "wo",
        "response": "42",
        "error": None,
        "tool": "get_work_orders",
        "tool_args": {"site": "MAIN"},
        "duration_ms": None,
    }
]


def test_run_id_is_the_harbor_trial_name(tmp_path: Path) -> None:
    agent, _ = _agent(tmp_path)
    assert agent.run_id == RUN_ID


def test_docker_code_backend_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no Docker daemon"):
        StirrupAgent(logs_dir=tmp_path / "agent", model_name="m", code_backend="docker")

    agent = StirrupAgent(
        logs_dir=tmp_path / "agent",
        model_name="m",
        code_backend="docker",
        allow_docker_backend=True,
        workspace_dir=SHARED_WORKSPACE,
    )
    assert agent.code_backend == "docker"


def test_kwargs_arrive_as_strings_from_the_cli(tmp_path: Path) -> None:
    agent = StirrupAgent(
        logs_dir=tmp_path / "agent",
        model_name="m",
        code_enabled="false",
        max_turns="40",
    )
    assert agent.code_enabled is False
    assert agent.max_turns == 40


def test_sdk_trajectory_converts_to_valid_atif(tmp_path: Path) -> None:
    agent, logs_dir = _agent(tmp_path)
    _write_record(logs_dir, SDK_TRAJECTORY)

    trajectory = agent.convert_trajectory(logs_dir)
    assert trajectory is not None
    Trajectory.model_validate(trajectory.model_dump())

    assert [s.source for s in trajectory.steps] == ["user", "agent", "agent", "agent"]
    assert trajectory.final_metrics.total_prompt_tokens == 2600
    assert trajectory.final_metrics.total_completion_tokens == 92
    assert trajectory.steps[-1].message == "42"


def test_tool_call_ids_are_synthesized_when_empty(tmp_path: Path) -> None:
    """ATIF rejects an observation result naming no tool call in its own step.

    agent.models.ToolCall.id defaults to "", so the adapter must supply one.
    """
    agent, logs_dir = _agent(tmp_path)
    _write_record(logs_dir, SDK_TRAJECTORY)

    trajectory = agent.convert_trajectory(logs_dir)
    tool_step = trajectory.steps[1]
    call_ids = {call.tool_call_id for call in tool_step.tool_calls}

    assert call_ids and "" not in call_ids
    assert {r.source_call_id for r in tool_step.observation.results} <= call_ids


def test_plan_execute_trajectory_converts_to_valid_atif(tmp_path: Path) -> None:
    agent, logs_dir = _agent(tmp_path)
    _write_record(logs_dir, PLAN_EXECUTE_TRAJECTORY)

    trajectory = agent.convert_trajectory(logs_dir)
    assert trajectory is not None
    Trajectory.model_validate(trajectory.model_dump())
    # StepResult carries no token counts.
    assert trajectory.final_metrics.total_prompt_tokens == 0


def test_absent_trajectory_still_yields_question_and_answer(tmp_path: Path) -> None:
    agent, logs_dir = _agent(tmp_path)
    _write_record(logs_dir, None)

    trajectory = agent.convert_trajectory(logs_dir)
    assert trajectory is not None
    Trajectory.model_validate(trajectory.model_dump())
    assert [s.source for s in trajectory.steps] == ["user", "agent"]


def test_missing_record_returns_none(tmp_path: Path) -> None:
    agent, logs_dir = _agent(tmp_path)
    assert agent.convert_trajectory(logs_dir) is None


def test_fixtures_match_the_real_dataclasses() -> None:
    """Catch a field rename in agent.models or plan_execute.models.

    Skips when the agent runner dependency tree is not installed, since
    agent/__init__ imports every runner eagerly.
    """
    import dataclasses

    models = pytest.importorskip("agent.models")
    plan_models = pytest.importorskip("agent.plan_execute.models")

    turn_fields = {f.name for f in dataclasses.fields(models.TurnRecord)}
    call_fields = {f.name for f in dataclasses.fields(models.ToolCall)}
    step_fields = {f.name for f in dataclasses.fields(plan_models.StepResult)}

    assert set(SDK_TRAJECTORY["turns"][0]) == turn_fields
    assert set(SDK_TRAJECTORY["turns"][0]["tool_calls"][0]) == call_fields
    assert set(PLAN_EXECUTE_TRAJECTORY[0]) == step_fields


def test_populate_context_post_run_writes_trajectory_and_tokens(tmp_path: Path) -> None:
    """Harbor never calls convert_trajectory itself.

    Trial._sync_agent_output calls populate_context_post_run and then reads
    logs_dir/trajectory.json back for model usage, so the hook must both write
    the file and fill the context.
    """
    from harbor.models.agent.context import AgentContext

    agent, logs_dir = _agent(tmp_path)
    _write_record(logs_dir, SDK_TRAJECTORY)

    context = AgentContext()
    agent.populate_context_post_run(context)

    written = logs_dir / "trajectory.json"
    assert written.exists(), "trajectory.json was not written"
    Trajectory.model_validate(json.loads(written.read_text()))

    assert context.n_input_tokens == 2600
    assert context.n_output_tokens == 92


def test_populate_context_post_run_is_quiet_without_a_record(tmp_path: Path) -> None:
    from harbor.models.agent.context import AgentContext

    agent, logs_dir = _agent(tmp_path)
    context = AgentContext()
    agent.populate_context_post_run(context)

    assert not (logs_dir / "trajectory.json").exists()
    assert context.n_input_tokens is None


@pytest.fixture
def no_router_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip router credentials from the process environment.

    _get_env falls back to os.environ by design, so a developer who exports
    these for real runs would otherwise see this test pass or fail depending on
    their shell.
    """
    for pair in ROUTER_CREDENTIALS.values():
        for name in pair:
            # setenv first so teardown restores the original state even when
            # the test itself writes the variable, which load_dotenv does.
            monkeypatch.setenv(name, "")
            monkeypatch.delenv(name)


def test_router_credentials_are_required_up_front(
    tmp_path: Path, no_router_credentials: None
) -> None:
    """Missing creds must fail before Harbor builds an image per trial."""
    with pytest.raises(ValueError, match="TOKENROUTER_BASE_URL"):
        StirrupAgent(logs_dir=tmp_path / "agent", model_name="tokenrouter/MiniMax-M3")


def test_router_credentials_from_agent_env_satisfy_the_check(
    tmp_path: Path, no_router_credentials: None
) -> None:
    agent = StirrupAgent(
        logs_dir=tmp_path / "agent",
        model_name="tokenrouter/MiniMax-M3",
        extra_env={
            "TOKENROUTER_BASE_URL": "https://example.invalid/v1",
            "TOKENROUTER_API_KEY": "k",
        },
    )
    forwarded = agent._credential_env()
    assert forwarded["TOKENROUTER_BASE_URL"] == "https://example.invalid/v1"
    assert forwarded["TOKENROUTER_API_KEY"] == "k"
    # Only what is set, never empty placeholders.
    assert all(forwarded.values())
    assert "OPENAI_API_KEY" not in forwarded or forwarded["OPENAI_API_KEY"]


def test_router_credentials_from_dotenv_satisfy_the_check(
    tmp_path: Path, no_router_credentials: None
) -> None:
    (tmp_path / ".env").write_text(
        "TOKENROUTER_BASE_URL=https://example.invalid/v1\nTOKENROUTER_API_KEY=k\n",
        encoding="utf-8",
    )
    agent = StirrupAgent(logs_dir=tmp_path / "agent", model_name="tokenrouter/MiniMax-M3")
    forwarded = agent._credential_env()
    assert forwarded["TOKENROUTER_BASE_URL"] == "https://example.invalid/v1"
    assert forwarded["TOKENROUTER_API_KEY"] == "k"


def test_exported_variables_win_over_dotenv(
    tmp_path: Path, no_router_credentials: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(
        "TOKENROUTER_BASE_URL=https://example.invalid/v1\n"
        "TOKENROUTER_API_KEY=from-file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TOKENROUTER_API_KEY", "from-shell")
    agent = StirrupAgent(logs_dir=tmp_path / "agent", model_name="tokenrouter/MiniMax-M3")
    forwarded = agent._credential_env()
    assert forwarded["TOKENROUTER_API_KEY"] == "from-shell"
    assert forwarded["TOKENROUTER_BASE_URL"] == "https://example.invalid/v1"


def test_unprefixed_models_need_no_router_creds(tmp_path: Path) -> None:
    StirrupAgent(logs_dir=tmp_path / "agent", model_name="watsonx/llama-4")


def test_router_map_matches_llm_routers() -> None:
    """Catch drift from src/llm/routers.py, the source of truth.

    Skips when the agent dependency tree is absent, since llm/__init__ imports
    the LiteLLM and OpenAI backends.
    """
    routers = pytest.importorskip("llm.routers")
    assert ROUTER_CREDENTIALS == routers.PROXY_ROUTERS


def test_docker_backend_requires_a_shared_workspace(tmp_path: Path) -> None:
    """The dind bind-mount trap must fail loudly, not silently lose files."""
    with pytest.raises(ValueError, match="workspace_dir"):
        StirrupAgent(
            logs_dir=tmp_path / "agent",
            model_name="m",
            code_enabled=True,
            code_backend="docker",
            allow_docker_backend=True,
        )

    agent = StirrupAgent(
        logs_dir=tmp_path / "agent",
        model_name="m",
        code_enabled=True,
        code_backend="docker",
        allow_docker_backend=True,
        workspace_dir=SHARED_WORKSPACE,
    )
    assert agent.workspace_dir == SHARED_WORKSPACE


def test_local_backend_needs_no_workspace(tmp_path: Path) -> None:
    agent = StirrupAgent(logs_dir=tmp_path / "agent", model_name="m", code_enabled=True)
    assert agent.code_backend == "local"
    assert agent.workspace_dir is None


def test_fmsr_model_env_matches_the_agent_package():
    """The literal here must track agent.runner.FMSR_MODEL_ENV.

    stirrup.py cannot import agent.runner: that pulls the agent SDKs into the
    host-side Harbor process. Same reason ROUTER_CREDENTIALS is copied.
    """
    from agent.runner import FMSR_MODEL_ENV as canonical

    assert FMSR_MODEL_ENV == canonical
    assert FMSR_MODEL_ENV in SETTING_ENV_VARS


def test_explicit_fmsr_model_reaches_the_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """An operator's FMSR_MODEL_ID must be forwarded, not dropped.

    Without it the in-container runner falls back to the agent's --model-id, so
    the documented "explicit value wins" would hold for the CLI only.
    """
    for name in ("LITELLM_BASE_URL", "LITELLM_API_KEY"):
        monkeypatch.setenv(name, "x")
    for name in ("TOKENROUTER_BASE_URL", "TOKENROUTER_API_KEY"):
        monkeypatch.setenv(name, "y")
    monkeypatch.setenv(FMSR_MODEL_ENV, "litellm_proxy/aws/claude-opus-5")

    agent = StirrupAgent(
        model_name="tokenrouter/MiniMax-M3",
        logs_dir=tmp_path,
        code_enabled=False,
    )
    env = agent._credential_env()
    assert env[FMSR_MODEL_ENV] == "litellm_proxy/aws/claude-opus-5"


def test_fmsr_router_credentials_are_required_up_front(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A differently-routed FMSR model must fail before the image build."""
    for name in ("TOKENROUTER_BASE_URL", "TOKENROUTER_API_KEY"):
        monkeypatch.setenv(name, "y")
    for name in ("LITELLM_BASE_URL", "LITELLM_API_KEY"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setenv(FMSR_MODEL_ENV, "litellm_proxy/aws/claude-opus-5")

    with pytest.raises(ValueError, match=f"used by {FMSR_MODEL_ENV}"):
        StirrupAgent(
            model_name="tokenrouter/MiniMax-M3",
            logs_dir=tmp_path,
            code_enabled=False,
        )
