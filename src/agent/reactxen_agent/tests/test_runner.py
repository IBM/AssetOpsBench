"""Exercise real constructor/tool wiring and mocked round boundaries."""

import json

import pytest

from agent.models import AgentResult, ToolCall, Trajectory, TurnRecord
from agent.reactxen_agent.runner import ReActXenAgentRunner
from agent.reactxen_agent.self_ask import SelfAskTool
from observability import set_run_context


def test_core_accepts_prompt_and_typed_tool():
    runner = ReActXenAgentRunner(
        server_paths={}, code_enabled=False, review_enabled=False
    )
    core = runner._core("custom guidance", SelfAskTool(complete=None))
    assert core._build_system_prompt() == "custom guidance"
    assert core._extra_tools[0].name == "self_ask"
    assert (
        core._extra_tools[0].parameters.model_validate({"context": "q"}).context == "q"
    )


@pytest.mark.anyio
async def test_proxy_critic_uses_named_credentials(monkeypatch):
    from types import SimpleNamespace

    import litellm

    monkeypatch.setenv("TOKENROUTER_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("TOKENROUTER_API_KEY", "test-only")

    async def complete(**kwargs):
        assert kwargs["model"] == "openai/MiniMax-M3"
        assert kwargs["api_base"] == "https://example.invalid/v1"
        assert kwargs["api_key"] == "test-only"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))], usage=None
        )

    monkeypatch.setattr(litellm, "acompletion", complete)
    assert await ReActXenAgentRunner(model="tokenrouter/MiniMax-M3")._complete(
        "s", "u"
    ) == ("ok", 0, 0)


@pytest.mark.anyio
async def test_rounds_save_guidance_self_ask_cost_and_final_record(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("AGENT_TRAJECTORY_DIR", str(tmp_path))
    set_run_context(run_id="regression", scenario_id="test")
    runner = ReActXenAgentRunner(server_paths={}, critic_mode="completion")
    prompts = []

    async def complete(system, user):
        if "OUTCOME" in system:
            verdict = "partially_accomplished" if len(prompts) == 1 else "accomplished"
            return (
                f"OUTCOME: {verdict}\nCRITIQUE: inspect evidence\nALTERNATIVES: inspect",
                10,
                2,
            )
        return "inspect the evidence", 10, 2

    runner._complete = complete

    class Core:
        async def run(self, question):
            await tool("evidence")
            return AgentResult(
                question,
                "answer",
                Trajectory(
                    turns=[TurnRecord(0, "", [ToolCall("get_sites", {})], 100, 20)]
                ),
            )

    def core(prompt, self_ask):
        nonlocal tool
        tool = self_ask
        prompts.append(prompt)
        return Core()

    tool = None
    monkeypatch.setattr(runner, "_core", core)
    result = await runner.run("question")
    assert len(result.trajectory.rounds) == 2
    assert "inspect the evidence" in prompts[1]
    assert result.trajectory.rounds[0].reflection == "inspect the evidence"
    assert result.trajectory.total_input_tokens == 250
    saved = json.loads((tmp_path / "regression.json").read_text())
    assert saved["runner"] == "reactxen-agent"
    assert len(saved["trajectory"]["rounds"]) == 2
    assert len(saved["trajectory"]["turns"]) == 2


def test_mcp_env_preserves_database_and_withholds_provider_secret(monkeypatch):
    from agent.stirrup_agent.runner import StirrupAgentRunner

    monkeypatch.setenv("COUCHDB_URL", "http://127.0.0.1:15984")
    monkeypatch.setenv("TOKENROUTER_API_KEY", "must-not-forward")
    config = StirrupAgentRunner(
        server_paths={"wo": "wo-mcp-server"}
    )._build_mcp_config()
    server = config.mcp_servers["wo"]
    assert server.env["COUCHDB_URL"] == "http://127.0.0.1:15984"
    assert "TOKENROUTER_API_KEY" not in server.env
