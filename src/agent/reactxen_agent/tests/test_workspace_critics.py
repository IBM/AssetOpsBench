"""Offline tests for workspace-only critics; no Docker, services or model calls."""

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.models import AgentResult, ToolCall, Trajectory, TurnRecord
from agent.reactxen_agent.models import Outcome, RoundRecord
from agent.reactxen_agent.runner import ReActXenAgentRunner
from agent.reactxen_agent.workspace import shared_workspace
from agent.stirrup_agent.runner import StirrupAgentRunner
from observability import set_run_context


def review_report(outcome="accomplished"):
    return json.dumps(
        {
            "outcome": outcome,
            "arithmetic_verified": True,
            "population_verified": outcome == "accomplished",
            "question_alignment_verified": outcome == "accomplished",
            "alignment_findings": "Compare original requested scope and constraints",
            "critique": "verify filter semantics",
            "alternative_search_summary": "compared filters",
            "alternatives": []
            if outcome == "accomplished"
            else [
                {
                    "interpretation": "compare filters",
                    "candidate_answer": None,
                    "status": "plausible",
                    "supporting_evidence": "alternate event selection",
                    "counterevidence_or_gap": "definition missing",
                    "evidence_paths": ["review/results.json"],
                }
            ],
            "evidence_paths": [],
        }
    )


def trace(name="code_exec", tokens=7):
    return Trajectory(turns=[TurnRecord(0, "", [ToolCall(name, {})], tokens, 2)])


def test_critics_have_no_mcp_or_self_ask_and_share_provider(monkeypatch):
    workspace = object()
    runner = ReActXenAgentRunner(
        model="tokenrouter/MiniMax-M3",
        server_paths={"wo": "wo-mcp-server"},
        review_max_turns=6,
        reflect_max_turns=4,
    )
    for role, turns in [("review", 6), ("reflect", 4)]:
        critic = runner._critic(role, workspace)
        assert critic._server_paths == {}
        assert critic._code_provider is workspace
        assert critic._max_turns == turns
        assert critic._model_id == "tokenrouter/MiniMax-M3"
        assert critic._extra_tools == []
        assert not critic._persist_result
        monkeypatch.setattr(
            critic,
            "_build_mcp_provider",
            lambda **kw: pytest.fail("MCP must not attach"),
        )
        # Avoid requiring a real execution provider when assembling handoff tools.
        import agent.stirrup_agent.runner as module

        monkeypatch.setattr(module, "build_handoff_tools", lambda provider: [])
        assert critic._build_tools() == [workspace]


@pytest.mark.anyio
async def test_one_workspace_lifetime_survives_phase_exits_and_failure(tmp_path):
    class Provider:
        temp_dir = tmp_path
        opens = closes = 0

        async def __aenter__(self):
            self.opens += 1
            return "code-tool"

        async def __aexit__(self, *args):
            self.closes += 1

        async def write_file_bytes(self, path, content):
            (self.temp_dir / path).write_bytes(content)

        async def read_file_bytes(self, path):
            return (self.temp_dir / path).read_bytes()

    provider = Provider()
    with pytest.raises(RuntimeError):
        async with shared_workspace(provider) as borrowed:
            async with borrowed as tool:
                assert tool == "code-tool"
                await borrowed.write_file_bytes("actor.json", b"evidence")
            async with borrowed:
                assert await borrowed.read_file_bytes("actor.json") == b"evidence"
            assert provider.closes == 0
            raise RuntimeError("reflect failed")
    assert (provider.opens, provider.closes) == (1, 1)


@pytest.mark.anyio
async def test_reviewer_receives_full_collected_outputs_and_requires_inspection():
    runner = ReActXenAgentRunner()
    workspace = MagicMock()
    workspace.write_file_bytes = AsyncMock()
    actor = RoundRecord(0, "candidate", trace("wo__list_workorders", 20))
    actor.trajectory.turns[0].tool_calls[0].output = {"small_result": [1, 2, 3]}
    critic = MagicMock()
    critic.run = AsyncMock(
        return_value=AgentResult("review", review_report(), trace("finish"))
    )
    runner._critic = lambda *args: critic
    verdict = await runner._react_review("request", actor, workspace)
    assert verdict.outcome is Outcome.PARTIAL
    path, data = workspace.write_file_bytes.call_args_list[0].args
    assert path == "context/round-000/actor.json"
    stored = json.loads(data)
    assert stored["trajectory"]["turns"][0]["tool_calls"][0]["output"] == {
        "small_result": [1, 2, 3]
    }
    critic.run.return_value.trajectory = trace("code_exec")
    verdict = await runner._react_review("request", actor, workspace)
    assert verdict.outcome is Outcome.ACCOMPLISHED


@pytest.mark.anyio
async def test_react_loop_reuses_workspace_saves_phases_and_totals(
    monkeypatch, tmp_path
):
    import agent.reactxen_agent.runner as module

    workspace = MagicMock()
    workspace.write_file_bytes = AsyncMock()

    @asynccontextmanager
    async def shared(provider):
        yield workspace

    monkeypatch.setattr(module, "shared_workspace", shared)
    monkeypatch.setattr(
        StirrupAgentRunner, "_build_code_provider", lambda self: object()
    )
    monkeypatch.setenv("AGENT_TRAJECTORY_DIR", str(tmp_path / "traces"))
    set_run_context(run_id="react-test", scenario_id="413")
    runner = ReActXenAgentRunner(
        workspace_dir=tmp_path / "workspace",
        max_rounds=2,
        distill_question_enabled=False,
    )
    actor = MagicMock()
    actor.run = AsyncMock(
        return_value=AgentResult("q", "answer", trace("wo__list_workorders", 20))
    )
    actor_prompts = []

    def actor_core(prompt, self_ask):
        actor_prompts.append(prompt)
        return actor

    runner._core = actor_core
    reviews = [
        review_report("partially_accomplished"),
        review_report(),
    ]
    created = []

    def critic(role, shared_provider, **kwargs):
        assert shared_provider is workspace
        phase = MagicMock()
        if role == "review":
            data = json.loads(reviews.pop(0))
            data["candidate_comparisons"] = [
                {
                    "candidate_id": entry["candidate_id"],
                    "status": "ruled_out",
                    "candidate_answer": None,
                    "interpretation": "verified the competing filter",
                    "supporting_evidence": "previous calculation",
                    "counterevidence_or_gap": "resolved against it",
                    "evidence_paths": ["review/results.json"],
                }
                for entry in workspace.candidate_ledger.entries
            ]
        else:
            data = {
                "selected_alternative": "alternative_1",
                "selection_reason": "verified the filter",
                "alternative_assessments": [
                    {
                        "alternative_id": "alternative_1",
                        "decision": "selected",
                        "reason": "matches evidence",
                        "evidence_paths": ["review/results.json"],
                    }
                ],
                "evidence_paths": ["review/results.json"],
                "candidate_answer": None,
                "correction_steps": ["Use the verified filter in review/results.json"],
                "candidate_assessments": [
                    {
                        "candidate_id": entry["candidate_id"],
                        "decision": "unresolved",
                        "reason": "actor must verify",
                        "evidence_paths": [],
                    }
                    for entry in workspace.candidate_ledger.entries
                ],
            }
        phase.run = AsyncMock(
            return_value=AgentResult(role, json.dumps(data), trace(tokens=7))
        )
        created.append(role)
        return phase

    runner._critic = critic
    result = await runner.run("q")
    assert created == ["review", "reflect", "review"]
    assert "ACTOR ROUND 1 OF 2" in actor_prompts[0]
    assert "FINAL ACTOR ROUND" in actor_prompts[1]
    assert "no additional Review/Reflect correction cycle" in actor_prompts[1]
    assert "Use the verified filter in review/results.json" in actor_prompts[1]
    assert result.trajectory.rounds[1].final_actor_attempt
    assert not result.trajectory.rounds[0].final_actor_attempt
    assert result.trajectory.stopped_because == "accomplished"
    assert result.trajectory.total_input_tokens == 61  # actor 40 + critics 21
    assert actor._code_provider is workspace
    assert actor._persist_result is False
    assert len(result.trajectory.turns) == 5
    saved = json.loads((tmp_path / "traces/react-test.json").read_text())
    assert saved["runner"] == "reactxen-agent"
    assert saved["trajectory"]["rounds"][0]["review_trajectory"]
    assert saved["trajectory"]["rounds"][0]["reflect_trajectory"]
    assert saved["trajectory"]["workspace_dir"]
    assert list((tmp_path / "traces").glob("*.json")) == [
        tmp_path / "traces/react-test.json"
    ]


def test_no_code_requires_explicit_legacy_mode_or_no_review():
    with pytest.raises(ValueError, match="require code_enabled"):
        ReActXenAgentRunner(code_enabled=False)
    ReActXenAgentRunner(code_enabled=False, critic_mode="completion")
    ReActXenAgentRunner(code_enabled=False, review_enabled=False)
