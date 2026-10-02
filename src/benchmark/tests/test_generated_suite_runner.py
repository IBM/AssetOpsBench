"""Generated suites cannot silently omit negatives or mix model results."""
import json
from pathlib import Path

import pytest

from benchmark import generated_suite_runner as runner


@pytest.fixture
def suite(tmp_path):
    root = tmp_path / "generated"
    root.mkdir()
    (root / "run.json").write_text(json.dumps({"status": "complete", "negative_count": 1,
        "config": {"num_scenarios": 1, "num_negative_scenarios": 1}}))
    for filename, identifier in [("scenarios.json", "positive"), ("negative_scenarios.json", "negative")]:
        (root / filename).write_text(json.dumps([{"id": identifier, "text": f"Question {identifier}"}]))
    return root


def test_both_scenario_kinds_run_and_resume_without_mixing_models(suite, tmp_path, monkeypatch):
    calls = []

    def child(command, *, env, **kwargs):
        sid = command[command.index("--scenario-id") + 1]
        rid = command[command.index("--run-id") + 1]
        model = command[command.index("--model-id") + 1]
        calls.append((model, sid))
        path = Path(env["AGENT_TRAJECTORY_DIR"]) / f"{rid}.json"
        path.write_text(json.dumps({"run_id": rid, "scenario_id": sid, "runner": "openai-agent",
            "model": model, "question": command[-1], "answer": "answer", "trajectory": []}))

    def invoke(command, *, env, record, record_path, **kwargs):
        child(command, env=env)
        record.update(status='completed', execution_start=None, execution_end=None, execution_duration_ms=None)
    monkeypatch.setattr(runner, "invoke", invoke)
    output = tmp_path / "results"
    runner.run_target(suite, output, "glm", "openai", "zai/glm-5.3")
    runner.run_target(suite, output, "glm", "openai", "zai/glm-5.3")
    assert calls == [("zai/glm-5.3", "positive"), ("zai/glm-5.3", "negative")]
    # A trajectory saved before a failed process exit must not masquerade as a completed run.
    measurement=output/'glm/measurements/glm_0002.json'
    data=json.loads(measurement.read_text());data['status']='failed';measurement.write_text(json.dumps(data))
    runner.run_target(suite, output, "glm", "openai", "zai/glm-5.3")
    assert calls[-1]==("zai/glm-5.3","negative")
    assert (output/'glm/failed_trajectories/glm_0002.attempt-1.json').exists()
    with pytest.raises(ValueError, match="different model"):
        runner.run_target(suite, output, "glm", "openai", "gpt-6-astra")
    assert len(calls) == 3


def test_incomplete_or_inconsistent_generation_never_launches_agents(suite, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "invoke", lambda *args, **kwargs: pytest.fail("must not launch"))
    manifest = suite / "run.json"
    data = json.loads(manifest.read_text())
    data["status"] = "running"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="complete"):
        runner.run_target(suite, tmp_path / "results", "opus", "claude", "claude-opus-5-5")
    data["status"] = "complete"
    manifest.write_text(json.dumps(data))
    (suite / "negative_scenarios.json").write_text("[]")
    with pytest.raises(ValueError, match="counts"):
        runner.run_target(suite, tmp_path / "results", "opus", "claude", "claude-opus-5-5")
    assert not (tmp_path / "results").exists()


@pytest.mark.anyio
async def test_zai_agent_preserves_tool_calls_through_real_sdk(monkeypatch, tmp_path):
    import httpx
    import openai
    from agents import ModelSettings
    from agents.models.interface import ModelTracing
    from agent.openai_agent import runner as agent_runner

    monkeypatch.setenv("ZAI_API_KEY", "test-key")
    monkeypatch.setenv("GLM_REASONING_EFFORT", "low")
    monkeypatch.delenv("ZAI_BASE_URL", raising=False)
    monkeypatch.setenv("AGENT_TRACE_FILE", str(tmp_path / "events.jsonl"))
    real_client = openai.AsyncOpenAI

    def handle(request):
        assert str(request.url) == "https://api.z.ai/api/paas/v4/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["model"] == "glm-5.3"
        assert payload["reasoning_effort"] == "low"
        assert payload["thinking"] == {"type": "enabled"}
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 1,
            "model": "glm-5.3", "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function",
                    "function": {"name": "iot_history", "arguments": '{"asset_id":"Transformer 1"}'}}]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}})

    clients = []
    def client(**kwargs):
        clients.append(real_client(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))))
        return clients[-1]

    monkeypatch.setattr(agent_runner, "AsyncOpenAI", client)
    config = agent_runner._build_run_config("zai/glm-5.3")
    try:
        result = await config.model_provider.get_model("glm-5.3").get_response(
            None, "Fetch transformer history", config.model_settings, [], None, [], ModelTracing.DISABLED)
        tool, = result.output
        assert tool.type == "function_call"
        assert tool.name == "iot_history"
        assert json.loads(tool.arguments) == {"asset_id": "Transformer 1"}
        events=[json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
        completed,=[e for e in events if e['kind']=='model_request_end']
        assert completed['duration_ms']>0
        assert completed['response']['output'][0]['name']=='iot_history'
        assert completed['usage']['input_tokens']==10
        assert 'test-key' not in (tmp_path/'events.jsonl').read_text()
    finally:
        for instance in clients:
            await instance.close()
