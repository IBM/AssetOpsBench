"""Budget regressions must fail at the provider boundary or saved-run outcome."""

import json

import httpx
import openai
import pytest

from scenarios.config import GeneratorConfig
from scenarios.generator import agent as agent_module, service
from scenarios.models import GroundingBundle


@pytest.fixture
def provider_run(monkeypatch, tmp_path):
    """Keep the real generator, backend, SDK, validation and output persistence.

    Replace external inventory/few-shot sources and simulate HTTP responses.
    The provider enforces a thinking+output budget rather than simply recording
    kwargs: undersized requests truncate; requests over capacity are rejected.
    """
    brief = tmp_path / "research.txt"
    brief.write_text("Power transformers use oil temperature readings for condition monitoring.")
    monkeypatch.setenv("ZAI_API_KEY", "test-key")
    monkeypatch.delenv("ZAI_BASE_URL", raising=False)
    monkeypatch.delenv("ZAI_THINKING", raising=False)
    monkeypatch.setattr(agent_module, "discover_grounding", lambda *args, **kwargs: GroundingBundle(asset_name="Transformer"))
    monkeypatch.setattr(agent_module, "fetch_hf_fewshot", lambda **kwargs: [])

    async def descriptions(self):
        return {"iot": "  - history(): Retrieve historical readings."}

    monkeypatch.setattr(agent_module.Executor, "get_server_descriptions", descriptions)
    positive = {
        "text": "Given oil temperatures of 60, 70, and 80 C at hourly intervals, calculate the hourly rate of increase.",
        "category": "Data Query", "characteristic_form": "Use iot.history to compare oil temperature readings.",
    }
    negative = {
        "text": "Give one oil temperature reading both above and below 100 C.",
        "category": "Data Query", "characteristic_form": "Cannot answer because the constraints contradict each other.",
    }
    replies = [{"description": "An oil-filled power transformer.", "relevant_tools": {"iot": [{"name": "history", "reason": "Compare recorded oil temperatures."}]}, "operator_tasks": ["Inspect oil temperature trends."], "manager_tasks": ["Review condition monitoring coverage."]}, [positive], [positive], [negative], [negative]]
    real_client = openai.OpenAI

    def install(model, capacity, *, truncate_stage=None):
        completed_stages = []

        def handler(request):
            payload = json.loads(request.content)
            budget = payload["max_tokens"]
            stage = len(completed_stages)
            if budget > capacity:
                return httpx.Response(400, json={"error": {"message": "Output budget exceeds model capacity", "type": "invalid_request_error"}})
            # A response needing 12K total tokens represents reasoning plus JSON.
            # It fails under the former 4K profile/8K generation budgets.
            truncated = budget < 12000 or stage == truncate_stage
            content = '{"description":' if truncated else json.dumps(replies[stage])
            if not truncated:
                completed_stages.append(stage)
            return httpx.Response(200, json={
                "id": "budget-regression", "object": "chat.completion", "created": 1,
                "model": model,
                "choices": [{"index": 0, "finish_reason": "length" if truncated else "stop",
                             "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": budget if truncated else 12000,
                          "total_tokens": 100 + (budget if truncated else 12000)},
            })

        def client(**kwargs):
            kwargs["max_retries"] = 0
            return real_client(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))

        monkeypatch.setattr(openai, "OpenAI", client)
        config = GeneratorConfig(
            asset_name="Transformer", backend="glm", model_id=model,
            scenario_plan={"iot": {"positive": 1, "negative": 1}},
            research_file=str(brief),
        )
        return config, completed_stages, positive, negative

    return install


@pytest.mark.anyio
@pytest.mark.parametrize("model,capacity", [("glm-5.3", 131072), ("glm-4.6v", 32768)])
async def test_reasoning_heavy_run_completes_within_model_capacity(provider_run, tmp_path, model, capacity):
    config, completed, positive, negative = provider_run(model, capacity)
    run = await service.generate_scenarios(config, output_dir=tmp_path / "run")
    assert run.complete
    assert len(completed) == 5  # Profile, generation/review, negative generation/review.
    assert [row.text for row in run.result.scenarios] == [positive["text"]]
    assert [row.text for row in run.result.negative_scenarios] == [negative["text"]]
    assert json.loads(run.output_path.read_text())[0]["text"] == positive["text"]
    assert json.loads(run.negative_output_path.read_text())[0]["text"] == negative["text"]
    assert json.loads((run.output_path.parent / "run.json").read_text())["status"] == "complete"


@pytest.mark.anyio
async def test_provider_truncation_cannot_be_saved_as_success(provider_run, tmp_path):
    config, completed, _, _ = provider_run("glm-5.3", 131072, truncate_stage=1)
    output = tmp_path / "failed-run"
    with pytest.raises(RuntimeError, match="reached max_tokens"):
        await service.generate_scenarios(config, output_dir=output)
    assert completed == [0]  # Profile succeeded; generation was truncated.
    assert json.loads((output / "run.json").read_text())["status"] == "failed"
    assert not (output / "scenarios.json").exists()
