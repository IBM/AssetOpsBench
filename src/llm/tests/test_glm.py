"""Verify the GLM adapter against the real SDK with a mock HTTP transport."""

import json

import httpx
import openai
import pytest

from llm.glm import DEFAULT_GLM_MODEL, GLMBackend


def response_body(*, content='{"ready": true}', finish_reason="stop", usage=True):
    return {
        "id": "glm-test", "object": "chat.completion", "created": 1,
        "model": DEFAULT_GLM_MODEL,
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": {
            "role": "assistant", "content": content, "reasoning_content": "not final text",
        }}],
        "usage": {"prompt_tokens": 35, "completion_tokens": 7, "total_tokens": 42} if usage else None,
    }


@pytest.fixture
def glm_env(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-zai-key")
    monkeypatch.delenv("ZAI_BASE_URL", raising=False)
    monkeypatch.delenv("ZAI_THINKING", raising=False)


def mock_api(monkeypatch, *, body=None, status=200, error=None):
    captured = {}
    real_client = openai.OpenAI

    def handler(request):
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        captured["payload"] = json.loads(request.content)
        if error:
            raise error("test transport failure", request=request)
        return httpx.Response(status, json=body if body is not None else response_body())

    def client(**kwargs):
        # Avoid retry sleeps in protocol tests.
        kwargs["max_retries"] = 0
        return real_client(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))

    monkeypatch.setattr(openai, "OpenAI", client)
    return captured


def test_default_endpoint_auth_text_and_usage(monkeypatch, glm_env):
    monkeypatch.setenv("LITELLM_API_KEY", "unrelated-key")
    captured = mock_api(monkeypatch)
    backend = GLMBackend()
    result = backend.generate_with_usage("private prompt")

    assert result.text == '{"ready": true}'
    assert (result.input_tokens, result.output_tokens) == (35, 7)
    assert captured["url"] == "https://api.z.ai/api/paas/v4/chat/completions"
    assert captured["authorization"] == "Bearer test-zai-key"
    assert captured["payload"] == {
        "model": "glm-5.3",
        "messages": [{"role": "user", "content": "private prompt"}],
        "temperature": 0.0, "max_tokens": 8192, "thinking": {"type": "enabled"},
    }


def test_overrides_and_generate_text_contract(monkeypatch, glm_env):
    monkeypatch.setenv("ZAI_BASE_URL", "https://glm.example.test/v4/")
    monkeypatch.setenv("ZAI_THINKING", "enabled")
    captured = mock_api(monkeypatch, body=response_body(usage=False))
    backend = GLMBackend("glm-4.7")
    assert backend.generate("prompt", temperature=0.5, max_tokens=4096) == '{"ready": true}'
    assert captured["url"] == "https://glm.example.test/v4/chat/completions"
    assert captured["payload"]["model"] == "glm-4.7"
    assert captured["payload"]["temperature"] == 0.5
    assert captured["payload"]["max_tokens"] == 4096
    assert captured["payload"]["thinking"] == {"type": "enabled"}
    result = backend.generate_with_usage("prompt")
    assert (result.input_tokens, result.output_tokens) == (0, 0)


@pytest.mark.parametrize("model", ["glm-5.3", "glm-5.3-flash"])
def test_required_thinking_is_enabled(monkeypatch, glm_env, model):
    captured = mock_api(monkeypatch)
    GLMBackend(model).generate("prompt")
    assert captured["payload"]["thinking"] == {"type": "enabled"}
    monkeypatch.setenv("ZAI_THINKING", "disabled")
    with pytest.raises(ValueError, match="requires ZAI_THINKING=enabled"):
        GLMBackend(model).generate("prompt")


@pytest.mark.parametrize("key", ["", "  "])
def test_missing_credentials_fail_clearly(monkeypatch, key):
    monkeypatch.setenv("ZAI_API_KEY", key)
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: pytest.fail("Missing credentials must not create an API client"))
    with pytest.raises(ValueError, match="Set ZAI_API_KEY"):
        GLMBackend().generate("prompt")


def test_invalid_configuration(glm_env, monkeypatch):
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: pytest.fail("Invalid configuration must not create an API client"))
    with pytest.raises(ValueError, match="model ID"):
        GLMBackend(" ")
    with pytest.raises(ValueError, match="positive"):
        GLMBackend().generate("prompt", max_tokens=0)
    monkeypatch.setenv("ZAI_THINKING", "auto")
    with pytest.raises(ValueError, match="enabled or disabled"):
        GLMBackend().generate("prompt")


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_http_failures_are_actionable_without_echoing_provider_body(monkeypatch, glm_env, status):
    mock_api(monkeypatch, status=status, body={"error": {"message": "private provider error"}})
    with pytest.raises(RuntimeError, match=f"HTTP {status}") as caught:
        GLMBackend().generate("prompt")
    assert "private provider error" not in str(caught.value)


@pytest.mark.parametrize("body,message", [
    ({**response_body(), "choices": []}, "no response choices"),
    (response_body(finish_reason="length"), "reached max_tokens"),
    (response_body(finish_reason="content_filter"), "did not complete"),
    (response_body(finish_reason="tool_calls"), "did not complete"),
    (response_body(content=None), "no final text"),
    (response_body(content="  "), "no final text"),
])
def test_incomplete_responses_never_become_scenarios(monkeypatch, glm_env, body, message):
    mock_api(monkeypatch, body=body)
    with pytest.raises(RuntimeError, match=message):
        GLMBackend().generate("prompt")


@pytest.mark.parametrize("error,message", [
    (httpx.ReadTimeout, "timed out"), (httpx.ConnectError, "Could not connect"),
])
def test_transport_failures(monkeypatch, glm_env, error, message):
    mock_api(monkeypatch, error=error)
    with pytest.raises(RuntimeError, match=message):
        GLMBackend().generate("prompt")
