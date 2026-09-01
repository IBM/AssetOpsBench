"""An empty completion must fail where it happens, not four frames later.

`content` is None whenever the model produced no visible text -- most often a
reasoning model that spent its whole token budget on `reasoning_content` and
stopped with finish_reason='length'. Returning that None let it travel into
`plan_execute.planner.parse_plan`, where a regex raised

    TypeError: expected string or bytes-like object, got 'NoneType'

blaming the parser for the backend's result.
"""

import sys
import types

import pytest

from llm.base import EmptyCompletionError
from llm.litellm import LiteLLMBackend


def _response(content, finish_reason="stop", completion_tokens=2048):
    """A litellm-shaped response object."""
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = types.SimpleNamespace(
        prompt_tokens=100, completion_tokens=completion_tokens
    )
    return types.SimpleNamespace(choices=[choice], usage=usage)


@pytest.fixture
def fake_litellm(monkeypatch):
    """Stand in for the litellm module the backend imports at call time."""
    module = types.ModuleType("litellm")
    module.captured = {}

    def completion(**kwargs):
        module.captured.update(kwargs)
        return module.next_response

    module.completion = completion
    monkeypatch.setitem(sys.modules, "litellm", module)
    monkeypatch.setenv("LITELLM_API_KEY", "test-key")
    monkeypatch.setenv("LITELLM_BASE_URL", "http://localhost")
    return module


def test_empty_content_raises_where_it_happens(fake_litellm):
    fake_litellm.next_response = _response(None, finish_reason="length")
    backend = LiteLLMBackend("litellm_proxy/some-reasoning-model")

    with pytest.raises(EmptyCompletionError) as excinfo:
        backend.generate_with_usage("plan this")

    error = excinfo.value
    assert error.finish_reason == "length"
    assert error.completion_tokens == 2048
    # The message must name the cause, not merely the symptom.
    assert "token cap" in str(error)
    assert "AOB_LLM_MAX_TOKENS" in str(error)


def test_empty_content_for_other_reasons_still_raises(fake_litellm):
    fake_litellm.next_response = _response(None, finish_reason="content_filter")
    backend = LiteLLMBackend("litellm_proxy/model")

    with pytest.raises(EmptyCompletionError) as excinfo:
        backend.generate_with_usage("prompt")
    assert excinfo.value.finish_reason == "content_filter"


def test_ordinary_completions_are_unaffected(fake_litellm):
    fake_litellm.next_response = _response("1. do the thing")
    backend = LiteLLMBackend("litellm_proxy/model")

    result = backend.generate_with_usage("prompt")
    assert result.text == "1. do the thing"
    assert result.input_tokens == 100
    assert result.output_tokens == 2048


def test_empty_string_is_not_an_empty_completion(fake_litellm):
    """Only None means "no content". An empty string is a real answer shape."""
    fake_litellm.next_response = _response("")
    backend = LiteLLMBackend("litellm_proxy/model")
    assert backend.generate_with_usage("prompt").text == ""


def test_token_cap_is_overridable(fake_litellm, monkeypatch):
    monkeypatch.setenv("AOB_LLM_MAX_TOKENS", "8192")
    fake_litellm.next_response = _response("ok")
    LiteLLMBackend("litellm_proxy/model").generate_with_usage("prompt")
    assert fake_litellm.captured["max_tokens"] == 8192


def test_token_cap_defaults_to_the_previous_value(fake_litellm, monkeypatch):
    monkeypatch.delenv("AOB_LLM_MAX_TOKENS", raising=False)
    fake_litellm.next_response = _response("ok")
    LiteLLMBackend("litellm_proxy/model").generate_with_usage("prompt")
    assert fake_litellm.captured["max_tokens"] == 2048
