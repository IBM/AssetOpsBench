from agent.direct_llm_agent.runner import DirectLLMAgentRunner


class DummyLLMResult:
    text = '{"test": 1}'
    input_tokens = 10
    output_tokens = 5


class DummyLLM:
    model_id = "dummy/direct-llm"

    def generate_with_usage(self, prompt: str, temperature: float = 0.0):
        self.prompt = prompt
        self.temperature = temperature
        return DummyLLMResult()


async def test_direct_llm_agent_returns_model_answer():
    llm = DummyLLM()
    runner = DirectLLMAgentRunner(llm=llm)

    result = await runner.run('Return only JSON: {"test": 1}')

    assert result.answer == '{"test": 1}'
    assert result.question == 'Return only JSON: {"test": 1}'
    assert result.trajectory.total_input_tokens == 10
    assert result.trajectory.total_output_tokens == 5
    assert len(result.trajectory.turns) == 1
    assert result.trajectory.turns[0].tool_calls == []
    assert "Return only JSON" in llm.prompt
    assert llm.temperature == 0.0
