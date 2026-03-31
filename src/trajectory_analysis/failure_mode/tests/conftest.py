"""Pytest configuration and fixtures for trajectory analysis tests."""

import os
import pytest
from dotenv import load_dotenv
from src.llm.litellm import LiteLLMBackend


@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Load environment variables from .env file for all tests."""
    load_dotenv()

    # Verify required environment variables are set
    required_vars = {
        "LITELLM": ["LITELLM_API_KEY", "LITELLM_BASE_URL"],
        "WATSONX": ["WATSONX_APIKEY", "WATSONX_PROJECT_ID"],
    }

    missing = []
    for provider, vars in required_vars.items():
        for var in vars:
            if not os.getenv(var):
                missing.append(f"{var} (for {provider})")

    if missing:
        pytest.skip(
            f"Missing environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your API keys."
        )


@pytest.fixture
def llm_claude():
    """Fixture for Claude 4 Sonnet LLM backend (default, best accuracy)."""
    return LiteLLMBackend("litellm_proxy/GCP/claude-4-sonnet")


@pytest.fixture
def llm_llama():
    """Fixture for Llama 3.3 70B LLM backend (cost-effective)."""
    return LiteLLMBackend("watsonx/meta-llama/llama-3-3-70b-instruct")


@pytest.fixture
def llm_granite():
    """Fixture for Granite LLM backend (lowest cost)."""
    return LiteLLMBackend("watsonx/ibm/granite-13b-instruct-v2")


@pytest.fixture
def sample_trajectory():
    """Sample trajectory data for testing."""
    return {
        "text": "Analyze the chiller performance and identify any issues",
        "trajectory": [
            {
                "task_description": "Check sensor data for anomalies",
                "agent_name": "IoTAgent",
                "response": "Found temperature spike at 15:30",
            },
            {
                "task_description": "Correlate with work orders",
                "agent_name": "WorkOrderAgent",
                "response": "No recent maintenance recorded",
            },
        ],
        "final_answer": "Chiller requires immediate inspection",
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary directory for test outputs."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return str(output_dir)


# Made with Bob
