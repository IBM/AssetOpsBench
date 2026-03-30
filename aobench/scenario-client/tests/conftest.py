"""Pytest configuration and fixtures for scenario-client tests."""

import pytest
from unittest.mock import MagicMock
from scenario_client.client import TrackingContext


@pytest.fixture
def mock_scenario_uri():
    """Return a mock scenario server URI."""
    return "https://test-scenario-server.com"


@pytest.fixture
def mock_tracking_uri():
    """Return a mock MLflow tracking URI."""
    return "https://test-mlflow-server.com"


@pytest.fixture
def sample_scenario_set():
    """Return a sample scenario set response."""
    return {
        "title": "Test Scenario Set",
        "scenarios": [
            {
                "id": "scenario-1",
                "inputs": {"prompt": "What is 2+2?"},
            },
            {
                "id": "scenario-2",
                "inputs": {"prompt": "What is the capital of France?"},
            },
        ],
    }


@pytest.fixture
def sample_tracking_context():
    """Return a sample tracking context."""
    return TrackingContext(
        uri="https://test-mlflow-server.com",
        experiment_id="test-experiment-123",
        run_id="test-run-456",
    )


@pytest.fixture
def sample_tracking_context_dict():
    """Return a sample tracking context as dict (for server responses)."""
    return {
        "uri": "https://test-mlflow-server.com",
        "experiment_id": "test-experiment-123",
        "run_id": "test-run-456",
    }


@pytest.fixture
def sample_scenario_set_with_tracking(
    sample_scenario_set, sample_tracking_context_dict
):
    """Return a scenario set response with tracking context."""
    return {
        **sample_scenario_set,
        "tracking_context": sample_tracking_context_dict,
    }


@pytest.fixture
def sample_answers():
    """Return sample answers for grading."""
    return [
        {"scenario_id": "scenario-1", "answer": "4"},
        {"scenario_id": "scenario-2", "answer": "Paris"},
    ]


@pytest.fixture
def sample_grade_response():
    """Return a sample grading response."""
    return {
        "score": 0.95,
        "total_scenarios": 2,
        "correct": 2,
        "details": [
            {"scenario_id": "scenario-1", "correct": True, "score": 1.0},
            {"scenario_id": "scenario-2", "correct": True, "score": 0.9},
        ],
    }


@pytest.fixture
def sample_deferred_grading_response():
    """Return a sample deferred grading response."""
    return {
        "grading_id": "grading-789",
        "status": "pending",
    }


@pytest.fixture
def sample_scenario_types():
    """Return sample scenario types."""
    return {
        "types": [
            {"id": "qa", "name": "Question Answering"},
            {"id": "summarization", "name": "Text Summarization"},
            {"id": "classification", "name": "Text Classification"},
        ]
    }


@pytest.fixture
def mock_mlflow(monkeypatch):
    """Mock MLflow module."""
    mock = MagicMock()
    mock.set_tracking_uri = MagicMock()
    mock.set_experiment = MagicMock()
    mock.start_run = MagicMock()
    mock.end_run = MagicMock()
    mock.set_tag = MagicMock()
    mock.start_span = MagicMock()
    mock.set_trace_tag = MagicMock()
    mock.search_traces = MagicMock()
    mock.langchain = MagicMock()
    mock.langchain.autolog = MagicMock()

    monkeypatch.setattr("scenario_client.client.mlflow", mock)
    return mock


@pytest.fixture
def simple_sync_function():
    """Return a simple synchronous function for testing."""

    def func(prompt: str) -> str:
        return f"Response to: {prompt}"

    return func


@pytest.fixture
def simple_async_function():
    """Return a simple asynchronous function for testing."""

    async def afunc(prompt: str) -> str:
        return f"Async response to: {prompt}"

    return afunc
