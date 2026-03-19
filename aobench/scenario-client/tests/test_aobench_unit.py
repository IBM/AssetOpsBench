"""Unit tests for AOBench class methods."""

import pytest
from unittest.mock import Mock, MagicMock
from scenario_client.client import AOBench, tag_latest_trace


class TestTagLatestTrace:
    """Test the tag_latest_trace helper function."""

    def test_tag_latest_trace_success(self, mock_mlflow):
        """Test tagging trace when exactly one trace is found."""
        # Setup mock DataFrame-like object
        mock_traces = MagicMock()
        mock_traces.__len__ = MagicMock(return_value=1)
        mock_traces.trace_id = ["trace-123"]
        mock_mlflow.search_traces.return_value = mock_traces

        # Call function
        tag_latest_trace("exp-1", "run-1", "scenario-1")

        # Verify calls
        mock_mlflow.search_traces.assert_called_once_with(
            experiment_ids=["exp-1"],
            run_id="run-1",
            order_by=["timestamp_ms DESC"],
            max_results=1,
        )
        mock_mlflow.set_trace_tag.assert_called_once_with(
            trace_id="trace-123", key="scenario_id", value="scenario-1"
        )

    def test_tag_latest_trace_no_traces(self, mock_mlflow):
        """Test when no traces are found."""
        mock_traces = MagicMock()
        mock_traces.__len__ = MagicMock(return_value=0)
        mock_mlflow.search_traces.return_value = mock_traces

        # Should not raise exception
        tag_latest_trace("exp-1", "run-1", "scenario-1")

        mock_mlflow.set_trace_tag.assert_not_called()

    def test_tag_latest_trace_multiple_traces(self, mock_mlflow):
        """Test when multiple traces are found (should not happen)."""
        mock_traces = MagicMock()
        mock_traces.__len__ = MagicMock(return_value=2)
        mock_mlflow.search_traces.return_value = mock_traces

        # Should not raise exception
        tag_latest_trace("exp-1", "run-1", "scenario-1")

        mock_mlflow.set_trace_tag.assert_not_called()


class TestAOBenchInit:
    """Test AOBench initialization."""

    def test_init_with_scenario_uri_only(self):
        """Test initialization with only scenario URI."""
        client = AOBench(scenario_uri="https://test.com")
        assert client.scenario_uri == "https://test.com"
        assert client.tracking_uri == ""

    def test_init_with_both_uris(self):
        """Test initialization with both URIs."""
        client = AOBench(
            scenario_uri="https://test.com", tracking_uri="https://mlflow.com"
        )
        assert client.scenario_uri == "https://test.com"
        assert client.tracking_uri == "https://mlflow.com"


class TestAOBenchRun:
    """Test synchronous run method."""

    def test_run_without_tracking(self, mock_scenario_uri, simple_sync_function):
        """Test run without tracking context."""
        client = AOBench(scenario_uri=mock_scenario_uri)

        result = client.run(
            func=simple_sync_function, scenario_id="scenario-1", prompt="Test prompt"
        )

        assert result["scenario_id"] == "scenario-1"
        assert result["answer"] == "Response to: Test prompt"

    def test_run_with_tracking(
        self,
        mock_scenario_uri,
        simple_sync_function,
        sample_tracking_context,
        mock_mlflow,
    ):
        """Test run with tracking context."""
        client = AOBench(scenario_uri=mock_scenario_uri)

        # Mock start_span as context manager
        mock_span = MagicMock()
        mock_mlflow.start_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_mlflow.start_span.return_value.__exit__ = Mock(return_value=False)

        result = client.run(
            func=simple_sync_function,
            scenario_id="scenario-1",
            run_name="Test Run",
            tracking_context=sample_tracking_context,
            prompt="Test prompt",
        )

        assert result["scenario_id"] == "scenario-1"
        assert result["answer"] == "Response to: Test prompt"
        mock_mlflow.set_tag.assert_called_once_with("mlflow.runName", "Test Run")
        mock_mlflow.start_span.assert_called_once_with(name="scenario-1")

    def test_run_with_post_process(self, mock_scenario_uri, simple_sync_function):
        """Test run with post-processing function."""
        client = AOBench(scenario_uri=mock_scenario_uri)

        def extract_upper(result):
            return result.upper()

        result = client.run(
            func=simple_sync_function,
            scenario_id="scenario-1",
            post_process=extract_upper,
            prompt="Test prompt",
        )

        assert result["scenario_id"] == "scenario-1"
        assert result["answer"] == "RESPONSE TO: TEST PROMPT"

    def test_run_exception_handling(self, mock_scenario_uri):
        """Test run handles exceptions properly."""
        client = AOBench(scenario_uri=mock_scenario_uri)

        def failing_func(**kwargs):
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            client.run(func=failing_func, scenario_id="scenario-1", prompt="Test")


class TestAOBenchArun:
    """Test asynchronous arun method."""

    @pytest.mark.asyncio
    async def test_arun_without_tracking(
        self, mock_scenario_uri, simple_async_function
    ):
        """Test async run without tracking context."""
        client = AOBench(scenario_uri=mock_scenario_uri)

        result = await client.arun(
            afunc=simple_async_function, scenario_id="scenario-1", prompt="Test prompt"
        )

        assert result["scenario_id"] == "scenario-1"
        assert result["answer"] == "Async response to: Test prompt"

    @pytest.mark.asyncio
    async def test_arun_with_tracking(
        self,
        mock_scenario_uri,
        simple_async_function,
        sample_tracking_context,
        mock_mlflow,
    ):
        """Test async run with tracking context."""
        client = AOBench(scenario_uri=mock_scenario_uri)

        # Mock start_span as context manager
        mock_span = MagicMock()
        mock_mlflow.start_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_mlflow.start_span.return_value.__exit__ = Mock(return_value=False)

        result = await client.arun(
            afunc=simple_async_function,
            scenario_id="scenario-1",
            run_name="Async Test Run",
            tracking_context=sample_tracking_context,
            prompt="Test prompt",
        )

        assert result["scenario_id"] == "scenario-1"
        assert result["answer"] == "Async response to: Test prompt"
        mock_mlflow.set_tag.assert_called_once_with("mlflow.runName", "Async Test Run")

    @pytest.mark.asyncio
    async def test_arun_with_post_process(
        self, mock_scenario_uri, simple_async_function
    ):
        """Test async run with post-processing function."""
        client = AOBench(scenario_uri=mock_scenario_uri)

        def extract_upper(result):
            return str(result).upper()

        result = await client.arun(
            afunc=simple_async_function,
            scenario_id="scenario-1",
            post_process=extract_upper,
            prompt="Test",
        )

        assert result["scenario_id"] == "scenario-1"
        assert isinstance(result["answer"], str)
        assert result["answer"].isupper()

    @pytest.mark.asyncio
    async def test_arun_exception_handling(self, mock_scenario_uri):
        """Test async run handles exceptions properly."""
        client = AOBench(scenario_uri=mock_scenario_uri)

        async def failing_afunc(**kwargs):
            raise ValueError("Async test error")

        with pytest.raises(ValueError, match="Async test error"):
            await client.arun(
                afunc=failing_afunc, scenario_id="scenario-1", prompt="Test"
            )


class TestAOBenchRunNameHandling:
    """Test run_name parameter handling."""

    def test_run_with_empty_run_name(
        self,
        mock_scenario_uri,
        simple_sync_function,
        sample_tracking_context,
        mock_mlflow,
    ):
        """Test that empty run_name doesn't set tag."""
        client = AOBench(scenario_uri=mock_scenario_uri)

        mock_span = MagicMock()
        mock_mlflow.start_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_mlflow.start_span.return_value.__exit__ = Mock(return_value=False)

        client.run(
            func=simple_sync_function,
            scenario_id="scenario-1",
            run_name="",
            tracking_context=sample_tracking_context,
            prompt="Test",
        )

        mock_mlflow.set_tag.assert_not_called()

    @pytest.mark.asyncio
    async def test_arun_with_empty_run_name(
        self,
        mock_scenario_uri,
        simple_async_function,
        sample_tracking_context,
        mock_mlflow,
    ):
        """Test that empty run_name doesn't set tag in async."""
        client = AOBench(scenario_uri=mock_scenario_uri)

        mock_span = MagicMock()
        mock_mlflow.start_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_mlflow.start_span.return_value.__exit__ = Mock(return_value=False)

        await client.arun(
            afunc=simple_async_function,
            scenario_id="scenario-1",
            run_name="",
            tracking_context=sample_tracking_context,
            prompt="Test",
        )

        mock_mlflow.set_tag.assert_not_called()
