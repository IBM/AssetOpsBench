"""Integration tests for AOBench HTTP API methods."""

import pytest
import httpx
import respx
from scenario_client.client import AOBench


class TestScenarioTypes:
    """Test scenario_types method."""

    @respx.mock
    def test_scenario_types_success(self, mock_scenario_uri, sample_scenario_types):
        """Test successful retrieval of scenario types."""
        route = respx.get(f"{mock_scenario_uri}/scenario-types").mock(
            return_value=httpx.Response(200, json=sample_scenario_types)
        )

        client = AOBench(scenario_uri=mock_scenario_uri)
        result = client.scenario_types()

        assert route.called
        assert result == sample_scenario_types
        assert "types" in result
        assert len(result["types"]) == 3

    @respx.mock
    def test_scenario_types_http_error(self, mock_scenario_uri):
        """Test scenario_types handles HTTP errors."""
        respx.get(f"{mock_scenario_uri}/scenario-types").mock(
            return_value=httpx.Response(500, json={"error": "Server error"})
        )

        client = AOBench(scenario_uri=mock_scenario_uri)

        with pytest.raises(httpx.HTTPStatusError):
            client.scenario_types()

    @respx.mock
    def test_scenario_types_network_error(self, mock_scenario_uri):
        """Test scenario_types handles network errors."""
        respx.get(f"{mock_scenario_uri}/scenario-types").mock(
            side_effect=httpx.ConnectError("Connection failed")
        )

        client = AOBench(scenario_uri=mock_scenario_uri)

        with pytest.raises(httpx.ConnectError):
            client.scenario_types()


class TestScenarioSet:
    """Test scenario_set method."""

    @respx.mock
    def test_scenario_set_without_tracking(
        self, mock_scenario_uri, sample_scenario_set
    ):
        """Test loading scenario set without tracking."""
        route = respx.get(
            f"{mock_scenario_uri}/scenario-set/test-set", params={"tracking": False}
        ).mock(return_value=httpx.Response(200, json=sample_scenario_set))

        client = AOBench(scenario_uri=mock_scenario_uri)
        scenario_set, tracking_context = client.scenario_set(
            scenario_set_id="test-set", tracking=False
        )

        assert route.called
        assert scenario_set["title"] == "Test Scenario Set"
        assert len(scenario_set["scenarios"]) == 2
        assert tracking_context is None

    @respx.mock
    def test_scenario_set_with_tracking(
        self, mock_scenario_uri, sample_scenario_set_with_tracking, mock_mlflow
    ):
        """Test loading scenario set with tracking enabled."""
        route = respx.get(
            f"{mock_scenario_uri}/scenario-set/test-set", params={"tracking": True}
        ).mock(return_value=httpx.Response(200, json=sample_scenario_set_with_tracking))

        client = AOBench(scenario_uri=mock_scenario_uri)
        scenario_set, tracking_context = client.scenario_set(
            scenario_set_id="test-set", tracking=True
        )

        assert route.called
        assert scenario_set["title"] == "Test Scenario Set"
        assert tracking_context is not None
        assert tracking_context.experiment_id == "test-experiment-123"
        assert tracking_context.run_id == "test-run-456"

        # Verify MLflow setup
        mock_mlflow.set_tracking_uri.assert_called_once()
        mock_mlflow.langchain.autolog.assert_called_once()
        mock_mlflow.set_experiment.assert_called_once()
        mock_mlflow.start_run.assert_called_once()

    @respx.mock
    def test_scenario_set_with_tracking_uri_override(
        self,
        mock_scenario_uri,
        mock_tracking_uri,
        sample_scenario_set_with_tracking,
        mock_mlflow,
    ):
        """Test that client tracking_uri overrides server tracking_uri."""
        route = respx.get(
            f"{mock_scenario_uri}/scenario-set/test-set", params={"tracking": True}
        ).mock(return_value=httpx.Response(200, json=sample_scenario_set_with_tracking))

        client = AOBench(scenario_uri=mock_scenario_uri, tracking_uri=mock_tracking_uri)
        scenario_set, tracking_context = client.scenario_set(
            scenario_set_id="test-set", tracking=True
        )

        assert route.called
        assert tracking_context is not None
        assert tracking_context.uri == mock_tracking_uri

    @respx.mock
    def test_scenario_set_http_error(self, mock_scenario_uri):
        """Test scenario_set handles HTTP errors."""
        respx.get(
            f"{mock_scenario_uri}/scenario-set/test-set", params={"tracking": False}
        ).mock(return_value=httpx.Response(404, json={"error": "Not found"}))

        client = AOBench(scenario_uri=mock_scenario_uri)

        with pytest.raises(httpx.HTTPStatusError):
            client.scenario_set(scenario_set_id="test-set", tracking=False)


class TestGrade:
    """Test grade method."""

    @respx.mock
    def test_grade_without_tracking(
        self, mock_scenario_uri, sample_answers, sample_grade_response
    ):
        """Test grading without tracking context."""
        route = respx.post(f"{mock_scenario_uri}/scenario-set/test-set/grade").mock(
            return_value=httpx.Response(200, json=sample_grade_response)
        )

        client = AOBench(scenario_uri=mock_scenario_uri)
        result = client.grade(
            scenario_set_id="test-set", answers=sample_answers, tracking_context=None
        )

        assert route.called
        assert result["score"] == 0.95
        assert result["total_scenarios"] == 2
        assert result["correct"] == 2

    @respx.mock
    def test_grade_with_tracking(
        self,
        mock_scenario_uri,
        sample_answers,
        sample_tracking_context,
        sample_grade_response,
        mock_mlflow,
    ):
        """Test grading with tracking context."""
        route = respx.post(f"{mock_scenario_uri}/scenario-set/test-set/grade").mock(
            return_value=httpx.Response(200, json=sample_grade_response)
        )

        client = AOBench(scenario_uri=mock_scenario_uri)
        result = client.grade(
            scenario_set_id="test-set",
            answers=sample_answers,
            tracking_context=sample_tracking_context,
        )

        assert route.called
        assert result["score"] == 0.95

        # Verify MLflow run was ended
        mock_mlflow.end_run.assert_called_once()

        # Verify request included tracking context
        request = route.calls.last.request
        request_json = request.read().decode()
        assert "tracking_context" in request_json

    @respx.mock
    def test_grade_http_error(self, mock_scenario_uri, sample_answers):
        """Test grade handles HTTP errors."""
        respx.post(f"{mock_scenario_uri}/scenario-set/test-set/grade").mock(
            return_value=httpx.Response(400, json={"error": "Bad request"})
        )

        client = AOBench(scenario_uri=mock_scenario_uri)

        with pytest.raises(httpx.HTTPStatusError):
            client.grade(
                scenario_set_id="test-set",
                answers=sample_answers,
                tracking_context=None,
            )

    @respx.mock
    def test_grade_timeout(self, mock_scenario_uri, sample_answers):
        """Test grade handles timeout errors."""
        respx.post(f"{mock_scenario_uri}/scenario-set/test-set/grade").mock(
            side_effect=httpx.TimeoutException("Request timeout")
        )

        client = AOBench(scenario_uri=mock_scenario_uri)

        with pytest.raises(httpx.TimeoutException):
            client.grade(
                scenario_set_id="test-set",
                answers=sample_answers,
                tracking_context=None,
            )


class TestDeferredGrading:
    """Test deferred grading methods."""

    @respx.mock
    def test_deferred_grading_submit(
        self, mock_scenario_uri, sample_answers, sample_deferred_grading_response
    ):
        """Test submitting for deferred grading."""
        route = respx.post(
            f"{mock_scenario_uri}/scenario-set/test-set/deferred-grading"
        ).mock(return_value=httpx.Response(200, json=sample_deferred_grading_response))

        client = AOBench(scenario_uri=mock_scenario_uri)
        result = client.deferred_grading(
            scenario_set_id="test-set", answers=sample_answers, tracking_context=None
        )

        assert route.called
        assert result["grading_id"] == "grading-789"
        assert result["status"] == "pending"

    @respx.mock
    def test_deferred_grading_with_tracking(
        self,
        mock_scenario_uri,
        sample_answers,
        sample_tracking_context,
        sample_deferred_grading_response,
        mock_mlflow,
    ):
        """Test deferred grading with tracking context."""
        route = respx.post(
            f"{mock_scenario_uri}/scenario-set/test-set/deferred-grading"
        ).mock(return_value=httpx.Response(200, json=sample_deferred_grading_response))

        client = AOBench(scenario_uri=mock_scenario_uri)
        result = client.deferred_grading(
            scenario_set_id="test-set",
            answers=sample_answers,
            tracking_context=sample_tracking_context,
        )

        assert route.called
        assert result["grading_id"] == "grading-789"

        # Verify MLflow run was ended
        mock_mlflow.end_run.assert_called_once()

    @respx.mock
    def test_deferred_grading_status(self, mock_scenario_uri):
        """Test checking deferred grading status."""
        status_response = {"status": "processing", "progress": 0.5}
        route = respx.get(
            f"{mock_scenario_uri}/deferred-grading/grading-789/status"
        ).mock(return_value=httpx.Response(200, json=status_response))

        client = AOBench(scenario_uri=mock_scenario_uri)
        result = client.deferred_grading_status(grading_id="grading-789")

        assert route.called
        assert result["status"] == "processing"
        assert result["progress"] == 0.5

    @respx.mock
    def test_deferred_grading_result(self, mock_scenario_uri, sample_grade_response):
        """Test retrieving deferred grading result."""
        route = respx.get(
            f"{mock_scenario_uri}/deferred-grading/grading-789/result"
        ).mock(return_value=httpx.Response(200, json=sample_grade_response))

        client = AOBench(scenario_uri=mock_scenario_uri)
        result = client.deferred_grading_result(grading_id="grading-789")

        assert route.called
        assert result["score"] == 0.95
        assert result["total_scenarios"] == 2

    @respx.mock
    def test_deferred_grading_status_not_found(self, mock_scenario_uri):
        """Test status check for non-existent grading ID."""
        respx.get(f"{mock_scenario_uri}/deferred-grading/invalid-id/status").mock(
            return_value=httpx.Response(404, json={"error": "Not found"})
        )

        client = AOBench(scenario_uri=mock_scenario_uri)

        with pytest.raises(httpx.HTTPStatusError):
            client.deferred_grading_status(grading_id="invalid-id")

    @respx.mock
    def test_deferred_grading_result_not_ready(self, mock_scenario_uri):
        """Test result retrieval when grading not complete."""
        respx.get(f"{mock_scenario_uri}/deferred-grading/grading-789/result").mock(
            return_value=httpx.Response(425, json={"error": "Not ready"})
        )

        client = AOBench(scenario_uri=mock_scenario_uri)

        with pytest.raises(httpx.HTTPStatusError):
            client.deferred_grading_result(grading_id="grading-789")


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    @respx.mock
    def test_complete_workflow_without_tracking(
        self,
        mock_scenario_uri,
        sample_scenario_set,
        sample_grade_response,
        simple_sync_function,
    ):
        """Test complete workflow from loading to grading."""
        # Mock scenario set endpoint
        respx.get(
            f"{mock_scenario_uri}/scenario-set/test-set", params={"tracking": False}
        ).mock(return_value=httpx.Response(200, json=sample_scenario_set))

        # Mock grading endpoint
        respx.post(f"{mock_scenario_uri}/scenario-set/test-set/grade").mock(
            return_value=httpx.Response(200, json=sample_grade_response)
        )

        # Execute workflow
        client = AOBench(scenario_uri=mock_scenario_uri)

        # Load scenarios
        scenario_set, tracking_context = client.scenario_set(
            scenario_set_id="test-set", tracking=False
        )

        # Run scenarios
        answers = []
        for scenario in scenario_set["scenarios"]:
            answer = client.run(
                func=simple_sync_function,
                scenario_id=scenario["id"],
                tracking_context=tracking_context,
                **scenario["inputs"],
            )
            answers.append(answer)

        # Grade
        result = client.grade(
            scenario_set_id="test-set",
            answers=answers,
            tracking_context=tracking_context,
        )

        assert len(answers) == 2
        assert result["score"] == 0.95

    @respx.mock
    def test_complete_deferred_workflow(
        self,
        mock_scenario_uri,
        sample_scenario_set,
        sample_deferred_grading_response,
        sample_grade_response,
        simple_sync_function,
    ):
        """Test complete deferred grading workflow."""
        # Mock scenario set endpoint
        respx.get(
            f"{mock_scenario_uri}/scenario-set/test-set", params={"tracking": False}
        ).mock(return_value=httpx.Response(200, json=sample_scenario_set))

        # Mock deferred grading submission
        respx.post(f"{mock_scenario_uri}/scenario-set/test-set/deferred-grading").mock(
            return_value=httpx.Response(200, json=sample_deferred_grading_response)
        )

        # Mock status check
        respx.get(f"{mock_scenario_uri}/deferred-grading/grading-789/status").mock(
            return_value=httpx.Response(200, json={"status": "completed"})
        )

        # Mock result retrieval
        respx.get(f"{mock_scenario_uri}/deferred-grading/grading-789/result").mock(
            return_value=httpx.Response(200, json=sample_grade_response)
        )

        # Execute workflow
        client = AOBench(scenario_uri=mock_scenario_uri)

        # Load and run scenarios
        scenario_set, _ = client.scenario_set("test-set", tracking=False)
        answers = [
            client.run(func=simple_sync_function, scenario_id=s["id"], **s["inputs"])
            for s in scenario_set["scenarios"]
        ]

        # Submit for deferred grading
        deferred_response = client.deferred_grading("test-set", answers, None)
        grading_id = deferred_response["grading_id"]

        # Check status
        status = client.deferred_grading_status(grading_id)
        assert status["status"] == "completed"

        # Get result
        result = client.deferred_grading_result(grading_id)
        assert result["score"] == 0.95
