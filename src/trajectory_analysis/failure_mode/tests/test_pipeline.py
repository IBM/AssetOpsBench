"""Tests for the failure mode pipeline.

Tests the high-level pipeline orchestration in pipeline.py.
"""

import pytest
import tempfile
from unittest.mock import Mock, patch, MagicMock
from src.trajectory_analysis.failure_mode.core.pipeline import run_failure_mode_pipeline


class TestRunFailureModePipeline:
    """Test the main pipeline function."""

    @patch("src.trajectory_analysis.failure_mode.core.pipeline.failure_mode_reduction")
    @patch("src.trajectory_analysis.failure_mode.core.pipeline.process_trajectories")
    @patch("src.trajectory_analysis.failure_mode.core.pipeline.LiteLLMBackend")
    def test_pipeline_with_default_llm(
        self, mock_litellm, mock_process, mock_reduction
    ):
        """Test pipeline uses default Claude 4 Sonnet when no LLM provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock the LLM backend
            mock_llm_instance = Mock()
            mock_litellm.return_value = mock_llm_instance

            # Mock process_trajectories return
            mock_process.return_value = {
                "combined_path": f"{tmpdir}/combined.pkl",
                "combined_df": Mock(),
                "per_timestamp_paths": [],
            }

            result = run_failure_mode_pipeline(
                traj_root_base=tmpdir, llm_backend=None  # Should use default
            )

            # Verify default LLM was created
            mock_litellm.assert_called_once_with("litellm_proxy/GCP/claude-4-sonnet")

            # Verify process_trajectories was called with the LLM
            mock_process.assert_called_once()
            call_kwargs = mock_process.call_args[1]
            assert call_kwargs["llm_backend"] == mock_llm_instance
            assert call_kwargs["traj_root_base"] == tmpdir

    @patch("src.trajectory_analysis.failure_mode.core.pipeline.failure_mode_reduction")
    @patch("src.trajectory_analysis.failure_mode.core.pipeline.process_trajectories")
    def test_pipeline_with_custom_llm(self, mock_process, mock_reduction):
        """Test pipeline with custom LLM backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create custom LLM mock
            custom_llm = Mock()

            mock_process.return_value = {
                "combined_path": f"{tmpdir}/combined.pkl",
                "combined_df": Mock(),
                "per_timestamp_paths": [],
            }

            result = run_failure_mode_pipeline(
                traj_root_base=tmpdir, llm_backend=custom_llm
            )

            # Verify custom LLM was used
            call_kwargs = mock_process.call_args[1]
            assert call_kwargs["llm_backend"] == custom_llm

    @patch("src.trajectory_analysis.failure_mode.core.pipeline.failure_mode_reduction")
    @patch("src.trajectory_analysis.failure_mode.core.pipeline.process_trajectories")
    def test_pipeline_with_temperature(self, mock_process, mock_reduction):
        """Test pipeline passes temperature parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_llm = Mock()

            mock_process.return_value = {
                "combined_path": f"{tmpdir}/combined.pkl",
                "combined_df": Mock(),
                "per_timestamp_paths": [],
            }

            result = run_failure_mode_pipeline(
                traj_root_base=tmpdir, llm_backend=mock_llm, temperature=0.8
            )

            # Verify temperature was passed
            call_kwargs = mock_process.call_args[1]
            assert call_kwargs["temperature"] == 0.8

    @patch("src.trajectory_analysis.failure_mode.core.pipeline.failure_mode_reduction")
    @patch("src.trajectory_analysis.failure_mode.core.pipeline.process_trajectories")
    def test_pipeline_with_timestamps(self, mock_process, mock_reduction):
        """Test pipeline with custom timestamps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_llm = Mock()
            timestamps = ["2024-01", "2024-02"]

            mock_process.return_value = {
                "combined_path": f"{tmpdir}/combined.pkl",
                "combined_df": Mock(),
                "per_timestamp_paths": [],
            }

            result = run_failure_mode_pipeline(
                traj_root_base=tmpdir, llm_backend=mock_llm, timestamps=timestamps
            )

            # Verify timestamps were passed
            call_kwargs = mock_process.call_args[1]
            assert call_kwargs["timestamps"] == timestamps

    @patch("src.trajectory_analysis.failure_mode.core.pipeline.failure_mode_reduction")
    @patch("src.trajectory_analysis.failure_mode.core.pipeline.process_trajectories")
    def test_pipeline_returns_generation_results(self, mock_process, mock_reduction):
        """Test pipeline returns generation results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_llm = Mock()
            mock_df = Mock()

            expected_gen_result = {
                "combined_path": f"{tmpdir}/combined.pkl",
                "combined_df": mock_df,
                "per_timestamp_paths": [f"{tmpdir}/1_db.pkl"],
            }
            mock_process.return_value = expected_gen_result

            result = run_failure_mode_pipeline(
                traj_root_base=tmpdir, llm_backend=mock_llm
            )

            # Verify result structure
            assert "generation" in result
            assert result["generation"] == expected_gen_result
            assert result["generation"]["combined_df"] == mock_df

    @patch("src.trajectory_analysis.failure_mode.core.pipeline.failure_mode_reduction")
    @patch("src.trajectory_analysis.failure_mode.core.pipeline.process_trajectories")
    def test_pipeline_with_all_parameters(self, mock_process, mock_reduction):
        """Test pipeline with all optional parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_llm = Mock()

            mock_process.return_value = {
                "combined_path": f"{tmpdir}/combined.pkl",
                "combined_df": Mock(),
                "per_timestamp_paths": [],
            }

            result = run_failure_mode_pipeline(
                traj_root_base=tmpdir,
                llm_backend=mock_llm,
                temperature=0.5,
                timestamps=["2024-01"],
                summary_dir="custom_summary",
                model_name="custom-model",
                k=5,
            )

            # Verify all parameters were passed correctly
            call_kwargs = mock_process.call_args[1]
            assert call_kwargs["traj_root_base"] == tmpdir
            assert call_kwargs["llm_backend"] == mock_llm
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["timestamps"] == ["2024-01"]

    @patch("src.trajectory_analysis.failure_mode.core.pipeline.failure_mode_reduction")
    @patch("src.trajectory_analysis.failure_mode.core.pipeline.process_trajectories")
    @patch("src.trajectory_analysis.failure_mode.core.pipeline.LiteLLMBackend")
    def test_pipeline_default_temperature(
        self, mock_litellm, mock_process, mock_reduction
    ):
        """Test pipeline uses default temperature of 0.0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_llm_instance = Mock()
            mock_litellm.return_value = mock_llm_instance

            mock_process.return_value = {
                "combined_path": f"{tmpdir}/combined.pkl",
                "combined_df": Mock(),
                "per_timestamp_paths": [],
            }

            result = run_failure_mode_pipeline(
                traj_root_base=tmpdir
                # No temperature specified, should default to 0.0
            )

            # Verify default temperature
            call_kwargs = mock_process.call_args[1]
            assert call_kwargs["temperature"] == 0.0


class TestPipelineIntegration:
    """Integration tests for the complete pipeline."""

    @patch("src.trajectory_analysis.failure_mode.core.pipeline.failure_mode_reduction")
    @patch("src.trajectory_analysis.failure_mode.core.pipeline.process_trajectories")
    def test_pipeline_end_to_end_mock(self, mock_process, mock_reduction):
        """Test complete pipeline flow with mocked components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock LLM
            mock_llm = Mock()
            mock_llm.generate.return_value = '{"failure_modes": {}}'

            # Create mock DataFrame
            import pandas as pd

            mock_df = pd.DataFrame(
                {
                    "model_id": ["llm_backend"],
                    "counter": [1],
                    "1.1 Disobey Task Specification": [True],
                }
            )

            mock_process.return_value = {
                "combined_path": f"{tmpdir}/combined_mllm_backend_db.pkl",
                "combined_df": mock_df,
                "per_timestamp_paths": [f"{tmpdir}/1_mllm_backend_db.pkl"],
            }

            # Run pipeline
            result = run_failure_mode_pipeline(
                traj_root_base=tmpdir, llm_backend=mock_llm, temperature=0.0
            )

            # Verify complete result structure
            assert "generation" in result
            assert result["generation"]["combined_df"] is not None
            assert len(result["generation"]["combined_df"]) == 1
            assert result["generation"]["combined_path"].endswith(".pkl")


class TestPipelineDocumentation:
    """Test that pipeline function has proper documentation."""

    def test_pipeline_has_docstring(self):
        """Test that pipeline function has comprehensive docstring."""
        assert run_failure_mode_pipeline.__doc__ is not None
        docstring = run_failure_mode_pipeline.__doc__

        # Check for key documentation elements
        assert "Args:" in docstring or "Parameters:" in docstring
        assert "Returns:" in docstring
        assert "Example:" in docstring or "example" in docstring.lower()

    def test_pipeline_signature(self):
        """Test pipeline function signature."""
        import inspect

        sig = inspect.signature(run_failure_mode_pipeline)

        # Verify required parameters
        assert "traj_root_base" in sig.parameters

        # Verify optional parameters with defaults
        assert "llm_backend" in sig.parameters
        assert sig.parameters["llm_backend"].default is None

        assert "temperature" in sig.parameters
        assert sig.parameters["temperature"].default == 0.0


# Made with Bob
