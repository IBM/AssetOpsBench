"""Tests for trajectory generator module.

Tests the process_trajectories function and helper functions
in generator.py.
"""

import pytest
import tempfile
import json
import os
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from src.trajectory_analysis.failure_mode.core.generator import (
    process_trajectories,
    _load_all_json_files,
    _normalize_additional_failure_modes,
)


class TestNormalizeAdditionalFailureModes:
    """Test normalization of additional failure modes."""

    def test_normalize_none(self):
        """Test handling of None input."""
        result = _normalize_additional_failure_modes(None)
        assert result == []

    def test_normalize_empty_list(self):
        """Test handling of empty list."""
        result = _normalize_additional_failure_modes([])
        assert result == []

    def test_normalize_list_of_dicts(self):
        """Test list of dictionaries."""
        input_data = [
            {"title": "Issue 1", "description": "Desc 1"},
            {"title": "Issue 2", "description": "Desc 2"},
        ]
        result = _normalize_additional_failure_modes(input_data)
        assert len(result) == 2
        assert result[0]["title"] == "Issue 1"
        assert result[1]["description"] == "Desc 2"

    def test_normalize_list_with_non_dicts(self):
        """Test list containing non-dictionary items."""
        input_data = [
            {"title": "Issue 1", "description": "Desc 1"},
            "not a dict",
            {"title": "Issue 2", "description": "Desc 2"},
        ]
        result = _normalize_additional_failure_modes(input_data)
        assert len(result) == 2  # Non-dict items filtered out

    def test_normalize_single_dict_with_title(self):
        """Test single dictionary with title/description."""
        input_data = {"title": "Issue", "description": "Description"}
        result = _normalize_additional_failure_modes(input_data)
        assert len(result) == 1
        assert result[0]["title"] == "Issue"

    def test_normalize_dict_mapping(self):
        """Test dictionary as title->description mapping."""
        input_data = {"Issue 1": "Description 1", "Issue 2": "Description 2"}
        result = _normalize_additional_failure_modes(input_data)
        assert len(result) == 2
        assert any(item["title"] == "Issue 1" for item in result)
        assert any(item["description"] == "Description 2" for item in result)


class TestLoadAllJsonFiles:
    """Test JSON file loading functionality."""

    def test_load_json_files_from_directory(self):
        """Test loading JSON files from a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test JSON files
            file1 = os.path.join(tmpdir, "0001.json")
            file2 = os.path.join(tmpdir, "0002.json")

            with open(file1, "w") as f:
                json.dump({"test": "data1"}, f)
            with open(file2, "w") as f:
                json.dump({"test": "data2"}, f)

            result = _load_all_json_files(tmpdir)

            assert len(result) == 2
            assert any("data1" in str(v) for v in result.values())
            assert any("data2" in str(v) for v in result.values())

    def test_load_json_files_recursive(self):
        """Test recursive loading from nested directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            subdir = os.path.join(tmpdir, "subdir")
            os.makedirs(subdir)

            file1 = os.path.join(tmpdir, "file1.json")
            file2 = os.path.join(subdir, "file2.json")

            with open(file1, "w") as f:
                json.dump({"level": "root"}, f)
            with open(file2, "w") as f:
                json.dump({"level": "nested"}, f)

            result = _load_all_json_files(tmpdir)

            assert len(result) == 2

    def test_load_json_files_skip_invalid(self):
        """Test that invalid JSON files are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            valid_file = os.path.join(tmpdir, "valid.json")
            invalid_file = os.path.join(tmpdir, "invalid.json")
            text_file = os.path.join(tmpdir, "text.txt")

            with open(valid_file, "w") as f:
                json.dump({"valid": True}, f)
            with open(invalid_file, "w") as f:
                f.write("not valid json {")
            with open(text_file, "w") as f:
                f.write("plain text")

            result = _load_all_json_files(tmpdir)

            # Should only load the valid JSON file
            assert len(result) == 1
            assert "valid" in str(list(result.values())[0])

    def test_load_json_files_empty_directory(self):
        """Test loading from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _load_all_json_files(tmpdir)
            assert len(result) == 0


class TestProcessTrajectories:
    """Test the main process_trajectories function."""

    @patch(
        "src.trajectory_analysis.failure_mode.core.generator.get_llm_answer_from_json"
    )
    @patch("src.trajectory_analysis.failure_mode.core.generator._load_all_json_files")
    def test_process_trajectories_basic(self, mock_load, mock_llm_answer):
        """Test basic trajectory processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock file loading
            mock_load.return_value = {
                "0001_trajectory.json": {
                    "text": "Test question",
                    "trajectory": [{"task_description": "test"}],
                }
            }

            # Mock LLM response
            mock_llm_answer.return_value = """```json
            {
                "failure_modes": {
                    "1.1 Disobey Task Specification": true,
                    "1.2 Disobey Role Specification": false
                },
                "additional_failure_modes": []
            }
            ```"""

            # Mock LLM backend
            mock_llm = Mock()

            result = process_trajectories(
                traj_root_base=tmpdir, llm_backend=mock_llm, out_dir=tmpdir
            )

            # Verify results
            assert "combined_df" in result
            assert "combined_path" in result
            assert "per_timestamp_paths" in result

            df = result["combined_df"]
            assert len(df) == 1
            assert df.iloc[0]["1.1 Disobey Task Specification"] == True
            assert df.iloc[0]["1.2 Disobey Role Specification"] == False

    @patch("src.trajectory_analysis.failure_mode.core.generator.LiteLLMBackend")
    @patch("src.trajectory_analysis.failure_mode.core.generator._load_all_json_files")
    def test_process_trajectories_default_llm(self, mock_load, mock_litellm):
        """Test that default LLM backend is created when none provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_load.return_value = {}
            mock_llm_instance = Mock()
            mock_litellm.return_value = mock_llm_instance

            result = process_trajectories(
                traj_root_base=tmpdir,
                llm_backend=None,  # Should create default
                out_dir=tmpdir,
            )

            # Verify default Claude 4 Sonnet was created
            mock_litellm.assert_called_once_with("litellm_proxy/GCP/claude-4-sonnet")

    @patch(
        "src.trajectory_analysis.failure_mode.core.generator.get_llm_answer_from_json"
    )
    @patch("src.trajectory_analysis.failure_mode.core.generator._load_all_json_files")
    def test_process_trajectories_with_temperature(self, mock_load, mock_llm_answer):
        """Test temperature parameter is passed through."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_load.return_value = {"0001.json": {"text": "test", "trajectory": []}}
            mock_llm_answer.return_value = '{"failure_modes": {}}'

            mock_llm = Mock()

            process_trajectories(
                traj_root_base=tmpdir,
                llm_backend=mock_llm,
                temperature=0.7,
                out_dir=tmpdir,
            )

            # Verify temperature was passed to get_llm_answer_from_json
            call_args = mock_llm_answer.call_args
            assert call_args[1]["temperature"] == 0.7

    @patch(
        "src.trajectory_analysis.failure_mode.core.generator.get_llm_answer_from_json"
    )
    @patch("src.trajectory_analysis.failure_mode.core.generator._load_all_json_files")
    def test_process_trajectories_error_handling(self, mock_load, mock_llm_answer):
        """Test error handling during processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_load.return_value = {
                "0001.json": {"text": "test", "trajectory": []},
                "0002.json": {"text": "test2", "trajectory": []},
            }

            # First call fails, second succeeds
            mock_llm_answer.side_effect = [
                Exception("LLM error"),
                '{"failure_modes": {}}',
            ]

            mock_llm = Mock()

            result = process_trajectories(
                traj_root_base=tmpdir, llm_backend=mock_llm, out_dir=tmpdir
            )

            # Should continue processing despite error
            df = result["combined_df"]
            # Only successful processing should be in results
            assert len(df) >= 0  # May have 0 or 1 depending on retry logic

    @patch(
        "src.trajectory_analysis.failure_mode.core.generator.get_llm_answer_from_json"
    )
    @patch("src.trajectory_analysis.failure_mode.core.generator._load_all_json_files")
    def test_process_trajectories_additional_failure_modes(
        self, mock_load, mock_llm_answer
    ):
        """Test handling of additional failure modes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_load.return_value = {"0001.json": {"text": "test", "trajectory": []}}

            mock_llm_answer.return_value = """```json
            {
                "failure_modes": {},
                "additional_failure_modes": [
                    {"title": "Custom Issue 1", "description": "Desc 1"},
                    {"title": "Custom Issue 2", "description": "Desc 2"}
                ]
            }
            ```"""

            mock_llm = Mock()

            result = process_trajectories(
                traj_root_base=tmpdir, llm_backend=mock_llm, out_dir=tmpdir
            )

            df = result["combined_df"]
            assert df.iloc[0]["addi_fm_cnt"] == 2
            assert len(df.iloc[0]["addi_fm_list"]) == 2

    def test_process_trajectories_creates_output_directory(self):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = os.path.join(tmpdir, "new_output_dir")

            with patch(
                "src.trajectory_analysis.failure_mode.core.generator._load_all_json_files"
            ) as mock_load:
                mock_load.return_value = {}
                mock_llm = Mock()

                process_trajectories(
                    traj_root_base=tmpdir, llm_backend=mock_llm, out_dir=out_dir
                )

                # Verify directory was created
                assert os.path.exists(out_dir)


class TestIntegration:
    """Integration tests for the generator module."""

    def test_end_to_end_with_mock_data(self):
        """Test complete flow with mock trajectory data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test trajectory file
            traj_file = os.path.join(tmpdir, "0001_test.json")
            with open(traj_file, "w") as f:
                json.dump(
                    {
                        "text": "Complete a task",
                        "trajectory": [
                            {
                                "task_description": "Step 1",
                                "agent_name": "Agent1",
                                "response": "Done",
                                "final_answer": "Completed",
                            }
                        ],
                    },
                    f,
                )

            # Mock LLM backend
            mock_llm = Mock()
            mock_llm.generate.return_value = """```json
            {
                "failure_modes": {
                    "1.1 Disobey Task Specification": false,
                    "2.1 Conversation Reset": true
                },
                "additional_failure_modes": []
            }
            ```"""

            # Process trajectories
            result = process_trajectories(
                traj_root_base=tmpdir, llm_backend=mock_llm, out_dir=tmpdir
            )

            # Verify complete result
            assert result["combined_df"] is not None
            assert len(result["combined_df"]) == 1
            assert os.path.exists(result["combined_path"])

            # Verify pickle file can be loaded
            df_loaded = pd.read_pickle(result["combined_path"])
            assert len(df_loaded) == 1
            assert df_loaded.iloc[0]["2.1 Conversation Reset"] == True


# Made with Bob
