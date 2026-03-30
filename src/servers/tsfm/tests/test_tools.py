"""Tests for the TSFM MCP server tools.

Unit tests cover input validation and static responses (no ML deps required).
Integration tests are gated by requires_tsfm and need tsfm_public installed
along with PATH_TO_MODELS_DIR / PATH_TO_DATASETS_DIR set.
"""

from __future__ import annotations

import pytest

from servers.tsfm.main import mcp
from .conftest import call_tool, requires_tsfm


# ── get_ai_tasks ──────────────────────────────────────────────────────────────

class TestGetAITasks:
    @pytest.mark.anyio
    async def test_returns_tasks_list(self):
        data = await call_tool(mcp, "get_ai_tasks", {})
        assert "tasks" in data
        assert isinstance(data["tasks"], list)
        assert len(data["tasks"]) == 4

    @pytest.mark.anyio
    async def test_contains_expected_task_ids(self):
        data = await call_tool(mcp, "get_ai_tasks", {})
        ids = {t["task_id"] for t in data["tasks"]}
        assert "tsfm_forecasting" in ids
        assert "tsfm_integrated_tsad" in ids
        assert "tsfm_forecasting_finetune" in ids

    @pytest.mark.anyio
    async def test_each_task_has_description(self):
        data = await call_tool(mcp, "get_ai_tasks", {})
        for task in data["tasks"]:
            assert task.get("task_description"), f"missing description on {task}"


# ── get_tsfm_models ───────────────────────────────────────────────────────────

class TestGetTSFMModels:
    @pytest.mark.anyio
    async def test_returns_models_list(self):
        data = await call_tool(mcp, "get_tsfm_models", {})
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) == 4

    @pytest.mark.anyio
    async def test_contains_expected_model_ids(self):
        data = await call_tool(mcp, "get_tsfm_models", {})
        ids = {m["model_id"] for m in data["models"]}
        assert "ttm_96_28" in ids
        assert "ttm_512_96" in ids

    @pytest.mark.anyio
    async def test_each_model_has_checkpoint_and_description(self):
        data = await call_tool(mcp, "get_tsfm_models", {})
        for model in data["models"]:
            assert model.get("model_checkpoint"), f"missing checkpoint on {model}"
            assert model.get("model_description"), f"missing description on {model}"


# ── run_tsfm_forecasting — input validation ───────────────────────────────────

class TestRunTSFMForecastingValidation:
    @pytest.mark.anyio
    async def test_empty_dataset_path_returns_error(self):
        data = await call_tool(
            mcp, "run_tsfm_forecasting",
            {"dataset_path": "", "timestamp_column": "ts", "target_columns": ["val"]},
        )
        assert "error" in data
        assert "dataset_path" in data["error"]

    @pytest.mark.anyio
    async def test_empty_target_columns_returns_error(self):
        data = await call_tool(
            mcp, "run_tsfm_forecasting",
            {"dataset_path": "/tmp/data.csv", "timestamp_column": "ts", "target_columns": []},
        )
        assert "error" in data
        assert "target_columns" in data["error"]

    @pytest.mark.anyio
    async def test_missing_deps_returns_error(self):
        # tsfm_public is not expected to be installed in the CI/MCP environment.
        # If it IS installed this test is a no-op (the import succeeds).
        data = await call_tool(
            mcp, "run_tsfm_forecasting",
            {
                "dataset_path": "/nonexistent/data.csv",
                "timestamp_column": "Timestamp",
                "target_columns": ["sensor_1"],
            },
        )
        # Either tsfm_public is missing (error about deps) or the file is missing
        # (runtime error). Either way, the tool must return {"error": ...}.
        assert "error" in data


# ── run_tsfm_finetuning — input validation ────────────────────────────────────

class TestRunTSFMFinetuningValidation:
    @pytest.mark.anyio
    async def test_empty_dataset_path_returns_error(self):
        data = await call_tool(
            mcp, "run_tsfm_finetuning",
            {"dataset_path": "", "timestamp_column": "ts", "target_columns": ["val"]},
        )
        assert "error" in data
        assert "dataset_path" in data["error"]

    @pytest.mark.anyio
    async def test_empty_target_columns_returns_error(self):
        data = await call_tool(
            mcp, "run_tsfm_finetuning",
            {"dataset_path": "/tmp/data.csv", "timestamp_column": "ts", "target_columns": []},
        )
        assert "error" in data
        assert "target_columns" in data["error"]


# ── run_tsad — input validation ───────────────────────────────────────────────

class TestRunTSADValidation:
    @pytest.mark.anyio
    async def test_empty_dataset_path_returns_error(self):
        data = await call_tool(
            mcp, "run_tsad",
            {
                "dataset_path": "",
                "tsfm_output_json": "/tmp/pred.json",
                "timestamp_column": "ts",
                "target_columns": ["val"],
            },
        )
        assert "error" in data
        assert "dataset_path" in data["error"]

    @pytest.mark.anyio
    async def test_empty_tsfm_output_json_returns_error(self):
        data = await call_tool(
            mcp, "run_tsad",
            {
                "dataset_path": "/tmp/data.csv",
                "tsfm_output_json": "",
                "timestamp_column": "ts",
                "target_columns": ["val"],
            },
        )
        assert "error" in data
        assert "tsfm_output_json" in data["error"]

    @pytest.mark.anyio
    async def test_invalid_task_returns_error(self):
        data = await call_tool(
            mcp, "run_tsad",
            {
                "dataset_path": "/tmp/data.csv",
                "tsfm_output_json": "/tmp/pred.json",
                "timestamp_column": "ts",
                "target_columns": ["val"],
                "task": "invalid_task",
            },
        )
        assert "error" in data
        assert "task" in data["error"]

    @pytest.mark.anyio
    async def test_empty_target_columns_returns_error(self):
        data = await call_tool(
            mcp, "run_tsad",
            {
                "dataset_path": "/tmp/data.csv",
                "tsfm_output_json": "/tmp/pred.json",
                "timestamp_column": "ts",
                "target_columns": [],
            },
        )
        assert "error" in data
        assert "target_columns" in data["error"]


# ── run_integrated_tsad — input validation ────────────────────────────────────

class TestRunIntegratedTSADValidation:
    @pytest.mark.anyio
    async def test_empty_dataset_path_returns_error(self):
        data = await call_tool(
            mcp, "run_integrated_tsad",
            {"dataset_path": "", "timestamp_column": "ts", "target_columns": ["val"]},
        )
        assert "error" in data
        assert "dataset_path" in data["error"]

    @pytest.mark.anyio
    async def test_empty_target_columns_returns_error(self):
        data = await call_tool(
            mcp, "run_integrated_tsad",
            {"dataset_path": "/tmp/data.csv", "timestamp_column": "ts", "target_columns": []},
        )
        assert "error" in data
        assert "target_columns" in data["error"]


# ── Integration tests (requires tsfm_public) ─────────────────────────────────

@requires_tsfm
class TestTSFMForecastingIntegration:
    @pytest.mark.anyio
    async def test_forecasting_returns_results_file(self, tmp_path):
        """Smoke test: run inference on a small synthetic CSV dataset."""
        import pandas as pd
        import numpy as np

        # Create a small synthetic sine-wave CSV
        n = 200
        df = pd.DataFrame({
            "Timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "sensor_1": np.sin(np.linspace(0, 4 * np.pi, n)),
        })
        csv_path = str(tmp_path / "synthetic.csv")
        df.to_csv(csv_path, index=False)

        data = await call_tool(
            mcp, "run_tsfm_forecasting",
            {
                "dataset_path": csv_path,
                "timestamp_column": "Timestamp",
                "target_columns": ["sensor_1"],
                "model_checkpoint": "ttm_96_28",
                "frequency_sampling": "15_minutes",
            },
        )
        assert "error" not in data, data.get("error")
        assert data["status"] == "success"
        assert data["results_file"]


@requires_tsfm
class TestIntegratedTSADIntegration:
    @pytest.mark.anyio
    async def test_integrated_tsad_returns_csv(self, tmp_path):
        """Smoke test: run integrated TSAD on a small synthetic dataset."""
        import pandas as pd
        import numpy as np

        n = 300
        df = pd.DataFrame({
            "Timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "sensor_1": np.sin(np.linspace(0, 6 * np.pi, n)) + np.random.randn(n) * 0.05,
        })
        csv_path = str(tmp_path / "synthetic_ad.csv")
        df.to_csv(csv_path, index=False)

        data = await call_tool(
            mcp, "run_integrated_tsad",
            {
                "dataset_path": csv_path,
                "timestamp_column": "Timestamp",
                "target_columns": ["sensor_1"],
                "model_checkpoint": "ttm_96_28",
                "frequency_sampling": "15_minutes",
            },
        )
        assert "error" not in data, data.get("error")
        assert data["status"] == "success"
        assert data["results_file"]
        assert data["total_records"] > 0
