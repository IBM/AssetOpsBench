"""Pydantic result models and static data for the TSFM MCP server."""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel


# ── Static data ───────────────────────────────────────────────────────────────

_AI_TASKS = [
    {
        "task_id": "tsfm_integrated_tsad",
        "task_description": "Time series Anomaly detection",
    },
    {
        "task_id": "tsfm_forecasting",
        "task_description": "Time series Multivariate Forecasting",
    },
    {
        "task_id": "tsfm_forecasting_finetune",
        "task_description": "Finetuning of Multivariate Forecasting models",
    },
    {
        "task_id": "tsfm_forecasting_evaluation",
        "task_description": "Evaluation of Forecasting models",
    },
]

_TSFM_MODELS = [
    {
        "model_id": "ttm_96_28",
        "model_checkpoint": "ttm_96_28",
        "model_description": "Pretrained forecasting model with context length 96",
    },
    {
        "model_id": "ttm_512_96",
        "model_checkpoint": "ttm_512_96",
        "model_description": "Pretrained forecasting model with context length 512",
    },
    {
        "model_id": "ttm_energy_96_28",
        "model_checkpoint": "ttm_96_28",
        "model_description": "Pretrained forecasting model tuned on energy data with context length 96",
    },
    {
        "model_id": "ttm_energy_512_96",
        "model_checkpoint": "ttm_512_96",
        "model_description": "Pretrained forecasting model tuned on energy data with context length 512",
    },
]


# ── Result models ─────────────────────────────────────────────────────────────


class ErrorResult(BaseModel):
    error: str


class AITaskEntry(BaseModel):
    task_id: str
    task_description: str


class AITasksResult(BaseModel):
    tasks: List[AITaskEntry]


class TSFMModelEntry(BaseModel):
    model_id: str
    model_checkpoint: str
    model_description: str


class TSFMModelsResult(BaseModel):
    models: List[TSFMModelEntry]


class ForecastingResult(BaseModel):
    status: str
    results_file: str
    dataquality_summary: Optional[Any] = None
    message: str


class FinetuningResult(BaseModel):
    status: str
    model_checkpoint: str
    results_file: str
    message: str


class TSADResult(BaseModel):
    status: str
    results_file: str
    total_records: int
    anomaly_count: int
    columns: List[str]
    message: str
