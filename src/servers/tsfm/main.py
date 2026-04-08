"""TSFM (Time Series Foundation Model) MCP Server.

Standalone implementation — no dependency on src/tsfmagent/.
The only external ML dependency is tsfm_public (IBM Granite TSFM):
  https://github.com/IBM-granite/granite-tsfm

Tools:
  get_ai_tasks          – list available AI task types (static, no deps)
  get_tsfm_models       – list available pre-trained model checkpoints (static)
  run_tsfm_forecasting  – zero-shot TTM inference on a dataset
  run_tsfm_finetuning   – few-shot finetuning of a TTM model
  run_tsad              – conformal anomaly detection on TSFM forecasts
  run_integrated_tsad   – end-to-end: forecasting + anomaly detection

Heavy ML dependencies (tsfm_public, transformers, torch) are imported lazily;
the server starts and exposes the static tools even when they are absent.

Required environment variables (path resolution):
  PATH_TO_MODELS_DIR    – directory containing TTM model checkpoint folders
  PATH_TO_DATASETS_DIR  – base directory for resolving relative dataset paths
  PATH_TO_OUTPUTS_DIR   – base directory for resolving output/save paths
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from functools import lru_cache
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from .anomaly import _TimeSeriesAnomalyDetectionConformalWrapper
from .forecasting import (
    _finetune_ttm_hf,
    _find_largest_tsfm_checkpoint_directory,
    _get_ttm_hf_inference,
    _tsfm_data_quality_filter,
)
from .io import (
    _get_dataset_path,
    _get_model_checkpoint_path,
    _get_outputs_path,
    _read_ts_data,
    _write_json_to_temp,
)
from .models import (
    _AI_TASKS,
    _TSFM_MODELS,
    AITaskEntry,
    AITasksResult,
    ErrorResult,
    FinetuningResult,
    ForecastingResult,
    TSADResult,
    TSFMModelEntry,
    TSFMModelsResult,
)

load_dotenv()

_log_level = getattr(
    logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING
)
logging.basicConfig(level=_log_level)
logger = logging.getLogger("tsfm-mcp-server")



# ── Internal helpers ──────────────────────────────────────────────────────────


@lru_cache(maxsize=16)
def _load_model_config(model_checkpoint: str) -> dict:
    """Load and cache model config.json to avoid repeated disk reads."""
    with open(model_checkpoint + "/config.json") as f:
        return json.load(f)


def _build_dataset_config(
    timestamp_column: str,
    target_columns: List[str],
    conditional_columns: Optional[List[str]],
    id_columns: Optional[List[str]],
    frequency_sampling: str,
    autoregressive_modeling: bool,
) -> dict:
    return {
        "column_specifiers": {
            "autoregressive_modeling": autoregressive_modeling,
            "timestamp_column": timestamp_column,
            "conditional_columns": conditional_columns or [],
            "target_columns": target_columns,
        },
        "id_columns": id_columns or [],
        "frequency_sampling": frequency_sampling,
    }


def _tsad_output_to_df(output: dict) -> pd.DataFrame:
    kpi = output.pop("KPI", None)
    df = pd.DataFrame.from_dict({k: np.array(v) for k, v in output.items()})
    if kpi is not None:
        df["KPI"] = (
            kpi[0] if (hasattr(kpi, "__len__") and not isinstance(kpi, str)) else kpi
        )
    return df


# ── FastMCP server ────────────────────────────────────────────────────────────

mcp = FastMCP("tsfm")


# ── Static tools ──────────────────────────────────────────────────────────────


@mcp.tool()
def get_ai_tasks() -> AITasksResult:
    """Returns the list of supported AI task types for time-series analysis.

    Tasks: tsfm_integrated_tsad, tsfm_forecasting, tsfm_forecasting_finetune,
    tsfm_forecasting_evaluation.
    """
    return AITasksResult(tasks=[AITaskEntry(**t) for t in _AI_TASKS])


@mcp.tool()
def get_tsfm_models() -> TSFMModelsResult:
    """Returns the list of available pre-trained TinyTimeMixer (TTM) model checkpoints.

    Models: ttm_96_28 (context_length=96), ttm_512_96 (context_length=512),
    and energy-domain fine-tuned variants of both.
    """
    return TSFMModelsResult(models=[TSFMModelEntry(**m) for m in _TSFM_MODELS])


# ── TSFM Forecasting (zero-shot inference) ────────────────────────────────────


@mcp.tool()
def run_tsfm_forecasting(
    dataset_path: str,
    timestamp_column: str,
    target_columns: List[str],
    model_checkpoint: str = "ttm_96_28",
    forecast_horizon: int = -1,
    conditional_columns: Optional[List[str]] = None,
    id_columns: Optional[List[str]] = None,
    frequency_sampling: str = "oov",
    autoregressive_modeling: bool = True,
    include_dataquality_summary: bool = False,
) -> Union[ForecastingResult, ErrorResult]:
    """Run zero-shot time-series forecasting with a TinyTimeMixer (TTM) model.

    Returns a ForecastingResult whose results_file is the path to a JSON file
    with raw predictions (target_prediction, timestamp, target_columns arrays).
    Pass that path to run_tsad as tsfm_output_json for anomaly detection.

    Args:
        dataset_path: Path to a CSV, JSON, or XLSX dataset.
        timestamp_column: Name of the timestamp column.
        target_columns: Columns to forecast.
        model_checkpoint: Model name (e.g. ttm_96_28) or absolute checkpoint path.
        forecast_horizon: Number of steps to forecast; -1 uses the model default.
        conditional_columns: Exogenous / conditional feature columns.
        id_columns: ID columns for multi-entity grouped time series.
        frequency_sampling: Sampling frequency string (e.g. '15_minutes') or
            'oov' to auto-detect from the data.
        autoregressive_modeling: Use autoregressive inference when True.
        include_dataquality_summary: Attach a data-quality report to the result.
    """
    if not dataset_path.strip():
        return ErrorResult(error="dataset_path is required")
    if not target_columns:
        return ErrorResult(error="target_columns must not be empty")

    try:
        import tsfm_public  # noqa: F401 – verify dependency present
    except ImportError as exc:
        return ErrorResult(error=f"tsfm dependencies unavailable: {exc}")

    model_checkpoint = _get_model_checkpoint_path(model_checkpoint)
    dataset_path = _get_dataset_path(dataset_path)
    dataset_config = _build_dataset_config(
        timestamp_column,
        target_columns,
        conditional_columns,
        id_columns,
        frequency_sampling,
        autoregressive_modeling,
    )

    try:
        data_df = _read_ts_data(dataset_path, dataset_config_dictionary=dataset_config)
        model_config = _load_model_config(model_checkpoint)

        output_data_quality = _tsfm_data_quality_filter(
            data_df, dataset_config, model_config, task="inference"
        )
        data_df = output_data_quality["data"]
        dataset_config = output_data_quality["dataset_config_dictionary"]

        inference_result_dict_data: dict = {
            "target_prediction": [],
            "timestamp": [],
            "target_columns": [],
        }

        if len(data_df) > 0:
            output = _get_ttm_hf_inference(
                data_df,
                dataset_config,
                model_config,
                model_checkpoint,
                forecast_horizon=forecast_horizon,
            )
            inference_result_dict_data["target_prediction"] = output[
                "target_prediction"
            ].tolist()
            inference_result_dict_data["timestamp"] = (
                np.array(output["timestamp_prediction"]).astype(str).tolist()
            )
            inference_result_dict_data["target_columns"] = output["target_columns"]
        else:
            return ErrorResult(
                error="Data quality was poor; after filtering, no continuous segment satisfied the "
                "context length requirement. Check Data Quality Summary."
            )

        # Trim to requested forecast horizon
        if forecast_horizon != -1 and "target_prediction" in inference_result_dict_data:
            target_prediction = np.array(
                inference_result_dict_data["target_prediction"]
            )
            if 0 < forecast_horizon <= target_prediction.shape[1]:
                inference_result_dict_data["target_prediction"] = target_prediction[
                    :, :forecast_horizon, :
                ].tolist()
                inference_result_dict_data["timestamp"] = np.array(
                    inference_result_dict_data["timestamp"]
                )[:, :forecast_horizon].tolist()

        results_file = _write_json_to_temp(
            json.dumps(inference_result_dict_data, indent=4)
        )

    except Exception as exc:
        logger.error("run_tsfm_forecasting failed: %s", exc)
        return ErrorResult(error=str(exc))

    dataquality_summary = (
        output_data_quality["dataquality_summary"]
        if include_dataquality_summary
        else None
    )
    return ForecastingResult(
        status="success",
        results_file=results_file,
        dataquality_summary=dataquality_summary,
        message=f"Forecasting complete. Predictions saved to {results_file}.",
    )


# ── TSFM Finetuning ───────────────────────────────────────────────────────────


@mcp.tool()
def run_tsfm_finetuning(
    dataset_path: str,
    timestamp_column: str,
    target_columns: List[str],
    model_checkpoint: str = "ttm_96_28",
    save_model_dir: str = "tuned_models",
    forecast_horizon: int = -1,
    n_finetune: float = 0.05,
    n_calibration: float = 0.0,
    n_test: float = 0.05,
    conditional_columns: Optional[List[str]] = None,
    id_columns: Optional[List[str]] = None,
    frequency_sampling: str = "oov",
    autoregressive_modeling: bool = True,
    include_dataquality_summary: bool = False,
) -> Union[FinetuningResult, ErrorResult]:
    """Few-shot fine-tune a TinyTimeMixer model on a local dataset.

    Returns a FinetuningResult with the saved checkpoint path and a JSON file
    containing per-forecast-horizon performance metrics.

    Args:
        dataset_path: Path to the training dataset (CSV/JSON/XLSX).
        timestamp_column: Name of the timestamp column.
        target_columns: Columns to forecast and fine-tune on.
        model_checkpoint: Base model to start from (e.g. ttm_96_28).
        save_model_dir: Directory to save the fine-tuned model checkpoint.
        forecast_horizon: Steps to forecast; -1 uses the model default.
        n_finetune: Fraction (≤1) or count (>1) of samples for fine-tuning.
        n_calibration: Fraction or count for calibration set (default 0).
        n_test: Fraction or count for test evaluation (default 0.05).
        conditional_columns: Exogenous feature columns.
        id_columns: ID columns for grouped time series.
        frequency_sampling: Sampling frequency string or 'oov' to auto-detect.
        autoregressive_modeling: Use autoregressive mode when True.
        include_dataquality_summary: Attach a data-quality report to the result.
    """
    if not dataset_path.strip():
        return ErrorResult(error="dataset_path is required")
    if not target_columns:
        return ErrorResult(error="target_columns must not be empty")

    try:
        import tsfm_public  # noqa: F401
    except ImportError as exc:
        return ErrorResult(error=f"tsfm dependencies unavailable: {exc}")

    model_checkpoint = _get_model_checkpoint_path(model_checkpoint)
    dataset_path = _get_dataset_path(dataset_path)
    abs_save_dir = _get_outputs_path(save_model_dir)
    dataset_config = _build_dataset_config(
        timestamp_column,
        target_columns,
        conditional_columns,
        id_columns,
        frequency_sampling,
        autoregressive_modeling,
    )

    try:
        data_df = _read_ts_data(dataset_path, dataset_config_dictionary=dataset_config)
        model_config = _load_model_config(model_checkpoint)

        os.makedirs(abs_save_dir, exist_ok=True)

        output_data_quality = _tsfm_data_quality_filter(
            data_df, dataset_config, model_config, task="finetuning"
        )
        data_df = output_data_quality["data"]
        dataset_config = output_data_quality["dataset_config_dictionary"]

        if len(data_df) == 0:
            return ErrorResult(
                error="Data quality was poor; after filtering, no continuous segment satisfied the "
                "context length requirement. Check Data Quality Summary."
            )

        output = _finetune_ttm_hf(
            data_df,
            dataset_config,
            model_config,
            abs_save_dir,
            n_finetune,
            n_calibration,
            n_test,
            model_checkpoint=model_checkpoint,
        )

        result_dict = output.copy()
        result_dict["performance"] = result_dict["performance"].to_dict()

        if "performance" in result_dict:
            df_perf = pd.DataFrame(result_dict["performance"])
            df_perf["forecast"] = df_perf["forecast"].values + 1
            max_forecast = df_perf["forecast"].max()
            if 0 < forecast_horizon <= max_forecast:
                result_dict["performance"] = df_perf.loc[
                    df_perf["forecast"] == forecast_horizon
                ].to_dict()

        if include_dataquality_summary:
            result_dict["dataquality_summary"] = output_data_quality[
                "dataquality_summary"
            ]

        results_file = _write_json_to_temp(json.dumps(result_dict, indent=4))

    except Exception as exc:
        logger.error("run_tsfm_finetuning failed: %s", exc)
        return ErrorResult(error=str(exc))

    try:
        fewshot_dir = abs_save_dir + "/fewshot/"
        saved_checkpoint = (
            _find_largest_tsfm_checkpoint_directory(fewshot_dir) or abs_save_dir
        ) + "/"
    except Exception as exc:
        logger.warning("Could not resolve finetuned checkpoint path: %s", exc)
        saved_checkpoint = save_model_dir

    return FinetuningResult(
        status="success",
        model_checkpoint=saved_checkpoint,
        results_file=results_file,
        message=(
            f"Fine-tuning complete. Model saved to {saved_checkpoint}. "
            f"Metrics saved to {results_file}."
        ),
    )


# ── TSAD (conformal anomaly detection on top of TSFM forecasts) ──────────────


@mcp.tool()
def run_tsad(
    dataset_path: str,
    tsfm_output_json: str,
    timestamp_column: str,
    target_columns: List[str],
    task: str = "fit",
    false_alarm: float = 0.05,
    ad_model_type: str = "timeseries_conformal_adaptive",
    ad_model_checkpoint: Optional[str] = None,
    ad_model_save: Optional[str] = None,
    n_calibration: float = 0.2,
    conditional_columns: Optional[List[str]] = None,
    id_columns: Optional[List[str]] = None,
    frequency_sampling: Optional[str] = None,
    autoregressive_modeling: bool = True,
) -> Union[TSADResult, ErrorResult]:
    """Run conformal anomaly detection on TSFM forecasting output.

    tsfm_output_json must be the results_file path returned by run_tsfm_forecasting.
    Fits (or loads) a conformal AD model and saves anomaly-labelled predictions to CSV.

    Args:
        dataset_path: Path to the original time-series dataset.
        tsfm_output_json: Path to JSON predictions file from run_tsfm_forecasting.
        timestamp_column: Name of the timestamp column.
        target_columns: Target columns that were forecast.
        task: 'fit' to train a new AD model, 'inference' to use an existing one.
        false_alarm: False alarm rate (1 − coverage); default 0.05 → 95% coverage.
        ad_model_type: 'timeseries_conformal' or 'timeseries_conformal_adaptive'.
        ad_model_checkpoint: Path to an existing AD model (required for 'inference').
        ad_model_save: Directory to save the fitted AD model.
        n_calibration: Fraction of data used for calibration (default 0.2).
        conditional_columns: Exogenous feature columns.
        id_columns: ID columns for grouped time series.
        frequency_sampling: Sampling frequency string or None to auto-detect.
        autoregressive_modeling: Use autoregressive mode when True.
    """
    if not dataset_path.strip():
        return ErrorResult(error="dataset_path is required")
    if not tsfm_output_json.strip():
        return ErrorResult(error="tsfm_output_json is required")
    if not target_columns:
        return ErrorResult(error="target_columns must not be empty")
    if task not in ("fit", "inference"):
        return ErrorResult(error="task must be 'fit' or 'inference'")

    try:
        import tsfm_public  # noqa: F401
    except ImportError as exc:
        return ErrorResult(error=f"tsfm dependencies unavailable: {exc}")

    dataset_config = _build_dataset_config(
        timestamp_column,
        target_columns,
        conditional_columns,
        id_columns,
        frequency_sampling or "",
        autoregressive_modeling,
    )

    try:
        with open(tsfm_output_json, "r") as fh:
            tsmodel_pred = json.load(fh)

        output = _TimeSeriesAnomalyDetectionConformalWrapper().run(
            dataset_path,
            dataset_config,
            tsmodel_pred,
            ad_model_checkpoint=ad_model_checkpoint,
            ad_model_save=ad_model_save,
            task=task,
            ad_model_type=ad_model_type,
            n_calibration=n_calibration,
            false_alarm=false_alarm,
        )
    except Exception as exc:
        logger.error("run_tsad failed: %s", exc)
        return ErrorResult(error=str(exc))

    try:
        df = _tsad_output_to_df(output)
        tmp_dir = tempfile.mkdtemp()
        csv_path = os.path.join(tmp_dir, f"tsad_output_{uuid.uuid4()}.csv")
        df.to_csv(csv_path, index=False)
        anomaly_count = (
            int(df["anomaly_label"].sum()) if "anomaly_label" in df.columns else 0
        )
    except Exception as exc:
        logger.error("run_tsad result serialisation failed: %s", exc)
        return ErrorResult(error=f"Failed to serialise TSAD output: {exc}")

    return TSADResult(
        status="success",
        results_file=csv_path,
        total_records=len(df),
        anomaly_count=anomaly_count,
        columns=list(df.columns),
        message=(
            f"Anomaly detection complete. {anomaly_count} anomalies in {len(df)} records. "
            f"Results saved to {csv_path}."
        ),
    )


# ── Integrated TSAD (forecasting + anomaly detection in one call) ─────────────


@mcp.tool()
def run_integrated_tsad(
    dataset_path: str,
    timestamp_column: str,
    target_columns: List[str],
    model_checkpoint: str = "ttm_96_28",
    false_alarm: float = 0.05,
    ad_model_type: str = "timeseries_conformal_adaptive",
    n_calibration: float = 0.2,
    conditional_columns: Optional[List[str]] = None,
    id_columns: Optional[List[str]] = None,
    frequency_sampling: str = "",
    autoregressive_modeling: bool = True,
) -> Union[TSADResult, ErrorResult]:
    """Run end-to-end time-series forecasting + anomaly detection in one call.

    For each target column: runs zero-shot TTM forecasting, then fits a conformal
    AD model and predicts anomaly labels. Saves a combined CSV with anomaly labels
    and KPI scores for all columns.

    Args:
        dataset_path: Path to the dataset (CSV/JSON/XLSX).
        timestamp_column: Name of the timestamp column.
        target_columns: Columns to run forecasting + anomaly detection on.
        model_checkpoint: Pre-trained TTM model name (default: ttm_96_28).
        false_alarm: False alarm rate; default 0.05 → 95% coverage.
        ad_model_type: 'timeseries_conformal' or 'timeseries_conformal_adaptive'.
        n_calibration: Fraction of data for AD calibration (default 0.2).
        conditional_columns: Exogenous feature columns.
        id_columns: ID columns for grouped time series.
        frequency_sampling: Sampling frequency string or '' to auto-detect.
        autoregressive_modeling: Use autoregressive mode when True.
    """
    if not dataset_path.strip():
        return ErrorResult(error="dataset_path is required")
    if not target_columns:
        return ErrorResult(error="target_columns must not be empty")

    try:
        import tsfm_public  # noqa: F401
    except ImportError as exc:
        return ErrorResult(error=f"tsfm dependencies unavailable: {exc}")

    model_checkpoint = _get_model_checkpoint_path(model_checkpoint)
    dataset_path = _get_dataset_path(dataset_path)

    try:
        ad_model_save = _get_outputs_path("tsad_model_save/")
        os.makedirs(ad_model_save, exist_ok=True)

        model_config = _load_model_config(model_checkpoint)
        df_combined = pd.DataFrame()

        # Read the full dataset once with all target columns, then subset per column
        full_config = _build_dataset_config(
            timestamp_column,
            target_columns,
            conditional_columns,
            id_columns,
            frequency_sampling,
            autoregressive_modeling,
        )
        full_data_df = _read_ts_data(dataset_path, dataset_config_dictionary=full_config)

        for col in target_columns:
            col_config = _build_dataset_config(
                timestamp_column,
                [col],
                conditional_columns,
                id_columns,
                frequency_sampling,
                autoregressive_modeling,
            )

            # 1. Quality-filter data for this column (reuse already-loaded data)
            data_df = full_data_df
            output_dq = _tsfm_data_quality_filter(
                data_df, col_config, model_config, task="inference"
            )
            data_df_filtered = output_dq["data"]
            col_config_filtered = output_dq["dataset_config_dictionary"]

            if len(data_df_filtered) == 0:
                logger.warning(
                    "Data quality filter removed all data for column %s; skipping.", col
                )
                continue

            # 2. Zero-shot forecasting for this column
            try:
                forecast_output = _get_ttm_hf_inference(
                    data_df_filtered,
                    col_config_filtered,
                    model_config,
                    model_checkpoint,
                )
            except Exception as exc:
                logger.warning("Forecasting failed for column %s: %s", col, exc)
                continue

            inference_data = {
                "target_prediction": forecast_output["target_prediction"].tolist(),
                "timestamp": np.array(forecast_output["timestamp_prediction"])
                .astype(str)
                .tolist(),
                "target_columns": forecast_output["target_columns"],
            }
            # 3. Conformal anomaly detection for this column
            tsmodel_pred = inference_data

            try:
                tsad_output = _TimeSeriesAnomalyDetectionConformalWrapper().run(
                    dataset_path,
                    col_config,
                    tsmodel_pred,
                    ad_model_checkpoint=None,
                    ad_model_save=ad_model_save,
                    task="fit",
                    ad_model_type=ad_model_type,
                    n_calibration=n_calibration,
                    false_alarm=false_alarm,
                )
            except Exception as exc:
                logger.warning("TSAD failed for column %s: %s", col, exc)
                continue

            df_col = _tsad_output_to_df(tsad_output)
            df_combined = pd.concat([df_combined, df_col], axis=0, ignore_index=True)

        if df_combined.empty:
            return ErrorResult(error="No TSAD results produced for any target column.")

        tmp_dir = tempfile.mkdtemp()
        csv_path = os.path.join(tmp_dir, f"integrated_tsad_{uuid.uuid4()}.csv")
        df_combined.to_csv(csv_path, index=False)
        anomaly_count = (
            int(df_combined["anomaly_label"].sum())
            if "anomaly_label" in df_combined.columns
            else 0
        )

    except Exception as exc:
        logger.error("run_integrated_tsad failed: %s", exc)
        return ErrorResult(error=str(exc))

    return TSADResult(
        status="success",
        results_file=csv_path,
        total_records=len(df_combined),
        anomaly_count=anomaly_count,
        columns=list(df_combined.columns),
        message=(
            f"Integrated TSAD complete. {anomaly_count} anomalies in {len(df_combined)} records "
            f"across {len(target_columns)} column(s). Results saved to {csv_path}."
        ),
    )


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
