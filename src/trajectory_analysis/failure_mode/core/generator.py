"""Failure mode generation from trajectory data.

This module processes agent trajectories and uses LLM analysis to identify
failure modes in agent behavior.
"""

import os
import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Sequence, Optional

from llm.base import LLMBackend
from llm.litellm import LiteLLMBackend
from .utils import get_llm_answer_from_json, extract_json_from_response

# Configure logger
logger = logging.getLogger(__name__)


def _load_all_json_files(root_path: str) -> Dict[str, Any]:
    """Load numeric-named files (e.g., '0001', '0002') recursively under root_path.

    Args:
        root_path: Root directory to search for JSON files

    Returns:
        Dictionary mapping file paths to their JSON content
    """
    json_data: Dict[str, Any] = {}
    total_files = 0
    valid_files = 0

    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_files += 1

            try:
                logger.info(f"📄 Loading: {file_path}")
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    json_data[file_path] = data
                    valid_files += 1
                    logger.info(f"   ✅ Valid JSON loaded")
            except json.JSONDecodeError as e:
                logger.warning(f"   ❌ Invalid JSON: {e}")
            except Exception as e:
                logger.warning(f"   ❌ Error loading file: {e}")

    logger.info(f"\n📊 Summary: {valid_files}/{total_files} files loaded successfully")
    return json_data


def _normalize_additional_failure_modes(obj: Any) -> List[Dict[str, Any]]:
    """Normalize additional failure modes to a consistent list format.

    Args:
        obj: Additional failure modes in various formats (list, dict, or None)

    Returns:
        List of dictionaries with 'title' and 'description' keys
    """
    if obj is None:
        return []
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        if "title" in obj or "description" in obj:
            return [obj]
        return [{"title": t, "description": d} for t, d in obj.items()]
    return []


def process_trajectories(
    timestamps: Optional[Sequence[str]] = None,
    traj_root_base: str = ".",
    llm_backend: Optional[LLMBackend] = None,
    temperature: float = 0.0,
    out_dir: str = "./src/trajectory_analysis/failure_mode/results/runs",
    run_id: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process trajectories using LLM and save results in timestamped run folders.

    Creates a timestamped run folder (e.g., 20260414_140523/) containing only:
    - failure_modes.pkl: Pickle file with all results
    - failure_modes.csv: CSV file with all results

    If `timestamps` is None, auto-discovers subfolders in `traj_root_base` and uses them as timestamps.
    If `llm_backend` is None, defaults to Claude 4 Sonnet via LiteLLM proxy.

    Args:
        timestamps: Optional list of timestamp strings to process
        traj_root_base: Root directory containing trajectory files
        llm_backend: LLM backend to use for analysis (defaults to Claude 4 Sonnet)
        temperature: Temperature parameter for LLM generation (default: 0.0)
        out_dir: Base output directory for run folders (default: ./results/runs)
        run_id: Optional run identifier (default: auto-generated timestamp YYYYMMDD_HHMMSS)

    Returns:
        Dictionary containing:
            - run_dir: Path to the run directory
            - run_id: Run identifier (timestamp)
            - failure_modes_path: Path to failure modes pickle file
            - failure_modes_csv_path: Path to failure modes CSV file
            - df: Pandas DataFrame with results
    """
    # Default to Claude Sonnet 4.6 if no backend provided
    # Note: GCP Claude models have configuration issues, AWS Claude works perfectly
    # See litellm_models_report.md for full test results
    if llm_backend is None:
        llm_backend = LiteLLMBackend("litellm_proxy/claude-sonnet-4-6")

    # Generate run ID if not provided (just timestamp, no "run_" prefix)
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create run directory
    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Run directory: {run_dir}")

    failure_mode_keys = [
        "1.1 Disobey Task Specification",
        "1.2 Disobey Role Specification",
        "1.3 Step Repetition",
        "1.4 Loss of Conversation History",
        "1.5 Unaware of Termination Conditions",
        "2.1 Conversation Reset",
        "2.2 Fail to Ask for Clarification",
        "2.3 Task Derailment",
        "2.4 Information Withholding",
        "2.5 Ignored Other Agent's Input",
        "2.6 Action-Reasoning Mismatch",
        "3.1 Premature Termination",
        "3.2 No or Incorrect Verification",
        "3.3 Weak Verification",
    ]

    print(f"\n🔍 Processing trajectories from: {traj_root_base}")
    root_directory = f"{traj_root_base}"
    all_jsons = _load_all_json_files(root_directory)
    print(f"  📊 Loaded {len(all_jsons)} trajectory files")

    # Use provided model_id or get from llm_backend
    if model_id is None:
        # Try to get model_id from backend (handles both public and private attributes)
        model_id = getattr(llm_backend, "model_id", None)
        if model_id is None:
            model_id = getattr(llm_backend, "_model_id", "unknown")

    df_columns = [
        "model_id",
        "trajectory_path",
        "format_handler",
        "counter",
        "ut_id",
        "addi_fm_cnt",
        "addi_fm_list",
    ] + failure_mode_keys
    df = pd.DataFrame(columns=df_columns)

    counter = 1

    for path, content in all_jsons.items():
        # Store the full path as trajectory_path
        rel_path = path

        # Extract ut_id from the filename (handle both with and without underscores)
        filename = os.path.basename(path)
        parts = filename.split("_")
        ut_id = parts[0]

        logger.info(f"\n🔄 Processing trajectory {counter}/{len(all_jsons)}: {ut_id}")
        logger.info(f"   📂 Path: {path}")

        max_trial = 2
        cur_trial = 0
        while cur_trial < max_trial:
            cur_trial = cur_trial + 1
            try:
                logger.info(
                    f"   🤖 Analyzing with LLM (attempt {cur_trial}/{max_trial})..."
                )
                raw_output, handler_name = get_llm_answer_from_json(
                    data=content, llm_backend=llm_backend, temperature=temperature
                )
                response_json = extract_json_from_response(raw_output)

                failure_modes = response_json.get("failure_modes", {})
                afm_list = _normalize_additional_failure_modes(
                    response_json.get("additional_failure_modes", [])
                )

                # Count detected failure modes
                fm_count = sum(1 for v in failure_modes.values() if v)
                logger.info(
                    f"   ✅ Analysis complete: {fm_count} failure modes detected, {len(afm_list)} additional"
                )

                row = {
                    "model_id": model_id,
                    "trajectory_path": rel_path,
                    "format_handler": handler_name,
                    "counter": counter,
                    "ut_id": ut_id,
                    "addi_fm_cnt": len(afm_list),
                    "addi_fm_list": afm_list,
                }
                for key in failure_mode_keys:
                    row[key] = bool(failure_modes.get(key, False))

                df.loc[len(df)] = row
                break
            except Exception as e:
                logger.error(f"   ❌ Failed to process {path}: {e}")
                if cur_trial >= max_trial:
                    logger.warning(f"   ⚠️  Skipping after {max_trial} attempts")

        counter += 1

    # Save results in run directory
    failure_modes_pkl = run_dir / "failure_modes.pkl"
    failure_modes_csv = run_dir / "failure_modes.csv"
    df.to_pickle(failure_modes_pkl)
    df.to_csv(failure_modes_csv, index=False)

    print(f"\n✅ Saved results:")
    print(f"  📦 Pickle: {failure_modes_pkl} ({len(df)} rows)")
    print(f"  📄 CSV: {failure_modes_csv} ({len(df)} rows)")

    return {
        "run_dir": str(run_dir),
        "run_id": run_id,
        "failure_modes_path": str(failure_modes_pkl),
        "failure_modes_csv_path": str(failure_modes_csv),
        "df": df,
    }


def combine_all_runs(
    runs_dir: str = "./src/trajectory_analysis/failure_mode/results/runs",
    summary_dir: str = "./src/trajectory_analysis/failure_mode/results/summary",
) -> Dict[str, Any]:
    """
    Combine all individual run results into a single combined file.

    Reads all failure_modes.pkl files from runs/ subdirectories and combines them
    into a single DataFrame saved in summary/combined_failure_modes.{pkl,csv}.

    Args:
        runs_dir: Directory containing individual run folders
        summary_dir: Directory to save combined results

    Returns:
        Dictionary containing:
            - combined_path: Path to combined pickle file
            - combined_csv_path: Path to combined CSV file
            - combined_df: Combined pandas DataFrame
            - n_runs: Number of runs combined
    """
    runs_path = Path(runs_dir)
    summary_path = Path(summary_dir)
    summary_path.mkdir(parents=True, exist_ok=True)

    # Find all run directories
    run_dirs = [d for d in runs_path.iterdir() if d.is_dir()]

    if not run_dirs:
        print(f"⚠️  No run directories found in {runs_dir}")
        return {
            "combined_path": None,
            "combined_csv_path": None,
            "combined_df": pd.DataFrame(),
            "n_runs": 0,
        }

    print(f"\n🔄 Combining {len(run_dirs)} runs...")

    # Load all run dataframes and add run_id column
    all_dfs = []
    for run_dir in sorted(run_dirs):
        pkl_file = run_dir / "failure_modes.pkl"
        if pkl_file.exists():
            try:
                df = pd.read_pickle(pkl_file)
                # Add run_id column from folder name
                df.insert(0, "run_id", run_dir.name)
                all_dfs.append(df)
                print(f"  ✅ Loaded: {run_dir.name} ({len(df)} rows)")
            except Exception as e:
                print(f"  ❌ Error loading {run_dir.name}: {e}")

    if not all_dfs:
        print(f"⚠️  No valid pickle files found")
        return {
            "combined_path": None,
            "combined_csv_path": None,
            "combined_df": pd.DataFrame(),
            "n_runs": 0,
        }

    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Save combined results
    combined_pkl = summary_path / "combined_failure_modes.pkl"
    combined_csv = summary_path / "combined_failure_modes.csv"
    combined_df.to_pickle(combined_pkl)
    combined_df.to_csv(combined_csv, index=False)

    print(f"\n✅ Combined results saved:")
    print(f"  📦 Pickle: {combined_pkl} ({len(combined_df)} total rows)")
    print(f"  📄 CSV: {combined_csv}")

    return {
        "combined_path": str(combined_pkl),
        "combined_csv_path": str(combined_csv),
        "combined_df": combined_df,
        "n_runs": len(all_dfs),
    }


