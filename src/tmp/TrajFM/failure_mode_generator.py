import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Sequence, Optional

from utils import get_llm_answer_from_json, extract_json_from_response


def _load_all_json_files(root_path: str) -> Dict[str, Any]:
    """Load numeric-named files (e.g., '0001', '0002') recursively under root_path."""
    json_data: Dict[str, Any] = {}
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    json_data[file_path] = data
            except Exception:
                pass
    return json_data


def _normalize_additional_failure_modes(obj: Any) -> List[Dict[str, Any]]:
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
    model_id: int = 18,
    out_dir: str = "processed_trajectories",
):
    """
    Process trajectories using LLM and save per-timestamp + combined pickles.

    If `timestamps` is None, auto-discovers subfolders in `traj_root_base` and uses them as timestamps.
    """

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

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    per_timestamp_paths: List[str] = []
    all_dfs: List[pd.DataFrame] = []

    timestamp = '1'
    print(f"\nProcessing timestamp {timestamp}")
    root_directory = f"{traj_root_base}"
    all_jsons = _load_all_json_files(root_directory)
    print(f"  Loaded {len(all_jsons)} files")

    df_columns = [
        "model_id",
        "counter",
        "timestamp",
        "vendor",
        "model",
        "ut_id",
        "addi_fm_cnt",
        "addi_fm_list",
    ] + failure_mode_keys
    df = pd.DataFrame(columns=df_columns)

    counter = 1
    for path, content in all_jsons.items():
        parts = os.path.relpath(path, root_directory).split('_')
        ut_id = parts[0]
        model = model_id
        vendor = ''

        max_trial = 2
        cur_trial = 0
        while cur_trial < max_trial:
            cur_trial = cur_trial + 1
            try:
                raw_output = get_llm_answer_from_json(data=content, model_id=model_id)
                response_text = raw_output["generated_text"]
                response_json = extract_json_from_response(response_text)

                failure_modes = response_json.get("failure_modes", {})
                afm_list = _normalize_additional_failure_modes(
                    response_json.get("additional_failure_modes", [])
                )

                row = {
                    "model_id": model_id,
                    "counter": counter,
                    "timestamp": timestamp,
                    "vendor": vendor,
                    "model": model,
                    "ut_id": ut_id,
                    "addi_fm_cnt": len(afm_list),
                    "addi_fm_list": afm_list,
                }
                for key in failure_mode_keys:
                    row[key] = bool(failure_modes.get(key, False))

                df.loc[len(df)] = row
                break
            except Exception as e:
                print(f"  Failed to process {path}: {e}")

        counter += 1

        df_file_path = f"{out_dir}/{timestamp}_m{model_id}_db.pkl"
        df.to_pickle(df_file_path)
        per_timestamp_paths.append(df_file_path)
        all_dfs.append(df)
        print(f"  Saved {df_file_path} with {len(df)} rows")

    combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    combined_file_path = f"{out_dir}/combined_m{model_id}_db.pkl"
    combined_df.to_pickle(combined_file_path)
    print(f"\nSaved combined DataFrame: {combined_file_path} ({len(combined_df)} rows)")

    return {
        "per_timestamp_paths": per_timestamp_paths,
        "combined_path": combined_file_path,
        "combined_df": combined_df,
    }
