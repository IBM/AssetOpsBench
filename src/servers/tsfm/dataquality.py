"""Time-series data quality helpers: NaN removal, segmentation, quality summary."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Condition helpers ─────────────────────────────────────────────────────────


def _threshold_condition_function(threshold, condition_type="<"):
    conditions = {
        "<": lambda x: x < threshold,
        "<=": lambda x: x <= threshold,
        ">": lambda x: x > threshold,
        ">=": lambda x: x >= threshold,
        "==": lambda x: x == threshold,
    }
    assert condition_type in conditions, (
        f"condition_type {condition_type!r} is not supported"
    )
    return conditions[condition_type]


# ── NaN statistics ────────────────────────────────────────────────────────────


def _df_nan_stats(df, perc_rows_less_than=None, perc_rows_more_than=None):
    if perc_rows_less_than is None:
        perc_rows_less_than = [10, 20, 50]
    if perc_rows_more_than is None:
        perc_rows_more_than = [50, 100]
    output: dict = {}
    nan_per_column = df.isna().mean() * 100
    output["%NaN_per_column"] = nan_per_column.to_dict()
    output["%rows_0_NaNs"] = (df.isna().sum(axis=1) == 0).mean() * 100
    if perc_rows_less_than and output["%rows_0_NaNs"] > 0:
        output["%rows_less_than"] = {
            f"{p}% NaNs": np.mean(df.isna().mean(axis=1) <= float(p / 100)) * 100
            for p in perc_rows_less_than
        }
    if perc_rows_more_than and output["%rows_0_NaNs"] > 0:
        output["%rows_more_than"] = {
            f"{p}% NaNs": np.mean(df.isna().mean(axis=1) > float(p / 100)) * 100
            for p in perc_rows_more_than
        }
    return output


def _df_percentage_samples_minutes_interval(
    df, date_col, lower_bound=14, upper_bound=16
):
    assert upper_bound >= lower_bound, "lower bound is larger than upper bound"
    df = df.sort_values(by=date_col)
    time_diffs = df[date_col].diff().dt.total_seconds() / 60.0
    interval_count = ((time_diffs >= lower_bound) & (time_diffs <= upper_bound)).sum()
    total_intervals = len(time_diffs) - 1
    return (interval_count / total_intervals) * 100 if total_intervals > 0 else 0


def _df_dt_stats(pd_dataset, date_col="Timestamp", intervals_dic=None):
    if intervals_dic is None:
        intervals_dic = {"14min_to_16min": (14, 16)}
    pd_dataset = pd_dataset.sort_values(by=date_col)
    earliest_date = pd_dataset[date_col].min()
    latest_date = pd_dataset[date_col].max()
    date_interval = latest_date - earliest_date
    time_intervals = pd_dataset[date_col].diff()
    time_intervals_dic = time_intervals.value_counts().to_dict()
    time_intervals_dic_json = {str(k): int(v) for k, v in time_intervals_dic.items()}
    perc_in_interval_dic = None
    if isinstance(intervals_dic, dict):
        perc_in_interval_dic = {
            key: _df_percentage_samples_minutes_interval(
                pd_dataset, date_col, lower_bound=bounds[0], upper_bound=bounds[1]
            )
            for key, bounds in intervals_dic.items()
        }
    data_specs: dict = {
        "initial_time": earliest_date.isoformat(),
        "final_time": latest_date.isoformat(),
        "interval": str(date_interval),
        "columns": pd_dataset.columns.values.tolist(),
        "number_samples": len(pd_dataset),
        "time_interval_between_samples": time_intervals_dic_json,
    }
    if perc_in_interval_dic is not None:
        data_specs["percentage_in_dt"] = perc_in_interval_dic
    return data_specs


def _df_single_columns_condition(df, condition_dic=None):
    if condition_dic is None:
        condition_dic = {}
    condition_count = {}
    for key, (column_name, condition) in condition_dic.items():
        if column_name in df.columns:
            mask = df[column_name].apply(condition)
            condition_count[key] = {
                "nsamples": int(np.sum(mask)),
                "percentile": 100 * np.sum(mask) / max(len(mask), 1),
            }
    return condition_count


# ── NaN removal ───────────────────────────────────────────────────────────────


def _efficient_nan_removal(pd_table, preference_tie="row"):
    def compute_removal_costs(df):
        row_non_nan = df.notna().sum(axis=1)
        col_non_nan = df.notna().sum(axis=0)
        row_costs = np.where(df.isna().sum(axis=1) == 0, np.inf, row_non_nan)
        col_costs = np.where(df.isna().sum(axis=0) == 0, np.inf, col_non_nan)
        return row_costs, col_costs

    def remove_lowest_cost(df, row_costs, col_costs, preference_tie="row"):
        min_row = np.min(row_costs)
        min_col = np.min(col_costs)
        if (min_row < min_col) or (min_row == min_col and preference_tie == "row"):
            row_to_remove = np.where(row_costs == min_row)[0]
            df = df.drop(df.index[row_to_remove])
            removed = f"Row {row_to_remove} removed with cost {min_row}"
        else:
            col_to_remove = np.where(col_costs == min_col)[0]
            df = df.drop(df.columns[col_to_remove], axis=1)
            removed = f"Column {col_to_remove} removed with cost {min_col}"
        return df, removed

    df_t = pd_table.copy()
    max_actions = len(df_t) + len(df_t.columns)
    t = 0
    actions = []
    while df_t.isna().any().any() and t < max_actions:
        row_costs, col_costs = compute_removal_costs(df_t)
        df_t, removed = remove_lowest_cost(df_t, row_costs, col_costs, preference_tie)
        actions.append(removed)
        t += 1
    cost_total = pd_table.notna().sum().sum() - df_t.notna().sum().sum()
    return {"df_filter": df_t, "actions": actions, "cost_total": cost_total}


def _remove_df_nans(df, p=50, dim="columns"):
    threshold = p / 100.0
    assert dim in ("columns", "rows")
    if dim == "columns":
        cols_to_drop = df.columns[df.isna().mean() > threshold]
        return df.drop(columns=cols_to_drop)
    rows_to_drop = df.index[df.isna().mean(axis=1) > threshold]
    return df.drop(index=rows_to_drop)


def _remove_df_rows_by_single_column_condition(df, column_name, condition):
    if column_name in df.columns:
        mask = df[column_name].apply(condition)
        return df[(1 - mask) == 1]
    return df


# ── Segmentation ──────────────────────────────────────────────────────────────


def _time_series_frequency_interval_segmentation(
    df, time_column, lower_bound=14, upper_bound=16
):
    df = df.sort_values(by=time_column).reset_index(drop=True)
    df["dt"] = df[time_column].diff().dt.total_seconds() / 60.0
    df["segment_id"] = 0
    segment_id = 0
    start_idx = 0
    for i in range(1, len(df)):
        if not (lower_bound <= df["dt"].iloc[i] <= upper_bound):
            df.loc[start_idx:i, "segment_id"] = segment_id
            segment_id += 1
            start_idx = i
    df.loc[start_idx:, "segment_id"] = segment_id
    return df.drop(columns="dt")


def _validate_time_series_segments(
    df_segment,
    segment_tag="segment_id",
    timestamp_tag="Timestamp",
    p_nan_rows=1,
    p_nan_columns=70,
    condition_off_dic=None,
    dt_bounds=None,
):
    if dt_bounds is None:
        dt_bounds = [14, 16]
    bad_quality_segments: dict = {}
    lower_bound, upper_bound = dt_bounds[0], dt_bounds[1]
    for seg_id in df_segment[segment_tag].unique():
        df_seg_i = df_segment.loc[df_segment[segment_tag] == seg_id]
        dic_nan = _df_nan_stats(
            df_seg_i, perc_rows_less_than=[p_nan_rows], perc_rows_more_than=[p_nan_rows]
        )
        if condition_off_dic is not None:
            df_cond = _df_single_columns_condition(
                df_seg_i, condition_dic=condition_off_dic
            )
        nan_cols = list(dic_nan["%NaN_per_column"].values())
        qc: dict = {
            "nan_per_column": np.max(np.array(nan_cols)) <= p_nan_columns,
            "nan_per_rows": list(dic_nan["%rows_more_than"].items())[0][1]
            <= p_nan_rows,
        }
        if condition_off_dic is not None:
            cond_vals = [df_cond[k]["nsamples"] for k in df_cond]
            qc["condition_off"] = np.max(np.array(cond_vals)) == 0
        perc = _df_percentage_samples_minutes_interval(
            df_seg_i,
            date_col=timestamp_tag,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        qc["sampling_dt_condition"] = perc == 100
        if not all(qc.values()):
            bad_quality_segments[seg_id] = qc
    return bad_quality_segments


def _time_series_segment_quality_summary(df, timestamp_column, segments_column):
    ts_cont_segments: dict = {}
    for ix_s in df[segments_column].unique():
        df_filter = df.loc[df[segments_column] == ix_s]
        ts_cont_segments[ix_s] = {
            "start": pd.to_datetime(df_filter[timestamp_column].values[0]).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "end": pd.to_datetime(df_filter[timestamp_column].values[-1]).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "samples": len(df_filter),
            "%nans": df_filter.isna().mean().mean() * 100,
        }
    return ts_cont_segments


# ── Orchestration ─────────────────────────────────────────────────────────────

_FILTERING_PARAMS_DEFAULT = {
    "nans": {"efficient_removal": {"preference_tie": "row"}},
    "dt": {"lower_bound": 14, "upper_bound": 16},
}


def _dq_timeseries_segmentation(
    pd_merge, filtering_params=None, timestamp_tag="Timestamp"
):
    if filtering_params is None:
        filtering_params = _FILTERING_PARAMS_DEFAULT
    df_cleaned = pd_merge.copy()
    p_nan_columns = 70
    p_nan_rows = 1

    if "nans" in filtering_params:
        if "efficient_removal" in filtering_params["nans"]:
            preference_tie = filtering_params["nans"]["efficient_removal"].get(
                "preference_tie", "row"
            )
            output_efficient = _efficient_nan_removal(
                df_cleaned, preference_tie=preference_tie
            )
            df_cleaned = output_efficient["df_filter"]
            p_nan_columns = 1
            p_nan_rows = 1
        if "p_nan_columns" in filtering_params["nans"]:
            p_nan_columns = filtering_params["nans"]["p_nan_columns"]
            df_cleaned = _remove_df_nans(df_cleaned, p=p_nan_columns, dim="columns")
        if "p_nan_rows" in filtering_params["nans"]:
            p_nan_rows = filtering_params["nans"]["p_nan_rows"]
            df_cleaned = _remove_df_nans(df_cleaned, p=p_nan_rows, dim="rows")

    df_cleaned[timestamp_tag] = pd.to_datetime(
        df_cleaned[timestamp_tag], errors="coerce"
    )
    df_cleaned = df_cleaned.dropna(subset=[timestamp_tag])

    condition_off_dic = None
    if "operation_condition" in filtering_params:
        operation_condition = filtering_params["operation_condition"]
        condition_off_dic = {}
        for op in operation_condition:
            col = operation_condition[op]["column"]
            if col in pd_merge.columns:
                condition_off_dic[op] = (
                    col,
                    _threshold_condition_function(
                        operation_condition[op]["threshold"],
                        condition_type=operation_condition[op]["condition_type"],
                    ),
                )
            else:
                logger.debug("Column %s not present in the cleaned dataset", col)
        if condition_off_dic:
            for key in condition_off_dic:
                df_cleaned = _remove_df_rows_by_single_column_condition(
                    df_cleaned, condition_off_dic[key][0], condition_off_dic[key][1]
                )

    lower_bound = filtering_params["dt"]["lower_bound"]
    upper_bound = filtering_params["dt"]["upper_bound"]
    df_segment = _time_series_frequency_interval_segmentation(
        df_cleaned, timestamp_tag, lower_bound=lower_bound, upper_bound=upper_bound
    )

    bad_quality_segments = _validate_time_series_segments(
        df_segment,
        segment_tag="segment_id",
        timestamp_tag=timestamp_tag,
        p_nan_rows=p_nan_rows,
        p_nan_columns=p_nan_columns,
        condition_off_dic=condition_off_dic,
        dt_bounds=[lower_bound, upper_bound],
    )
    if bad_quality_segments:
        for seg_id in bad_quality_segments:
            df_segment = df_segment[df_segment["segment_id"] != seg_id]
    return df_segment
