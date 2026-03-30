"""Conformal anomaly detection for time-series forecasting outputs."""

from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pandas as pd

from .io import _read_ts_data


# ── Conformal anomaly detection ───────────────────────────────────────────────

_NONCONFORMITY_SCORES = ["absolute_error"]


def _absolute_error(y, y_pred):
    assert y.shape == y_pred.shape, (
        f"y and y_pred shapes do not match: {y.shape} vs {y_pred.shape}"
    )
    error = np.abs(y - y_pred)
    if len(error.shape) > 1:
        error = np.mean(error, axis=-1)
    return error


def _nonconformity_score_functions(
    y_pred, y_gt, X=None, nonconformity_score="absolute_error"
):
    assert nonconformity_score in _NONCONFORMITY_SCORES
    if nonconformity_score == "absolute_error":
        return _absolute_error(y_gt, y_pred)


def _conformal_set(y_pred, score_threshold, nonconformity_score="absolute_error"):
    if nonconformity_score == "absolute_error":
        return {"y_low": y_pred - score_threshold, "y_high": y_pred + score_threshold}


def _weighted_conformal_quantile(
    scores, weights, alpha=0.05, conformal_correction=False, max_score=np.inf
):
    if weights is None:
        weights = np.ones_like(scores)
    assert np.max(weights) <= 1
    assert np.min(weights) >= 0
    assert weights.shape[0] == scores.shape[0]
    if conformal_correction:
        weights = np.append(weights, np.array([1]))
        scores = np.append(scores, np.array([max_score]))
    weights = np.array(weights) / np.sum(weights)
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumulative_weights = np.cumsum(sorted_weights)
    quantile_index = np.searchsorted(cumulative_weights, 1 - alpha)
    return sorted_scores[quantile_index]


def _weighted_conformal_alpha(
    scores, weights, score_observed, conformal_correction=False, max_score=np.inf
):
    if weights is None:
        weights = np.ones_like(score_observed)
    if conformal_correction:
        weights = np.append(weights, np.array([1]))
        scores = np.append(scores, np.array([max_score]))
    weights = np.array(weights) / np.sum(weights)
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_weights = weights[sorted_indices]
    return np.sum(sorted_weights[sorted_scores > score_observed])


class _TSADWeightedConformalWrapper:
    def __init__(
        self,
        nonconformity_score="absolute_error",
        false_alarm=0.05,
        weighting="uniform",
        weighting_params=None,
        threshold_function="weighting",
        window_size=None,
        online_adaptive=False,
    ):
        if weighting_params is None:
            weighting_params = {}
        self.nonconformity_score = nonconformity_score
        assert self.nonconformity_score in _NONCONFORMITY_SCORES
        self.nonconformity_score_func = _nonconformity_score_functions
        self.quantile = 1 - false_alarm
        self.false_alarm = false_alarm
        self.weighting = weighting
        self.weighting_params = weighting_params
        self.window_size = window_size
        self.online_size = 1
        self.online = online_adaptive
        self.threshold_function = threshold_function
        self.cal_scores: list = []
        self.weights: list = []
        self.cal_X: list = []
        self.cal_timestamps: list = []
        self.score_threshold = None

    def fit(self, y_cal_pred, y_cal_gt, X_cal=None, cal_timestamps=None):
        if self.window_size is None:
            self.window_size = y_cal_pred.shape[0]
        self.cal_scores = _nonconformity_score_functions(
            y_cal_pred, y_cal_gt, X=X_cal, nonconformity_score=self.nonconformity_score
        )
        self.cal_scores = self.cal_scores[-self.window_size :]
        if X_cal is not None:
            self.cal_X = X_cal[-self.window_size :]
        if cal_timestamps is not None:
            self.cal_timestamps = cal_timestamps[-self.window_size :]
        if self.weighting in ("uniform", "exponential_decay"):
            cal_weights = self.get_weights()
            self.weights.append(cal_weights)
            if self.threshold_function == "weighting":
                self.score_threshold = self._score_threshold_func(
                    cal_weights, false_alarm=self.false_alarm
                )
        critical_efficient_size = np.ceil(1 / self.false_alarm)
        assert np.sum(cal_weights) >= critical_efficient_size, (
            f" The effective size is too small for the desired false alarm of {self.false_alarm}, "
            f"the calibration set should be larger than {critical_efficient_size}"
        )

    def get_weights(self, y_pred=None, X=None, timestamps=None, false_alarm=None):
        if false_alarm is None:
            false_alarm = self.false_alarm
        if self.weighting in ("uniform", "exponential_decay"):
            if self.weights:
                return self.weights[-1]
            if self.weighting == "uniform":
                return np.ones(self.window_size)
            if self.weighting == "exponential_decay":
                decay_param = self.weighting_params.get("decay_param", 0.99)
                return decay_param ** (self.window_size - np.arange(self.window_size))

    def _score_threshold_func(
        self,
        cal_weights,
        cal_scores=None,
        y_pred=None,
        X=None,
        timestamps=None,
        false_alarm=None,
    ):
        if cal_scores is None:
            cal_scores = self.cal_scores
        if false_alarm is None:
            false_alarm = self.false_alarm
        score_threshold = []
        if self.threshold_function == "weighting":
            if len(cal_weights.shape) == 1:
                score_threshold = _weighted_conformal_quantile(
                    np.append(cal_scores, np.array([np.infty]), axis=0),
                    np.append(cal_weights, np.array([1]), axis=0),
                    alpha=false_alarm,
                )
            else:
                for i in range(cal_weights.shape[0]):
                    st_i = _weighted_conformal_quantile(
                        np.append(cal_scores, np.array([np.infty]), axis=0),
                        np.append(cal_weights[i, :], np.array([1]), axis=0),
                        alpha=false_alarm,
                    )
                    score_threshold.append(st_i)
                score_threshold = np.array(score_threshold)
        return score_threshold

    def predict_batch(
        self, y_pred, y_gt=None, X=None, timestamps=None, false_alarm=None, update=None
    ):
        if false_alarm is None:
            false_alarm = self.false_alarm
        if update is None:
            update = self.online
        cal_weights = self.get_weights(
            y_pred, X=X, timestamps=timestamps, false_alarm=false_alarm
        )
        if (
            false_alarm == self.false_alarm
            and self.weighting in ("uniform",)
            and self.threshold_function in ("weighting",)
        ):
            score_threshold = self.score_threshold
        else:
            score_threshold = self._score_threshold_func(
                cal_weights,
                y_pred=y_pred,
                X=X,
                timestamps=timestamps,
                false_alarm=false_alarm,
            )
        prediction_interval = _conformal_set(
            y_pred, score_threshold, nonconformity_score=self.nonconformity_score
        )
        output: dict = {}
        if y_gt is not None:
            test_scores = _nonconformity_score_functions(
                y_pred, y_gt, X=X, nonconformity_score=self.nonconformity_score
            )
            test_outliers = np.array(test_scores > score_threshold).astype("int")
            test_ad_scores = [
                _weighted_conformal_alpha(
                    np.append(self.cal_scores, np.array([np.infty]), axis=0),
                    np.append(cal_weights, np.array([1]), axis=0),
                    score,
                )
                for score in test_scores
            ]
            if update:
                self.update(test_scores, X=X, timestamps=timestamps)
            output["outliers"] = test_outliers
            output["outliers_scores"] = np.array(test_ad_scores).flatten()
        output["prediction_interval"] = prediction_interval
        return output

    def predict(
        self, y_pred, y_gt=None, X=None, timestamps=None, false_alarm=None, update=None
    ):
        if false_alarm is None:
            false_alarm = self.false_alarm
        if update is None:
            update = self.online
        n_samples = y_pred.shape[0]
        n_batches = int(np.ceil(n_samples / self.online_size))
        if y_gt is not None and update:
            output = None
            for ix_b in range(n_batches):
                ix_ini = int(ix_b * self.online_size)
                ix_end = min(
                    int(ix_b * self.online_size + self.online_size), y_pred.shape[0]
                )
                y_pred_b = y_pred[ix_ini:ix_end]
                y_gt_b = y_gt[ix_ini:ix_end]
                X_b = X[ix_ini:ix_end] if X is not None else None
                ts_b = timestamps[ix_ini:ix_end] if timestamps is not None else None
                output_b = self.predict_batch(
                    y_pred_b,
                    y_gt=y_gt_b,
                    X=X_b,
                    timestamps=ts_b,
                    false_alarm=false_alarm,
                    update=update,
                )
                if output is None:
                    output = output_b.copy()
                else:
                    for k in output_b:
                        if k == "prediction_interval":
                            for k2 in output_b[k]:
                                output[k][k2] = np.append(
                                    output[k][k2], np.array(output_b[k][k2]), axis=0
                                )
                        else:
                            output[k] = np.append(
                                output[k], np.array(output_b[k]), axis=0
                            )
        else:
            output = self.predict_batch(
                y_pred,
                y_gt=y_gt,
                X=X,
                timestamps=timestamps,
                false_alarm=false_alarm,
                update=update,
            )
        return output

    def update(self, scores, X=None, timestamps=None):
        self.cal_scores = np.append(self.cal_scores, scores, axis=0)
        self.cal_scores = self.cal_scores[-self.window_size :]
        if timestamps is not None:
            self.cal_timestamps.extend(timestamps)
            self.cal_timestamps = self.cal_timestamps[-self.window_size :]
        if X is not None:
            self.cal_X = np.append(self.cal_X, X, axis=0)
            self.cal_X = self.cal_X[-self.window_size :]
        if self.weighting == "uniform":
            cal_weights = self.get_weights()
            if self.threshold_function == "weighting":
                self.score_threshold = self._score_threshold_func(
                    cal_weights, false_alarm=self.false_alarm
                )


# ── TSAD data alignment ───────────────────────────────────────────────────────


def _get_tsfm_dataloaders(
    df_dataframe, model_config, dataset_config_dictionary, scaling=False
):
    from tsfm_public.toolkit.dataset import ForecastDFDataset
    from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
    from tsfm_public.toolkit.util import select_by_index

    forecast_horizon = model_config["prediction_length"]
    context_length = model_config["context_length"]
    assert context_length <= len(df_dataframe), (
        " length of dataframe needs to be >= context length"
    )

    column_specifiers = dataset_config_dictionary["column_specifiers"]
    id_columns = dataset_config_dictionary.get("id_columns", [])

    data = select_by_index(
        df_dataframe, id_columns=[], start_index=0, end_index=len(df_dataframe)
    )

    tsp = TimeSeriesPreprocessor(
        **column_specifiers, scaling=scaling, encode_categorical=False
    )
    tsp = tsp.train(data)

    dataset_inference = ForecastDFDataset(
        tsp.preprocess(data),
        **column_specifiers,
        context_length=context_length,
        prediction_length=forecast_horizon,
        id_columns=id_columns,
    )
    return dataset_inference


def _tsfm_dataloader_to_array(
    dataset_calibration, ix_target_features, x_context_window=-1
):
    y_gt = []
    X = []
    timestamp_id_value_dic: dict = {}
    for i in range(len(dataset_calibration)):
        try:
            y_gt.append(
                dataset_calibration[i]["future_values"][:, ix_target_features]
                .detach()
                .numpy()
            )
        except Exception as ex:
            raise ValueError(
                f"At least one of the target_columns is not in the input file: {ix_target_features}"
            ) from ex
        X.append(dataset_calibration[i]["past_values"].detach().numpy())
        if "timestamp" in dataset_calibration[i]:
            timestamp_id_value_dic.setdefault("timestamp", []).append(
                dataset_calibration[i]["timestamp"]
            )
        if "id" in dataset_calibration[i]:
            timestamp_id_value_dic.setdefault("id", []).extend(
                list(dataset_calibration[i]["id"])
            )
    y_gt_arr = np.array(y_gt)
    if len(y_gt_arr.shape) > 1:
        y_gt_arr = y_gt_arr[:, 0]
    y_gt_arr = np.squeeze(y_gt_arr)
    X_arr = np.array(X)
    if x_context_window > 0:
        X_arr = X_arr[:, -int(x_context_window) :, :]
    X_arr = X_arr.reshape([X_arr.shape[0], -1])
    return X_arr, y_gt_arr, timestamp_id_value_dic


def _get_tsad_aligned_data(
    df_data, dataset_config, ad_config, tsmodel_prediction_dictionary
):
    from tsfm_public.toolkit.time_series_preprocessor import create_timestamps

    context_length = ad_config["context_length"]
    prediction_length = ad_config["prediction_length"]
    scaling = ad_config["scaling"]
    ix_target_features = list(
        np.arange(len(dataset_config["column_specifiers"]["target_columns"]))
    )

    df_data[dataset_config["column_specifiers"]["timestamp_column"]] = pd.to_datetime(
        df_data[dataset_config["column_specifiers"]["timestamp_column"]]
    )
    dataset_inference = _get_tsfm_dataloaders(
        df_data,
        {"prediction_length": prediction_length, "context_length": context_length},
        dataset_config,
        scaling=scaling,
    )
    X, y_gt, timestamp_id_value_dic = _tsfm_dataloader_to_array(
        dataset_inference, ix_target_features, x_context_window=context_length
    )

    source_timestamp = np.array(tsmodel_prediction_dictionary["timestamp"])[:, 0]
    target_timestamp = timestamp_id_value_dic["timestamp"]

    forecast_horizon = 1
    target_timestamp_updated = []
    for ts in target_timestamp:
        ts_updated = create_timestamps(
            last_timestamp=ts,
            time_sequence=target_timestamp,
            periods=forecast_horizon,
        )[0]
        target_timestamp_updated.append(ts_updated)
    target_timestamp = np.array(
        np.array(target_timestamp_updated, dtype="datetime64[ns]")
    )
    source_timestamp = np.array(np.array(source_timestamp, dtype="datetime64[ns]"))

    frequency_sampling_median = np.median(target_timestamp[1:] - target_timestamp[:-1])
    tolerance_frequency_sampling = 0.2

    time_diff = np.abs(target_timestamp[:, None] - source_timestamp)
    matching_pairs = np.where(
        time_diff <= frequency_sampling_median * tolerance_frequency_sampling
    )
    index_timestamp = matching_pairs[0]
    index_timestamp_source = matching_pairs[1]

    X_cp = X[index_timestamp]
    y_gt_cp = y_gt[index_timestamp]
    y_pred = np.array(tsmodel_prediction_dictionary["target_prediction"])[
        index_timestamp_source, 0, 0
    ]
    timestamps_source = np.array(source_timestamp)[index_timestamp_source]

    return {
        "X": X_cp,
        "y_gt": y_gt_cp,
        "y_pred": y_pred,
        "timestamp": timestamps_source,
    }


# ── TSAD orchestration ────────────────────────────────────────────────────────

_AD_CONFIG_DEFAULT = {
    "ad_model_type": "timeseries_conformal",
    "context_length": 1,
    "false_alarm": 0.01,
    "window_size": None,
    "weighting": "uniform",
    "weighting_params": {},
}


class _TimeSeriesAnomalyDetectionConformalWrapper:
    def run(
        self,
        dataset_path: str,
        dataset_config_dictionary: dict,
        tsmodel_prediction_dictionary: dict,
        ad_model_checkpoint: str = None,
        ad_model_save: str = None,
        task: str = "inference",
        ad_model_type: str = None,
        n_calibration=None,
        false_alarm: float = None,
        context_length: int = None,
    ) -> dict:
        ad_model = None
        if ad_model_checkpoint is not None:
            if os.path.exists(ad_model_checkpoint):
                assert os.path.exists(ad_model_checkpoint + "/model.pkl")
                assert os.path.exists(ad_model_checkpoint + "/config.json")
                with open(ad_model_checkpoint + "/model.pkl", "rb") as _f:
                    ad_model = pickle.load(_f)
                with open(ad_model_checkpoint + "/config.json") as _f:
                    ad_config = json.load(_f)
                ad_model_type = ad_config["ad_model_type"]
                context_length = ad_config["context_length"]
                if false_alarm is None:
                    false_alarm = ad_config["false_alarm"]
                elif ad_config["false_alarm"] != false_alarm:
                    if task != "fit":
                        false_alarm = ad_config["false_alarm"]
        else:
            ad_config = {
                "context_length": context_length
                if context_length is not None
                else _AD_CONFIG_DEFAULT["context_length"],
                "false_alarm": false_alarm
                if false_alarm is not None
                else _AD_CONFIG_DEFAULT["false_alarm"],
                "ad_model_type": ad_model_type
                if ad_model_type is not None
                else _AD_CONFIG_DEFAULT["ad_model_type"],
            }
            context_length = ad_config["context_length"]
            false_alarm = ad_config["false_alarm"]
            ad_model_type = ad_config["ad_model_type"]

        df_data = _read_ts_data(
            dataset_path, dataset_config_dictionary=dataset_config_dictionary
        )
        context_length = ad_config["context_length"]
        output_tsad_aligned = _get_tsad_aligned_data(
            df_data,
            dataset_config_dictionary,
            ad_config={
                "prediction_length": 1,
                "context_length": context_length,
                "scaling": False,
            },
            tsmodel_prediction_dictionary=tsmodel_prediction_dictionary,
        )

        timestamps_source = output_tsad_aligned["timestamp"]
        X_cp = output_tsad_aligned["X"]
        y_gt_cp = output_tsad_aligned["y_gt"]
        y_pred = output_tsad_aligned["y_pred"]

        output_ad: dict = {}
        if task == "fit":
            if n_calibration is None:
                n_calibration = y_pred.shape[0]
            if n_calibration < 1:
                n_calibration = int(np.ceil(y_pred.shape[0] * n_calibration))
            n_calibration = int(n_calibration)
            n_critical = int(np.ceil(1 / false_alarm))
            assert n_calibration >= n_critical, (
                f" n_calibration should be >= {n_critical}, "
                f"otherwise increase false alarm to {round(1 / n_calibration, 2)}"
            )

            X_cp_cal = X_cp[:n_calibration]
            y_gt_cp_cal = y_gt_cp[:n_calibration]
            y_pred_cal = y_pred[:n_calibration]

            if ad_model_type in (
                "timeseries_conformal",
                "timeseries_conformal_adaptive",
            ):
                update = ad_model_type == "timeseries_conformal_adaptive"
                ad_model = _TSADWeightedConformalWrapper(
                    false_alarm=false_alarm,
                    weighting=_AD_CONFIG_DEFAULT["weighting"],
                    window_size=_AD_CONFIG_DEFAULT["window_size"],
                    weighting_params=_AD_CONFIG_DEFAULT["weighting_params"],
                    online_adaptive=update,
                )
                ad_model.fit(
                    y_cal_pred=np.array(y_pred_cal), y_cal_gt=np.array(y_gt_cp_cal)
                )
                output_prediction = ad_model.predict(
                    y_pred=np.array(y_pred), y_gt=np.array(y_gt_cp), update=update
                )
                output_ad = {
                    "timestamp": timestamps_source,
                    "KPI": tsmodel_prediction_dictionary["target_columns"],
                    "value": np.array(y_gt_cp),
                    "upper_bound": output_prediction["prediction_interval"]["y_high"],
                    "lower_bound": output_prediction["prediction_interval"]["y_low"],
                    "anomaly_score": 1 - output_prediction["outliers_scores"],
                    "anomaly_label": output_prediction["outliers"] == 1,
                    "split": [
                        "calibration" if i < n_calibration else "test"
                        for i in range(y_pred.shape[0])
                    ],
                }

            if ad_model is not None and ad_model_save is not None:
                with open(ad_model_save + "/model.pkl", "wb") as _f:
                    pickle.dump(ad_model, _f)
                with open(ad_model_save + "/config.json", "w") as _f:
                    json.dump(ad_config, _f)

        if task == "inference":
            if false_alarm is None:
                false_alarm = ad_model.false_alarm
            output_prediction = ad_model.predict(
                y_pred=np.array(y_pred), y_gt=np.array(y_gt_cp)
            )
            output_ad = {
                "timestamp": timestamps_source,
                "KPI": tsmodel_prediction_dictionary["target_columns"],
                "value": np.array(y_gt_cp),
                "upper_bound": output_prediction["prediction_interval"]["y_high"],
                "lower_bound": output_prediction["prediction_interval"]["y_low"],
                "anomaly_score": 1 - output_prediction["outliers_scores"],
                "anomaly_label": output_prediction["outliers"] == 1,
                "split": ["test"] * output_prediction["outliers_scores"].shape[0],
            }

        return output_ad
