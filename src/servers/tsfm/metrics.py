"""Forecasting metrics, spectral/trend loss functions, and frequency token constants."""

from __future__ import annotations

import numpy as np


# ── Basic forecasting metrics ─────────────────────────────────────────────────


def _RMSE(y_true, y_pred, axis=None):
    values = (
        np.mean((y_true - y_pred) ** 2)
        if axis is None
        else np.mean((y_true - y_pred) ** 2, axis=axis)
    )
    return np.sqrt(values)


def _MAE(y_true, y_pred, axis=None):
    if axis is None:
        return np.mean(np.abs(y_true - y_pred))
    return np.mean(np.abs(y_true - y_pred), axis=axis)


def _MAPE(y_true, y_pred, axis=None):
    non_zero_mask = np.array(y_true != 0).astype("int")
    y_true_denom = np.array(y_true) * non_zero_mask + (1 - non_zero_mask) * 1e-15
    values = np.abs((y_true - y_pred)) / np.abs(y_true_denom)
    if axis is None:
        if np.sum(non_zero_mask) > 0:
            return np.sum(values * non_zero_mask) / np.sum(non_zero_mask) * 100
        return None
    if np.sum(non_zero_mask) > 0:
        numerator = np.sum(values * non_zero_mask, axis=axis)
        denominator = np.sum(non_zero_mask, axis=axis)
        output = 100 * numerator
        output[denominator == 0] = None
        output[denominator != 0] /= denominator[denominator != 0]
        return output
    return None


def _SMAPE(y_true, y_pred, axis=None):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    non_zero_mask = np.array(denominator != 0).astype("int")
    denominator = np.array(denominator) * non_zero_mask + (1 - non_zero_mask) * 1e-15
    values = np.abs(y_true - y_pred) / denominator
    if axis is None:
        return np.mean(values) * 100
    return np.mean(values, axis=axis) * 100


def _WAPE(y_true, y_pred, axis=None):
    numerator = np.abs((y_true - y_pred))
    if axis is None:
        denominator = np.sum(np.abs(y_true))
        if denominator == 0:
            return None
        return np.sum(numerator) / denominator
    denominator = np.sum(np.abs(y_true), axis=axis)
    values = np.sum(numerator, axis=axis)
    values[denominator == 0] = None
    values[denominator != 0] /= denominator[denominator != 0]
    return values


def _Bias(y_true, y_pred, axis=None):
    delta = y_pred - y_true
    if axis is None:
        return np.mean(delta)
    return np.mean(delta, axis=axis)


def _NRMSE(y_true, y_pred, axis=None, norm="mean"):
    values = _RMSE(y_true, y_pred, axis=axis)
    den = (np.max(y_true) - np.min(y_true)) if norm == "minmax" else np.mean(y_true)
    return np.sqrt(values) / np.abs(den)


def _cosine_similarity_matrix(A, B, axis=1):
    dot_product = np.sum(A * B, axis=axis)
    norm_A = np.linalg.norm(A, axis=axis)
    norm_B = np.linalg.norm(B, axis=axis)
    return dot_product / (norm_A * norm_B)


# ── Spectral / trend losses (require torch lazily) ────────────────────────────


def _loss_helper(outputs, targets, fn, axis=1):
    import torch

    outputs = outputs.astype(np.float64)
    targets = targets.squeeze()
    outputs = outputs.squeeze()
    if len(targets.shape) == 0 or targets.shape[0] == 0:
        return np.array(0.0)
    if len(targets.shape) == 1:
        if targets.shape[0] < 4:
            return np.array([0.0])
        return (
            fn(
                torch.from_numpy(targets.reshape(1, -1)),
                torch.from_numpy(outputs.reshape(1, -1)),
            )
            .cpu()
            .detach()
            .item()
        )
    B, T = targets.shape
    if axis == 1:
        if T < 4:
            return np.array([0.0])
        return (
            fn(torch.from_numpy(targets), torch.from_numpy(outputs))
            .cpu()
            .detach()
            .numpy()
        )
    return np.array(0.0)


def _amp_loss(outputs, targets):
    import torch

    B, T = outputs.shape
    fft_size = 1 << (2 * T - 1).bit_length()
    out_f = torch.fft.fft(outputs, fft_size, dim=-1)
    tgt_f = torch.fft.fft(targets, fft_size, dim=-1)
    out_norm = torch.linalg.vector_norm(outputs, dim=-1)
    tgt_norm = torch.linalg.vector_norm(targets, dim=-1)
    auto_corr = torch.fft.ifft(tgt_f * tgt_f.conj(), dim=-1).real
    auto_corr = torch.cat([auto_corr[..., -(T - 1) :], auto_corr[..., :T]], dim=-1)
    norm = torch.where(tgt_norm == 0, 1e-9, tgt_norm * tgt_norm)
    nac_tgt = auto_corr / norm.unsqueeze(1)
    cross_corr = torch.fft.ifft(tgt_f * out_f.conj(), dim=-1).real
    cross_corr = torch.cat([cross_corr[..., -(T - 1) :], cross_corr[..., :T]], dim=-1)
    norm2 = torch.where(tgt_norm * out_norm == 0, 1e-9, tgt_norm * out_norm)
    nac_out = cross_corr / (tgt_norm * out_norm).unsqueeze(1)
    return torch.mean(torch.abs(nac_tgt - nac_out), dim=-1)


def _ashift_loss(outputs, targets):
    import torch

    B, T = outputs.shape
    return T * torch.mean(
        torch.abs(1 / T - torch.softmax(outputs - targets, dim=-1)), dim=-1
    )


def _phase_loss(outputs, targets):
    import torch

    B, T = outputs.shape
    out_f = torch.fft.fft(outputs, dim=-1)
    tgt_f = torch.fft.fft(targets, dim=-1)
    tgt_f_sq = tgt_f.real**2 + tgt_f.imag**2
    mask = (tgt_f_sq > T).float()
    topk_indices = tgt_f_sq.topk(k=int(T**0.5), dim=-1).indices
    mask = mask.scatter_(-1, topk_indices, 1.0)
    mask[..., 0] = 1.0
    mask = torch.where(mask > 0, 1.0, 0.0)
    mask = mask.bool()
    not_mask = (~mask).float()
    not_mask /= torch.mean(not_mask, dim=-1).unsqueeze(1)
    zero_error = torch.abs(out_f) * not_mask
    zero_error = torch.where(
        torch.isnan(zero_error), torch.zeros_like(zero_error), zero_error
    )
    mask_f = mask.float()
    mask_f /= torch.mean(mask_f, dim=-1).unsqueeze(1)
    ae = torch.abs(out_f - tgt_f) * mask_f
    ae = torch.where(torch.isnan(ae), torch.zeros_like(ae), ae)
    return (torch.mean(zero_error, dim=-1) + torch.mean(ae, dim=-1)) / (T**0.5)


def _tildeq_loss(target, output):
    amp = _amp_loss(target, output)
    shift = _ashift_loss(target, output)
    phase = _phase_loss(target, output)
    return 0.5 * phase + 0.5 * shift + 0.01 * amp


def _TILDEQ(outputs, targets, axis=1):
    return _loss_helper(outputs, targets, _tildeq_loss, axis=axis)


def _derivatives(inp, device="cpu"):
    import torch

    batch_size, lens = inp.shape[0:2]
    input2 = inp[:, 2:lens].to(device)
    input1 = inp[:, 0 : lens - 2].to(device)
    return input2 - input1


def _w_mse(targets, outputs, device="cpu"):
    import torch

    batch_size, lens = targets.shape[0:2]
    t1 = targets[:, 1:lens].to(device)
    t2 = targets[:, 0 : lens - 1].to(device)
    o1 = outputs[:, 1:lens].to(device)
    o2 = outputs[:, 0 : lens - 1].to(device)
    sigma = torch.tanh((t1 - t2) * (o1 - o2))
    nt = targets[:, 1:lens].to(device)
    no = outputs[:, 1:lens].to(device)
    return torch.abs(no - nt) * (1.0 - sigma)


def _trend_loss(targets, outputs, alpha=0.5, device="cpu"):
    import torch

    sq_error = _w_mse(targets, outputs, device)
    error1 = torch.mean(sq_error, dim=-1)
    x1 = _derivatives(targets, device)
    x2 = _derivatives(outputs, device)
    _xt1 = x1.squeeze()
    _xt2 = x2.squeeze()
    if len(_xt1.shape) == 1:
        _xt1 = torch.reshape(_xt1, (1, _xt1.shape[0]))
    _xt1 = _xt1.T
    _xt2 = _xt2.T
    xc1 = (_xt1 - _xt1.mean(dim=0)).T
    xc2 = (_xt2 - _xt2.mean(dim=0)).T
    p_corr = torch.nn.functional.cosine_similarity(xc1, xc2, dim=-1)
    w_corr = 1 - p_corr
    dd = torch.norm(targets - outputs, dim=-1)
    return error1 + alpha * w_corr * dd


def _TREND(targets, outputs, axis=1):
    return _loss_helper(outputs, targets, _trend_loss, axis=axis)


_METRICS_FORECAST = {
    "RMSE": _RMSE,
    "MAE": _MAE,
    "MAPE": _MAPE,
    "SMAPE": _SMAPE,
    "WAPE": _WAPE,
    "Bias": _Bias,
    "NRMSE": _NRMSE,
    "TREND": _TREND,
    "TILDEQ": _TILDEQ,
    "COSSIM": _cosine_similarity_matrix,
}


# ── Frequency token constants ─────────────────────────────────────────────────

_freq_token_mapping = {
    "oov": 0,
    "minutely": 1,
    "2_minutes": 2,
    "5_minutes": 3,
    "10_minutes": 4,
    "15_minutes": 5,
    "half_hourly": 6,
    "hourly": 7,
}
_freq_token_to_minutes = {
    "oov": None,
    "minutely": 1,
    "2_minutes": 2,
    "5_minutes": 5,
    "10_minutes": 10,
    "15_minutes": 15,
    "half_hourly": 30,
    "hourly": 60,
}
_TSFREQUENCY_TOLERANCE = 0.2
