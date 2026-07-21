"""
.. module:: functions
   :synopsis: Functions for feature extraction.
   Some of functions of this module are taken from following repository and modified as per the requirement.
   https://github.com/blue-yonder/tsfresh/blob/master/tsfresh/feature_extraction/feature_calculators.py
   License Information:
   __author__ = "Maximilian Christ, Blue Yonder GmbH"
   __copyright__ = "Maximilian Christ, Blue Yonder GmbH"
   __license__ = "MIT"
   Some of functions  of this module are taken from following repository and modified as per the requirement.
   https://github.com/isadoranun/FATS/blob/master/FATS/FeatureFunctionLib.py
   License Information:
   __author__ = "Isadora Nun"
   __copyright__ = "Isadora Nun"
   __license__ = "MIT"
.. moduleauthor:: SROM Team
"""
import math
import numpy as np
import pandas as pd
import scipy.special as sp
import warnings
import itertools
import scipy.stats
from scipy.stats import linregress
import logging

LOGGER = logging.getLogger(__name__)

try:
    from statsmodels.tsa import stattools
    from statsmodels.tsa.stattools import acf, adfuller, pacf
    from statsmodels.tools.sm_exceptions import MissingDataError
except ImportError as e:
    raise ImportError(
        "feature_extraction requires statsmodels: pip install statsmodels"
    ) from e

try:
    from pywt import wavedec
except ImportError:
    LOGGER.error("ImportError in function.py : PyWavelets is not installed ")

from scipy.stats import anderson
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.stats import moment
from scipy.stats import trim_mean
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import mode
from scipy.stats.mstats import gmean
from scipy.stats import zscore
from numpy.linalg import LinAlgError
from numpy import cov, corrcoef
from math import factorial, log
from scipy.signal import periodogram, welch
from scipy.fftpack import dct
from collections import Counter

warnings.filterwarnings("ignore")
LEPS = 1e-20
THRESHOLD_TO_USE_FFT = 1250


def check_single_dimensional_array(x):
    """
    Check if input is single dimensional array.
    Parameters:
        x (numpy array, required): array of numbers.
    Return:
        return array or exception.
    """
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        x = x.to_numpy()
    if x.dtype == "object":
        x = x.astype(np.float64)
    if not isinstance(x, np.ndarray):
        raise ValueError("input is not numpy array")
    if x.ndim > 1:
        raise ValueError("input is not single dimensional array")
    return x


def remove_nan_single_dimensional_array(x):
    """
    remove nan from single dimensional array.
    Parameters:
        x (numpy array, required): array of numbers.
    Return:
        return array.
    """
    # we come to know isnan is not supported for object type of numpy array
    if x.dtype == "object":
        return x[~pd.isnull(x)]
    return x[~np.isnan(x)]


def remove_zero_single_dimensional_array(x):
    """
    remove zero from single dimensional array.
    Parameters:
        x (numpy array, required): array of numbers.
    Return:
        return array.
    """
    return x[x != 0]


def length(x, ignore_nan=True):
    """
    Computes mean of an array.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored and aggregates
        are calculated.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    return len(x)


def sum_values(x, ignore_nan=True):
    """
    Computes sum of an array.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored and aggregates
        are calculated.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    return np.sum(x)


def minimum(x, ignore_nan=True):
    """
    Computes min element of an array.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored and aggregates
        are calculated.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return np.min(x)


def maximum(x, ignore_nan=True):
    """
    Computes maximum element of an array.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored and aggregates
        are calculated.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return np.max(x)


def variance(x, ignore_nan=True):
    """
    Computes variance of an array.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored and aggregates
        are calculated.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return np.var(x)


def count(x, ignore_nan=True):
    """
    Computes length of an array, count on number of elements.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        integer.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    return len(x)


def quantile(x, q=0.5, ignore_nan=True):
    """
    Computes quartile of distribution in an array.
    Parameters:
        x (numpy array, required): array of numbers.
        q (float, required): quantile value for the distribution. Default 0.5
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return np.quantile(x, q)


def quantile_25(x, ignore_nan=True):
    """
    Computes first quartile of distribution in an array.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    return quantile(x, 0.25, ignore_nan)


def quantile_75(x, ignore_nan=True):
    """
    Computes third quartile of distribution in an array.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    return quantile(x, 0.75, ignore_nan)


def quantile_range(x, ignore_nan=True):
    """
    Computes quantile range of array.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    return quantile(x, 0.75, ignore_nan) - quantile(x, 0.25, ignore_nan)


def rate_of_change(x, ignore_nan=True):
    """
    Computes rate of change.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) <= 1:
        return float(np.nan)
    return (x[len(x) - 1] - x[0]) / (len(x) * 1.0)


def sum_of_change(x, ignore_nan=True):
    """
    Computes sum of change.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        number.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) <= 1:
        return float(np.nan)
    x = np.diff(x)
    return np.sum(x)


def absoluate_sum_of_changes(x, ignore_nan=True):
    """
    Computes absolute sum of changes.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        number.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) <= 1:
        return float(np.nan)
    x = np.diff(x)
    return np.sum(abs(x))


def absoluate_sum_of_change(x, ignore_nan=True):
    """
    Computes absolute sum of change
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    return absoluate_sum_of_changes(x, ignore_nan)


def trend_slop(x, ignore_nan=True):
    """
    Computes trend slope.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    slope = float(np.nan)
    if len(x) <= 1:
        return slope
    x1 = list(range(len(x)))
    try:
        slope, _, _, _, _ = linregress(x1, x)
    except:
        return slope
    return slope


def mean_abs_change(x, ignore_nan=True):
    """
    Computes mean absolute change.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        number.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return np.mean(abs(np.diff(x)))


def mean_second_derivative_central(x, ignore_nan=True):
    """
    Computes mean second derivate central.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        number.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    diff = (np.roll(x, 1) - 2 * x + np.roll(x, -1)) / 2.0
    return np.mean(diff[1:-1])


def last_location_of_maximum(x, ignore_nan=True):
    """
    Computes last location of maximum.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)

    return 1.0 - np.argmax(x[::-1]) / len(x) if len(x) > 0 else float(np.nan)


def last_location_of_minimum(x, ignore_nan=True):
    """
    Computes last location of minimum.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)

    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else float(np.nan)


def first_location_of_maximum(x, ignore_nan=True):
    """
    Computes first location of maximum.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return np.argmax(x) / len(x) if len(x) > 0 else float(np.nan)


def first_location_of_minimum(x, ignore_nan=True):
    """
    Computes first location of minimum.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return np.argmin(x) / len(x) if len(x) > 0 else float(np.nan)


def ratio_beyond_r_sigma(x, r=2.0, ignore_nan=True):
    """
    Computes ratio of values that are more than r*std(x) \
    (so r sigma) away from the mean of x.
    Parameters:
        x (numpy array, required): array of numbers.
        r (float, required): float. Default 2.0.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return np.sum(np.abs(x - np.mean(x)) > r * np.std(x)) / len(x)


def large_standard_deviation(x, r=2.0, ignore_nan=True):
    """
    Computes if std deviation is larger than r times diff of max and min.
    Parameters:
        x (numpy array, required): array of numbers.
        r (float, required): float. Default 2.0.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        integer.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return 0
    return int(np.std(x) > (r * (np.max(x) - np.min(x))))


def longest_strike_below_mean(x, ignore_nan=True):
    """
    Computes longest strike below mean.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    res = [0]
    x1 = x <= np.mean(x)
    if len(x1) > 1:
        res = [len(list(group)) for value, group in itertools.groupby(x1) if value == 1]
    return np.max(res) if len(x) > 0 else 0


def longest_strike_above_mean(x, ignore_nan=True):
    """
    Computes longest strike above mean.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    res = [0]
    x1 = x >= np.mean(x)
    if len(x1) > 1:
        res = [len(list(group)) for value, group in itertools.groupby(x1) if value == 1]
    return np.max(res) if len(x) > 0 else 0


def percentage_of_reoccurring_datapoints_to_all_datapoints(x, ignore_nan=True):
    """
    Computes the percentage of unique values, that are present more than once.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return 0
    _, counts = np.unique(x, return_counts=True)
    if counts.shape[0] == 0:
        return 0
    return np.sum(counts > 1) / float(counts.shape[0])


def percentage_of_reoccurring_values_to_all_values(x, ignore_nan=True):
    """
    Computes percentage of recurring values.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return 0
    x = pd.Series(x)
    value_counts = x.value_counts()
    return value_counts[value_counts > 1].sum() / len(x)


def sum_of_reoccurring_values(x, ignore_nan=True):
    """
    Computes the sum of all values, that are present more than once.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return 0
    unique, counts = np.unique(x, return_counts=True)
    counts[counts < 2] = 0
    counts[counts > 1] = 1
    return np.sum(counts * unique)


def sum_of_reoccurring_data_points(x, ignore_nan=True):
    """
    Computes the sum of all values, that are present more than once.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return 0
    unique, counts = np.unique(x, return_counts=True)
    counts[counts < 2] = 0
    return np.sum(counts * unique)


def ratio_unique_value_number_to_time_series_length(x, ignore_nan=True):
    """
    Computes a factor which is 1 if all values occur only once,
    and below one if this is not the case.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if len(x) < 1:
        return 0
    return len(np.unique(x)) / len(x)


def corr_coefficient(x, ignore_nan=True):
    """
    Computes coefficient.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    r_value = float(np.nan)
    if len(x) <= 1:
        return r_value
    x1 = list(range(len(x)))
    try:
        _, _, r_value, _, _ = linregress(x1, x)
    except:
        return r_value
    return r_value


def delta_diff(x, ignore_nan=True):
    """
    Computes the difference.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        number.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return x[-1] - x[0]


def past_value(x, ignore_nan=True):
    """
    Computes the head, i.e., oldest value in the window
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        number.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return x[0]


def fft_coefficient_real(x, ignore_nan=True):
    """
    Computes the real part of resulting fft coefficient.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        ndarray.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    fft = np.fft.rfft(x)
    return fft.real


def fft_coefficient_abs(x, ignore_nan=True):
    """
    Computes the absolute value of resulting fft coefficient.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        ndarray.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    fft = np.fft.rfft(x)
    return np.abs(fft)


def amplitude(x, ignore_nan=True):
    """
    Computes half the difference between the maximum and the minimum magnitude.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    sorted_mag = np.sort(x)
    return (
        np.median(sorted_mag[-int(math.ceil(0.05 * len(x))) :])
        - np.median(sorted_mag[0 : int(math.ceil(0.05 * len(x)))])
    ) / 2.0


def rate_of_cumulative_sum(x, ignore_nan=True):
    """
    Computes range of cumulative sum.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        number.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    sigma = np.std(x)
    m = np.mean(x)
    s = np.cumsum(x - m) * 1.0 / (len(x) * sigma)
    r = np.max(s) - np.min(s)
    return r


def mean_variance(x, ignore_nan=True):
    """
    Computes variability index.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return np.std(x) / np.mean(x)


def _tmp_acf(x, nlags):
    """ 
    Internal method as a helper method
    """
    # Slice the relevant subseries based on the lag
    y1 = x[: (len(x) - nlags)]
    y2 = x[nlags:]
    # Subtract the mean of the whole series x to calculate Cov
    sum_product = np.sum((y1 - np.mean(x)) * (y2 - np.mean(x)))
    # Normalize with var of whole series
    return sum_product / ((len(x) - nlags) * np.var(x))


def autocor_length(x, lags=None, ignore_nan=True):
    """
    Computes auto correlation length.
    Parameters:
        x (numpy array, required): array of numbers.
        lags (integer): integer.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return 0
    if lags is None:
        lags = min(int(len(x / 2)), 100)
    ac = _tmp_acf(x, nlags=lags)
    if isinstance(ac, float):
        ac = [ac]
    k = next((index for index, value in enumerate(ac) if value < np.exp(-1)), None)
    if k is None:
        return 1
    return k


def outlier_segment_count(x, consecutive=3, ignore_nan=True):
    """
    Computes the count of given number of consecutive measurements
    that are out of 2sigma range, and normalized by N-2.
    Parameters:
        x (numpy array, required): array of numbers.
        consecutive (integer): integer.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        integer.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    n = len(x)
    if n < consecutive:
        return 0
    if n < 1:
        return 0
    sigma = np.std(x)
    m = np.mean(x)
    count = 0
    for i in range(n - consecutive + 1):
        flag = 0
        for j in range(consecutive):
            if x[i + j] > m + 2 * sigma or x[i + j] < m - 2 * sigma:
                flag = 1
            else:
                flag = 0
                break
        if flag:
            count = count + 1
    return count * 1.0 / (n - consecutive + 1)


def small_kurtosis(x, ignore_nan=True):
    """
    Computes small sample kurtosis of the array.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    n = len(x)
    if n < 1:
        return float(np.nan)
    mean = np.mean(x)
    std = np.std(x)
    s = sum(((x - mean) / std) ** 4)
    c1 = float(n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
    c2 = float(3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return c1 * s - c2


def median_abs_dev(x, ignore_nan=True):
    """
    Computes median absolute deviation of an array.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    return np.median(abs(x - np.median(x)))


def median_buffer_range_percentage(x, ignore_nan=True):
    """
    Computes median buffer range percentage.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    median = np.median(x)
    amp = (np.max(x) - np.min(x)) / 10
    count = np.sum(np.logical_and(x < median + amp, x > median - amp))
    return float(count) / len(x)


def pair_slope_trend(x, n=None, ignore_nan=True):
    """
    Computes considering the last n (time-sorted) measurements of array,
    the fraction of increasing first differences minus the fraction of
    decreasing first differences.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    if n is None:
        n = int(len(x) / 4)
    data_last = x[-n:]
    return (
        float(
            len(np.where(np.diff(data_last) > 0)[0])
            - len(np.where(np.diff(data_last) <= 0)[0])
        )
        / n
    )


def flux_percentile_ratio_mid20(x, ignore_nan=True):
    """
    Computes flux percentile ratio of mid 20.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    per_60 = np.percentile(x, 60, interpolation="nearest")
    per_40 = np.percentile(x, 40, interpolation="nearest")
    per_95 = np.percentile(x, 95, interpolation="nearest")
    per_5 = np.percentile(x, 5, interpolation="nearest")
    return (per_60 - per_40) / (per_95 - per_5)


def flux_percentile_ratio_mid35(x, ignore_nan=True):
    """
    Computes flux percentile ratio of mid 35.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    per_675 = np.percentile(x, 67.5, interpolation="nearest")
    per_325 = np.percentile(x, 32.5, interpolation="nearest")
    per_95 = np.percentile(x, 95, interpolation="nearest")
    per_5 = np.percentile(x, 5, interpolation="nearest")
    return (per_675 - per_325) / (per_95 - per_5)


def flux_percentile_ratio_mid50(x, ignore_nan=True):
    """
    Computes flux percentile ratio of mid 50.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    per_25 = np.percentile(x, 25, interpolation="nearest")
    per_75 = np.percentile(x, 75, interpolation="nearest")
    per_95 = np.percentile(x, 95, interpolation="nearest")
    per_5 = np.percentile(x, 5, interpolation="nearest")
    return (per_75 - per_25) / (per_95 - per_5)


def flux_percentile_ratio_mid65(x, ignore_nan=True):
    """
    Computes flux percentile ratio of mid 65.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    per_175 = np.percentile(x, 17.5, interpolation="nearest")
    per_825 = np.percentile(x, 82.5, interpolation="nearest")
    per_95 = np.percentile(x, 95, interpolation="nearest")
    per_5 = np.percentile(x, 5, interpolation="nearest")
    return (per_825 - per_175) / (per_95 - per_5)


def flux_percentile_ratio_mid80(x, ignore_nan=True):
    """
    Computes flux percentile ratio of mid 80.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    per_10 = np.percentile(x, 10, interpolation="nearest")
    per_90 = np.percentile(x, 90, interpolation="nearest")
    per_95 = np.percentile(x, 95, interpolation="nearest")
    per_5 = np.percentile(x, 5, interpolation="nearest")
    return (per_90 - per_10) / (per_95 - per_5)


def percent_difference_flux_percentile(x, ignore_nan=True):
    """
    Computes percentile ratio against median.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    median_data = np.median(x)
    per_95 = np.percentile(x, 95, interpolation="nearest")
    per_5 = np.percentile(x, 5, interpolation="nearest")
    return (per_95 - per_5) / median_data


def percent_amplitude(x, ignore_nan=True):
    """
    Computes percent amplitude of an array.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    median_data = np.median(x)
    distance_median = np.abs(x - median_data)
    max_distance = np.max(distance_median)
    percent_amplitude = max_distance / median_data
    return percent_amplitude


def anderson_darling(x, ignore_nan=True):
    """
    Computes anderson_darling of an array.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    ander = anderson(x)[0]
    return 1 / (1.0 + np.exp(-10 * (ander - 0.3)))


def g_skew(x, ignore_nan=True):
    """
    Computes median-based measure of the skew of an array.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    median_mag = np.median(x)
    f_3_value = np.percentile(x, 3)
    f_97_value = np.percentile(x, 97)
    return np.median(x[x <= f_3_value]) + np.median(x[x >= f_97_value]) - 2 * median_mag


def cumsum(x, ignore_nan=True):
    """
    Computes cumulative sum of an array.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        number.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return 0
    cum_sum = x.cumsum(axis=0)
    return cum_sum


def quantize(x, quantiles=50, ignore_nan=True):
    """
    Computes quantization of an array.
    Parameters:
        x (numpy array, required): array of numbers.
        quantiles (integer, required): integer. Default 50.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        numpy array.
    """
    x = check_single_dimensional_array(x)
    
    # FIX: Convert quantiles to integer in case it comes as a list or array
    if isinstance(quantiles, (list, tuple)):
        quantiles = int(quantiles[0]) if len(quantiles) > 0 else 50
    elif isinstance(quantiles, np.ndarray):
        quantiles = int(quantiles.flatten()[0]) if quantiles.size > 0 else 50
    else:
        quantiles = int(quantiles)
    
    # Validate quantiles is positive
    if quantiles <= 0:
        quantiles = 50
    
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    sig = pd.Series(x)
    q = pd.Series(range(0, quantiles)).apply(lambda x: sig.quantile(x / quantiles))
    m_curve = q.to_dict()

    def signal2quant(z):
        curve = m_curve
        return min(curve.keys(), key=lambda y: abs(float(curve[y]) - z))

    l_signal_q = sig.apply(signal2quant)
    np_array = l_signal_q.values
    return np_array

def box_cox(x, i_lambda=1, ignore_nan=True):
    """
    Computes box_cox of an array.
    Parameters:
        x (numpy array, required): array of numbers.
        i_lambda (float, required): float. Default 1.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        numpy array.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    global LEPS
    df = pd.Series(x)
    log_df = df.apply(lambda x: np.log(max(x, LEPS)))
    if abs(i_lambda) <= 0.01:
        np_array = log_df.values
        return np_array
    cal_df = (np.exp(log_df * i_lambda) - 1) / i_lambda
    np_array = cal_df.values
    return np_array


def differencing(x, ignore_nan=True):
    """
    Computes difference of an array.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        numpy array.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return np.diff(x)


def relative_differencing(x, ignore_nan=True):
    """
    Computes relative difference of an array.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        numpy array.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    df = pd.Series(x)
    global LEPS
    df1 = df.apply(lambda x: x if (abs(x) > LEPS) else LEPS)
    df_shifted = df1.shift(1)
    rate = (df1 - df_shifted) / df_shifted
    rate.iloc[0] = 0.0
    rate = rate.clip(-1.0e8, +1.0e8)
    np_array = rate.values
    return np_array


def logit(x, ignore_nan=True):
    """
    Computes logit of an array.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        numpy array.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return sp.logit(x)


def ans_combe(x, ignore_nan=True):
    """
    Computes ans_combe of an array.
    https://en.wikipedia.org/wiki/Anscombe_transform
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        numpy array.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    sig = pd.Series(x)
    M_CONSTANT = 3.0 / 8.0
    y = sig.apply(lambda x: 2 * np.sqrt(x + M_CONSTANT))
    np_array = y.values
    return np_array


def fisher(x, ignore_nan=True):
    """
    Computes fisher of an array.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        numpy array.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    sig = pd.Series(x)
    EPS = 1.0e-8
    y = sig.apply(lambda x: np.arctanh(np.clip(x, -1 + EPS, 1.0 - EPS)))
    np_array = y.values
    return np_array


def lumpiness(x, width=2, ignore_nan=True):
    """
    Computes lumpiness an array.
    Parameters:
        x (numpy array, required): array of numbers.
        width(integer): integer. Default 1.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    length = len(x)
    if length < 1:
        return float(np.nan)
    start = []
    end = []
    if (length < (2 * width)) or (width <= 0):
        return 0
    for i in range(0, length, width):
        start.append(i)
    for j in range(width - 1, length + width, width):
        end.append(j)
    n_segs = int(length / width)
    var_x = []
    for idx in range(0, n_segs):
        var_x.append(np.var(x[start[idx] : end[idx] + 1], ddof=1))
    return np.var(var_x, ddof=1)


def stability(x, width=1, ignore_nan=True):
    """
    Computes stability an array.
    Parameters:
        x (numpy array, required): array of numbers.
        width (integer): integer. Default 1.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    length = len(x)
    if length < 1:
        return float(np.nan)
    start = []
    end = []
    if (length < (2 * width)) or (width <= 0):
        return 0
    for i in range(0, length, width):
        start.append(i)
    for j in range(width - 1, length + width, width):
        end.append(j)
    n_segs = int(length / width)
    mean_x = []
    for idx in range(0, n_segs):
        mean_x.append(np.mean(x[start[idx] : end[idx] + 1]))
    return np.var(mean_x, ddof=1)


def median_crossing_points(x, ignore_nan=True):
    """
    Computes median crossing points of an array.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        integer.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    n = len(x)
    if n < 1:
        return 0
    midline = np.median(x)
    ab = list(map(lambda z: True if z <= midline else False, x))
    p1 = np.array(ab[0 : n - 1])
    p2 = np.array(ab[1:n])
    cross = (p1 & ~p2) | (p2 & ~p1)
    return np.sum(cross)


def mean_crossing_points(x, ignore_nan=True):
    """
    Computes mean crossing points of an array.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        integer.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    n = len(x)
    if n < 1:
        return 0
    midline = np.mean(x)
    ab = list(map(lambda z: True if z <= midline else False, x))
    p1 = np.array(ab[0 : n - 1])
    p2 = np.array(ab[1:n])
    cross = (p1 & ~p2) | (p2 & ~p1)
    return np.sum(cross)


def flat_spots(x, ignore_nan=True):
    """
    Computes flat spots of an array.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        integer.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return int(np.nan)
    try:
        cutx = pd.cut(x, 10, include_lowest=True, labels=False)
    except:
        return int(np.nan)
    from itertools import groupby

    lengths = [(sum(1 for i in g)) for k, g in groupby(cutx)]
    return max(lengths)


def high_low_mu(x, ignore_nan=True):
    """
    Computes high low mu statistic.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    mu = np.mean(x)  # mean of data
    mhi = np.mean(x[np.where(x > mu)])  # mean of data above the mean
    mlo = np.mean(x[np.where(x < mu)])  # mean of data below the mean
    return (mhi - mu) / (mu - mlo)


def burstiness(x, ignore_nan=True):
    """
    Computes burstiness statistic of a time series.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    r = np.std(x) / np.mean(x)
    return (r - 1) / (r + 1)


def improved_burstiness(x, ignore_nan=True):
    """
    Computes improved burstiness statistic of a time series.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    n = len(x)
    if n < 1:
        return float(np.nan)
    r = np.std(x) / np.mean(x)
    return (math.sqrt(n + 1) * r - math.sqrt(n - 1)) / (
        (math.sqrt(n + 1) - 2) * r + math.sqrt(n - 1)
    )


def pearson_skewness(x, ignore_nan=True):
    """
    Computes Pearson skewness.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return (3 * np.mean(x) - np.median(x)) / np.std(x)


def bowley_skewness(x, ignore_nan=True):
    """
    Computes Bowley skewness.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    q_25 = quantile(x, 0.25, ignore_nan)
    q_50 = quantile(x, 0.5, ignore_nan)
    q_75 = quantile(x, 0.75, ignore_nan)
    return (q_75 + q_25) - 2 * q_50 / (q_75 - q_25)


def histogram_mode(x, num_bins="auto", ignore_nan=True):
    """
    Computes the mode of the data vector using histograms with a given number.
    Parameters:
        x (numpy array, required): array of numbers.
        num_bins (string or integer): string or integer default 'auto'.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    n, bin_edges = np.histogram(x, bins=num_bins)
    bin_centers = np.mean([bin_edges[0:-1], bin_edges[1 : len(bin_edges)]], axis=0)
    return np.mean(bin_centers[np.where(n == max(n))])


def moments(x, mom=1, ignore_nan=True):
    """
    Computes moment of the distribution of the input time series.
    Parameters:
        x (numpy array, required): array of numbers.
        mom (integer): integer. Default 1.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return moment(x, moment=mom) / np.std(x)


def mean_outlier_test(x, per=2, ignore_nan=True):
    """
    Computes how distributional mean statistics depend on distributional outliers.
    Parameters:
        x (numpy array, required): array of numbers.
        per (integer): integer, the percentage of values to remove beyond upper and lower percentiles.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    ret = None
    try:
        ret = np.mean(
            x[np.where((x > np.percentile(x, per)) & (x < np.percentile(x, 100 - per)))]
        )
    except:
        ret = float(np.nan)
    return ret


def std_outlier_test(x, per=2, ignore_nan=True):
    """
    Computes how distributional statistics depend on distributional outliers.
    Parameters:
        x (numpy array, required): array of numbers.
        per (integer): integer, the percentage of values to remove beyond upper and lower percentiles.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    ret = None
    try:
        ret = np.std(
            x[np.where((x > np.percentile(x, per)) & (x < np.percentile(x, 100 - per)))]
        ) / np.std(x)
    except:
        ret = float(np.nan)
    return ret


def piecewise_aggregate_mean(x, n_segments=3, ignore_nan=True):
    """
    Computes piecewise aggregate mean.
    Parameters:
        x (numpy array, required): array of numbers.
        n_segments (integer): integer, default 3.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        numpy array.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    sz_segment = len(x)
    if sz_segment < 1:
        return float(np.nan)
    if sz_segment < n_segments:
        return np.mean(x)
    val_segment = []
    for i_seg in range(n_segments):
        start = i_seg * n_segments
        end = start + n_segments
        val_segment.append(x[start:end].mean())
    return np.array(val_segment)


def piecewise_aggregate_median(x, n_segments=3, ignore_nan=True):
    """
    Computes piecewise aggregate mean.
    Parameters:
        x (numpy array, required): array of numbers.
        n_segments (integer): integer, default 3.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        numpy array.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    sz_segment = len(x)
    if sz_segment < 1:
        return float(np.nan)
    if sz_segment < n_segments:
        return np.median(x)
    val_segment = []
    for i_seg in range(n_segments):
        start = i_seg * n_segments
        end = start + n_segments
        val_segment.append(np.median(x[start:end]))
    return np.array(val_segment)


def p_left(x, th=0.1, ignore_nan=True):
    """
    Computes distance from the mean at which a given proportion of data are more distant.
    Measures the maximum distance from the mean at which a given fixed proportion,
    p, of the time-series data points are further.
    Normalizes by the standard deviation of the time series
    Parameters:
        x (numpy array, required): array of numbers.
        th (float): float,default 0.1,the proportion of data further than p from the mean.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    p = quantile(abs(x - np.mean(x)), 1 - th)
    return p / np.std(x)


def proportion_values_positive(x, ignore_nan=True):
    """
    Computes statistics the proportion of positive values
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return np.sum(x[np.where(x > 0)]) / len(x)


def proportion_values_zero(x, ignore_nan=True):
    """
    Computes statistics on the proportion of zeros,
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return np.sum(x[np.where(x == 0)]) / len(x)


def mean_second_derivate_central(x, ignore_nan=True):
    """
    Computes mean second derivate central.

    Parameters:
        x (numpy array, required): Array of integers.

    Return:
        integer
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 2:
        return float(np.nan)
    diff = (np.roll(x, 1) - 2 * np.array(x) + np.roll(x, -1)) / 2.0
    return np.mean(diff[1:-1])


def mean_abs_dev(x, ignore_nan=True):
    """
    Computes mean absolute deviation of an array.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return np.mean(abs(x - np.mean(x)))


def trimmed_mean(x, n=0.1, ignore_nan=True):
    """
    Computes mean of the trimmed time series using trim_mean.
    Parameters:
        x (numpy array, required): array of numbers.
        n (number): number, the percent of highest and lowest values in x to exclude from the mean
        calculation.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return trim_mean(x, n)


def unique_prop(x, ignore_nan=True):
    """
    Computes proportion of the time series that are unique values.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return len(np.unique(x)) / len(x)


def proportion_within_std_mean(x, p=1, ignore_nan=True):
    """
    Computes proportion of data points within p standard deviations of the mean.
    Parameters:
        x (numpy array, required): array of numbers.
        p (number): number, the number (proportion) of standard deviations. Default 1.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    mu = np.mean(x)
    sig = np.std(x)
    try:
        ret = np.sum(x[np.where((x >= (mu - p * sig)) & (x <= (mu + p * sig)))]) / len(
            x
        )
    except:
        ret = float(np.nan)
    return ret


def proportion_within_std_median(x, p=1, ignore_nan=True):
    """
    Computes proportion of data points within p standard deviations of the mean.
    Parameters:
        x (numpy array, required): array of numbers.
        p (number): number, the number (proportion) of standard deviations.Default 1.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    mu = np.median(x)
    sig = 1.35 * quantile_range(x)
    try:
        ret = np.sum(x[np.where((x >= (mu - p * sig)) & (x <= (mu + p * sig)))]) / len(
            x
        )
    except:
        ret = float(np.nan)
    return ret


def coefficient_variation(x, k=1, ignore_nan=True):
    """
    Computes coefficient of variation of order k is sigma^k / mu^k (for sigma, standard
    deviation and mu, mean) of a data x.
    Parameters:
        x (numpy array, required): array of numbers.
        k (number): number, the order of coefficient of variation (k = 1 is default).
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    if k < 0:
        return float(np.nan)
    return (np.std(x)) ** k / (np.mean(x)) ** k


def mle_gaussian(x, ignore_nan=True):
    """
    Computes maximum likelihood distribution fit to gaussian data.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        numpy array.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return np.array(norm.fit(x))


def mle_uniform(x, ignore_nan=True):
    """
    Computes maximum likelihood distribution fit to uniform data.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        numpy array.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return np.array(uniform.fit(x))


def variance_larger_than_standard_deviation(x, ignore_nan=True):
    """
    Computes variable denoting if the variance of x is greater than its standard deviation.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        integer.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return 0
    y = np.var(x)
    return int(y > np.sqrt(y))


def has_duplicate_max(x, ignore_nan=True):
    """
    Computes if the maximum value of x is observed more than once.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        integer.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return 0
    return int(np.sum(x == np.max(x)) >= 2)


def has_duplicate_min(x, ignore_nan=True):
    """
    Computes if the minimal value of x is observed more than once.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        integer.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return 0
    return int(np.sum(x == np.min(x)) >= 2)


def has_duplicate(x, ignore_nan=True):
    """
    Computes if any value in x occurs more than once.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        integer.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return 0
    return int(len(x) != len(np.unique(x)))


def approximate_entropy(x, m=1, r=0.2, ignore_nan=True):
    """
    Computes approximate entropy of x.
    Parameters:
        x (numpy array, required): array of numbers.
        m (number): number.  The length of compared run of data. Default 1.
        r (number): number. The threshold for judging closeness/similarity. Default 0.2
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """

    def _maxdist(x_i, x_j):
        """
            Internal method as a helper method.
        """
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        """
            Internal method as a helper method.
        """
        y = [[x[j] for j in range(i, i + m - 1 + 1)] for i in range(n - m + 1)]
        c = [
            len([1 for y_j in y if _maxdist(y_i, y_j) <= r]) / (n - m + 1.0)
            for y_i in y
        ]
        return (n - m + 1.0) ** (-1) * sum(np.log(c))

    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    n = len(x)
    if n < 1:
        return float(np.nan)
    return abs(_phi(m + 1) - _phi(m))


def auto_correlation(x, lag=None, ignore_nan=True):
    """
    Computes the auto correlation of the specified lags.
    Parameters:
        x (numpy array, required): array of numbers.
        lag (integer) : integer. Default len(x)/2.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    if lag is None:
        lag = int(len(x) / 2)
    if len(x) < lag:
        return float(np.nan)
    # Slice the relevant subseries based on the lag
    y1 = x[: (len(x) - lag)]
    y2 = x[lag:]
    # Subtract the mean of the whole series x
    x_mean = np.mean(x)
    # The result is sometimes referred to as "covariation"
    sum_product = np.sum((y1 - x_mean) * (y2 - x_mean))
    # Return the normalized unbiased covariance
    v = np.var(x)
    if np.isclose(v, 0):
        return float(np.nan)
    else:
        return sum_product / ((len(x) - lag) * v)


def auto_covariance(x, ignore_nan=True):
    """
    Computes autocovariances.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        numpy array.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return stattools.acovf(x)


def time_reversal_asymmetry_statistic(x, lag=None, ignore_nan=True):
    """
    Computes time reversal asymmetry statistic.
    Parameters:
        x (numpy array, required): array of numbers.
        lag (integer): integer. Default len(x)/2.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    n = len(x)
    if n < 1:
        return 0
    if lag is None:
        lag = int(len(x) / 2)
    if 2 * lag >= n:
        return 0
    else:
        one_lag = _roll(x, -lag)
        two_lag = _roll(x, 2 * -lag)
        return np.mean(
            (two_lag * two_lag * one_lag - one_lag * x * x)[0 : (n - 2 * lag)]
        )


def _roll(a, shift=1):
    """
    Internal method as a helper method.
    Rolls the array 'a' by 'shift' positions.
    'shift' is now optional and defaults to 1.
    """
    # Ensure a is treated as an array for length calculation
    a = np.array(a)
    
    # Handle the case where the input array is empty
    if len(a) == 0:
        return a
        
    idx = shift % len(a)
    return np.concatenate([a[-idx:], a[:-idx]])


def c3(x, lag=None, ignore_nan=True):
    """
    Computes c3 stats.
    Parameters:
        x (numpy array, required): array of numbers.
        lag (integer) : integer. Default len(x)/2
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    n = len(x)
    if n < 1:
        return 0
    if lag is None:
        lag = int(n / 2)
    if 2 * lag >= n:
        return 0
    else:
        return np.mean((_roll(x, 2 * -lag) * _roll(x, -lag) * x)[0 : (n - 2 * lag)])

def ar_coefficient(x, coeff=1, k=None, ignore_nan=True):
    """
    Computes ar coefficient.
    Parameters:
        x (numpy array, required): array of numbers.
        coeff (number): number. Default 1.
        k (number): number. Default len(x)/2.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    return 1.0
#     x = check_single_dimensional_array(x)
#     if ignore_nan:
#         x = remove_nan_single_dimensional_array(x)
#     if len(x) < 1:
#         return float(np.nan)
#     if k is None:
#         k = int(len(x) / 2)
#     calculated_ar_params = {}
#     x_as_list = list(x)
#     calculated_AR = AR(x_as_list)
#     res = {}
#     try:
#         calculated_ar_params[k] = calculated_AR.fit(maxlag=k, solver="mle").params
#     except (LinAlgError, ValueError):
#         calculated_ar_params[k] = [np.nan] * k
#     mod = calculated_ar_params[k]
#     if coeff <= k:
#         try:
#             return mod[coeff]
#         except IndexError:
#             return 0.0
#     else:
#         return float(np.nan)


def symmetry_looking(x, r=0.5, ignore_nan=True):
    """
    Computes if distribution is symmetric.
    Parameters:
        x (numpy array, required): array of numbers.
        r (float): float,r the percentage of the range to compare with. Default 0.5.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        integer.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return 0
    mean_median_difference = np.abs(np.mean(x) - np.median(x))
    max_min_difference = np.max(x) - np.min(x)
    return int(mean_median_difference < (r * max_min_difference))


def agg_auto_correlation_var(x, lag=None, ignore_nan=True):
    """
    Computes aggregated autocorrelation variance.
    Parameters:
        x (numpy array, required): array of numbers.
        lag (integer) : integer. Default len(x)/2.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    # if the time series is longer than the following threshold, we use fft to calculate the acf
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    n = len(x)
    if n < 1:
        return float(np.nan)
    global THRESHOLD_TO_USE_FFT
    var = np.var(x)
    if lag is None:
        lag = int(n / 2)

    if np.abs(var) < 10**-10 or n == 1:
        a = [0] * len(x)
    else:
        a = acf(x, fft=n > THRESHOLD_TO_USE_FFT, nlags=lag)[1:]
    return np.var(a[:lag])


def agg_auto_correlation_median(x, lag=None, ignore_nan=True):
    """
    Computes aggregated autocorrelation median.
    Parameters:
        x (numpy array, required): array of numbers.
        lag (integer) : integer. Default len(x)/2.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    # if the time series is longer than the following threshold, we use fft to calculate the acf
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    n = len(x)
    if len(x) < 1:
        return float(np.nan)
    global THRESHOLD_TO_USE_FFT
    var = np.var(x)
    if lag is None:
        lag = int(n / 2)
    if np.abs(var) < 10**-10 or n == 1:
        a = [0] * len(x)
    else:
        a = acf(x, fft=n > THRESHOLD_TO_USE_FFT, nlags=lag)[1:]
    return np.median(a[:lag])


def agg_auto_correlation_mean(x, lag=None, ignore_nan=True):
    """
    Computes aggregated autocorrelation mean.
    Parameters:
        x (numpy array, required): array of numbers.
        lag (integer) : integer.Default len(x)/2.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    # if the time series is longer than the following threshold, we use fft to calculate the acf
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    n = len(x)
    if n < 1:
        return float(np.nan)
    global THRESHOLD_TO_USE_FFT
    var = np.var(x)
    if lag is None:
        lag = int(n / 2)

    if np.abs(var) < 10**-10 or n == 1:
        a = [0] * len(x)
    else:
        a = acf(x, fft=n > THRESHOLD_TO_USE_FFT, nlags=lag)[1:]
    return np.mean(a[:lag])


def agg_auto_correlation_std(x, lag=None, ignore_nan=True):
    """
    Computes aggregated autocorrelation std.
    Parameters:
        x (numpy array, required): array of numbers.
        lag (integer) : integer. Default len(x)/2.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    # if the time series is longer than the following threshold, we use fft to calculate the acf
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    n = len(x)
    if n < 1:
        return float(np.nan)
    global THRESHOLD_TO_USE_FFT
    var = np.var(x)
    if lag is None:
        lag = int(n / 2)

    if np.abs(var) < 10**-10 or n == 1:
        a = [0] * len(x)
    else:
        a = acf(x, fft=n > THRESHOLD_TO_USE_FFT, nlags=lag)[1:]
    return np.std(a[:lag])


def energy_ratio_by_chunks(x, num_segments=10, segment_focus=1, ignore_nan=True):
    """
    Computes energy ratio by chunks.
    Parameters:
        x (numpy array, required): array of numbers.
        num_segments (integer) : integer. Default 10.
        segment_focus (integer) : integer. Default 1.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    full_series_energy = np.sum(x**2)
    if segment_focus > num_segments:
        return float(np.nan)
    if num_segments <= 0:
        return float(np.nan)
    return (
        np.sum(np.array_split(x, num_segments)[segment_focus] ** 2.0)
        / full_series_energy
    )


def index_mass_quantile(x, q=0.5, ignore_nan=True):
    """
    Computes index mass quantile.
    Parameters:
        x (numpy array, required): array of numbers.
        q (number) : number. Default 0.5.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    abs_x = np.abs(x)
    s = sum(abs_x)
    if s == 0:
        return float(np.nan)
    else:
        mass_centralized = np.cumsum(abs_x) / s
        return (np.argmax(mass_centralized >= q) + 1) / len(x)


def friedrich_coefficients(x, m=2, r=4, coeff=1, ignore_nan=True):
    """
    Computes friedrich coefficients.
    Parameters:
        x (numpy array, required): array of numbers.
        m (integer): integer, the order of polynom to fit for estimating. Default 2.
        r(integer): integer, the number of quartiles to use for averaging. Default 4
        coeff(number): coefficient. Default 1.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1 or coeff < 0:
        return float(np.nan)
    try:
        return _estimate_friedrich_coefficients(x, m, r)[coeff]
    except IndexError:
        return float(np.nan)


def _estimate_friedrich_coefficients(x, m, r):
    """
    Internal method as a helper method.
    """
    if m <= 0:
        raise ValueError(
            "Order of polynomial need to be positive integer, found {}".format(m)
        )

    df = pd.DataFrame({"signal": x[:-1], "delta": np.diff(x)})
    try:
        df["quantiles"] = pd.qcut(df.signal, r)
    except ValueError:
        return [np.nan] * (m + 1)
    quantiles = df.groupby("quantiles")
    result = pd.DataFrame(
        {"x_mean": quantiles.signal.mean(), "y_mean": quantiles.delta.mean()}
    )
    result.dropna(inplace=True)
    try:
        return np.polyfit(result.x_mean, result.y_mean, deg=m)
    except (np.linalg.LinAlgError, ValueError):
        return [np.nan] * (m + 1)


def augmented_dickey_fuller_teststat(x, ignore_nan=True):
    """
    Computes augmented dickey fuller teststat.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    res = float(np.nan)
    try:
        res = adfuller(x)
        return res[0]
    except (LinAlgError, ValueError, MissingDataError) as error:
        return res


def augmented_dickey_fuller_p_value(x, ignore_nan=True):
    """
    Computes pvalue of augmented dickey fuller teststat.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    res = float(np.nan)
    try:
        res = adfuller(x)
        return res[1]
    except (LinAlgError, ValueError, MissingDataError) as error:
        return res


def augmented_dickey_fuller_used_lag(x, ignore_nan=True):
    """
    Computes usedlag of augmented dickey fuller teststat.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    res = float(np.nan)
    try:
        res = adfuller(x)
        return res[2]
    except (LinAlgError, ValueError, MissingDataError) as error:
        return res


def partial_auto_correlation(x, lag=None, ignore_nan=True):
    """
    Computes the value of the partial autocorrelation function at the given lag.
    Parameters:
        x (numpy array, required): array of numbers.
        lag (integer) : integer.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    n = len(x)
    if n < 1:
        return float(np.nan)
    if lag is None:
        lag = int(n / 2)
    max_demanded_lag = lag
    if n <= 1:
        pacf_coeffs = [np.nan] * (max_demanded_lag + 1)
    else:
        if n <= max_demanded_lag:
            max_lag = n - 1
        else:
            max_lag = max_demanded_lag
        pacf_coeffs = list(pacf(x, method="ld", nlags=max_lag))
        pacf_coeffs = pacf_coeffs + [np.nan] * max(0, (max_demanded_lag - max_lag))
    return pacf_coeffs[lag]


def _fft_aggregated(x, type):
    """ """

    def _get_moment(y, moment):
        """ """
        return y.dot(np.arange(len(y)) ** moment) / y.sum()

    def _get_centroid(y):
        """ """
        return _get_moment(y, 1)

    def _get_variance(y):
        """ """
        return _get_moment(y, 2) - _get_centroid(y) ** 2

    def _get_skew(y):
        """ """
        variance = _get_variance(y)
        if variance < 0.5:
            return float(np.nan)
        else:
            return (
                _get_moment(y, 3)
                - 3 * _get_centroid(y) * variance
                - _get_centroid(y) ** 3
            ) / _get_variance(y) ** (1.5)

    def _get_kurtosis(y):
        """ """
        variance = _get_variance(y)
        if variance < 0.5:
            return float(np.nan)
        else:
            return (
                _get_moment(y, 4)
                - 4 * _get_centroid(y) * _get_moment(y, 3)
                + 6 * _get_moment(y, 2) * _get_centroid(y) ** 2
                - 3 * _get_centroid(y)
            ) / _get_variance(y) ** 2

    calculation = dict(
        centroid=_get_centroid,
        variance=_get_variance,
        skew=_get_skew,
        kurtosis=_get_kurtosis,
    )
    fft_abs = np.abs(np.fft.rfft(x))
    return calculation[type](fft_abs)


def fft_aggregated_centroid(x, ignore_nan=True):
    """
    Computes the spectral centroid (mean) of the absolute fourier transform spectrum.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return _fft_aggregated(x, "centroid")


def fft_aggregated_variance(x, ignore_nan=True):
    """
    Computes the spectral variance of the absolute fourier transform spectrum.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return _fft_aggregated(x, "variance")


def fft_aggregated_skew(x, ignore_nan=True):
    """
    Computes the spectral skew of the absolute fourier transform spectrum.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return _fft_aggregated(x, "skew")


def fft_aggregated_kurtosis(x, ignore_nan=True):
    """
    Computes the spectral kurtosis of the absolute fourier transform spectrum.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return _fft_aggregated(x, "kurtosis")


def max_langevin_fixed_point(x, r=4, m=2, ignore_nan=True):
    """
    Computes max langevin fixed point.
    Parameters:
        x (numpy array, required): array of numbers.
        r(integer): integer,the number of quartiles. Default 4.
        m (integer): integer,the order of polynom to fit for estimating. Default 2.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float: the value of this feature
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    coeff = _estimate_friedrich_coefficients(x, m, r)
    try:
        max_fixed_point = np.max(np.real(np.roots(coeff)))
    except (np.linalg.LinAlgError, ValueError):
        return float(np.nan)
    return max_fixed_point


def petrosian_fd(x, ignore_nan=True):
    """
    Computes petrosian fractal dimension.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    n = len(x)
    if n < 1:
        return float(np.nan)
    diff = np.ediff1d(x)
    n_delta = (diff[1:-1] * diff[0:-2] < 0).sum()
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_delta)))


def katz_fd(x, ignore_nan=True):
    """
    Computes Katz Fractal dimension..
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    dists = np.abs(np.ediff1d(x))
    ll = dists.sum()
    ln = np.log10(np.divide(ll, dists.mean()))
    aux_d = x - x[0]
    d = np.max(np.abs(aux_d[1:]))
    return np.divide(ln, np.add(ln, np.log10(np.divide(d, ll))))


def _time_delay_embed(x, order=3, delay=1):
    """
    A univariate time delay embedding function.  Let ts_len as the length of the time series

    Arguments:
        x -- a one-dimentional numpy array represnted a one-dimentional time series, ordered following time
        order --- number of columns going to be generated (M=order)
        delay --- the shift of the time index to align into a same row.

    Returns:
        z -- output of numpy array of (ts_len-d+1) x d (d is the order)

    Example:

        Orginal time series:
            x = array([ 0.,  1.,  2.,  3.,  4., nan,  6.,  7.,  8.,  9.])
            ts_len = 10
            order = 4
            delay = 1
        Output:
            z=array([[ 0.,  1.,  2.,  3.],
                    [ 1.,  2.,  3.,  4.],
                    [ 2.,  3.,  4., nan],
                    [ 3.,  4., nan,  6.],
                    [ 4., nan,  6.,  7.],
                    [nan,  6.,  7.,  8.],
                    [ 6.,  7.,  8.,  9.]])

    """
    n = len(x)
    if order * delay > n:
        raise ValueError("Error: order * delay should be lower than len(x)")
    if delay < 1:
        raise ValueError("Delay has to be at least 1.")
    if order < 2:
        raise ValueError("Order has to be at least 2.")
    y = np.zeros((order, n - (order - 1) * delay))
    for i in range(order):
        y[i] = x[i * delay : i * delay + y.shape[1]]
    return y.T


def _multivariate_time_delay_embed(X, order=3, delay=1):
    """
    A multi-variate time delay embedding function, form a NxM dimentional matrix in form of numpy array.
    Each column is a time series for a specific variable.   Each row are the variables harvested from same time

    Arguments:
        x -- A two-dimentional numpy array (matrix of NxM) represnted a multi-variate time series, ordered following the time delay
            The number of row (N) is the lenght of time series
             The number of columns (M) is the number of variables.
        order --- number of columns going to be generated (M=order)
        delay --- the shift of the column from previous column (say [i, i + N]) is index the previous column
                selection to choose values from a time series
                [i+delay, i+N+delay] is the index for next column
                We always keep delay = 1

    Returns:
        z -- output of numpy array of NxM as discussed, say as N' x M'
        The value of N' is the time series length (N) minus the order (N '= N - order)
        M' is the order multiples the order (M' = M * order)

        Let assume that we have two variables x1 and x2, their time embedding becomes X1 and X2.
        Then, we have:  z =  (X1, X2).  Here is X1 and X2 are side-by-side in parallel.


    Example:

        Orginal time series:
              X = array([[  0., 100.],
                    [  1., 101.],
                    [  2., 102.],
                    [  3., 103.],
                    [  4., 104.],
                    [ nan, 105.],
                    [  6., 106.],
                    [  7.,  nan],
                    [  8., 108.],
                    [  9., 109.]])

        If order = 4, we have:
        Output:
            z = array([[  0.,   1.,   2.,   3., 100., 101., 102., 103.],
                    [  1.,   2.,   3.,   4., 101., 102., 103., 104.],
                    [  2.,   3.,   4.,  nan, 102., 103., 104., 105.],
                    [  3.,   4.,  nan,   6., 103., 104., 105., 106.],
                    [  4.,  nan,   6.,   7., 104., 105., 106.,  nan],
                    [ nan,   6.,   7.,   8., 105., 106.,  nan, 108.],
                    [  6.,   7.,   8.,   9., 106.,  nan, 108., 109.]])
    """

    num_vars = X.shape[1]

    embedded = []
    for i in range(num_vars):
        X_i = X[:, i]
        tmp_X_i = _time_delay_embed(X_i, order, delay)
        embedded.append(tmp_X_i)
    ans = np.hstack(embedded)

    return ans


def _revese_time_delay_embed(x, ts_len, order=1):
    """
    Reverse time delay embedding function, convert a matrix in form of numpy array to a time series.  Output is time series computed from the matrix

    Also see method:  _time_delay_embed()

    Arguments:

        X: a N x M metric (M is the same as the value of order)
        ls_len:  the lenght of output time series
        order: number of columns generated from univariate time series (M=order)

    Output:
        A time series with its values comes from the summary of the metrix element.
        For a time series element with order index as i (i = [0,...,ts_len-1]). It is the average of the
        matrix elements with its row index and column index sum together.

        Using the following example:

            value(i=4) = avg(x(4,0) + x(3,1) + x(2,2) + x(1,3) + x(0,4))

    Example:  we have input x as follows, and ts_len as 10, and order = 4
        x =  array([[0. , 1. , 2. , 3. ],
                    [1. , 2. , 3. , 4. ],
                    [2. , 3. , 4. , 7.3],
                    [3. , 4. , 5.2, 6. ],
                    [4. , 6.5, 6. , 7. ],
                    [5.5, 6. , 7. , 8. ],
                    [6. , 7. , 8. , 9. ]])

        Output:
            array([0.,1.,2.,3.,4.,6.125,6.,7.,8.,9.])
    """

    if ts_len < 1:
        raise ValueError("Time series has to be positive integer.")

    y = np.zeros(ts_len)
    for i in range(ts_len):
        candidate_num = []
        for j in range(order):
            if (i - j) < x.shape[0] and (i - j) >= 0:
                candidate_num.append(x[i - j, j])
        y[i] = np.mean(
            candidate_num
        )  # it can use candidate_num to generate a confidence curve
    return y


def _multivariate_reverse_time_delay_embed(X, ts_len, order):
    """
    Reverse time delay embedding function, convert a matrix in form of numpy array to a time series.
    Output is multi-variate time series computed from the matrix

    The number of variables is the number of columns of matrix divided by the order
        num_vars = int(X.shape[1] / order)

    Also see: _multivariate_time_delay_embed() and
              _reverse_time_delay_embed()

    Arguments:

        x: a N' x M' metric (M' is the same as the value of order multipling the number of columns)
        ls_len:  the lenght of output time series
        order: number of columns generated from univariate time series (M=order)

    Output:

        A multi-variate time series in the form of numpy array

    Example:  we have input x as follows, and ts_len as 10, and order = 4
        x =  array(
        [
            [0.0, 1.0, 2.0, 3.0, 100.0, 101.0, 102.0, 103.0],
            [1.0, 2.0, 3.0, 4.0, 101.0, 102.0, 103.0, 104.0],
            [2.0, 3.0, 4.0, 5.0, 102.0, 103.0, 104.0, 105.0],
            [3.0, 4.0, 5.1, 6.0, 103.0, 104.0, 105.0, 106.0],
            [4.0, 4.9, 6.0, 7.0, 104.0, 105.0, 106.0, 106.0],
            [5.0, 6.0, 7.0, 8.0, 105.0, 106.0, 107.0, 108.0],
            [6.0, 7.0, 8.0, 9.0, 106.0, 108.0, 108.0, 109.0],
        ]
        )

        Output:
            array(
                [[  0. 100.]
                [  1. 101.]
                [  2. 102.]
                [  3. 103.]
                [  4. 104.]
                [  5. 105.]
                [  6. 106.]
                [  7. 107.]
                [  8. 108.]
                [  9. 109.]]
            )
    """

    if order < 2:
        raise ValueError("Order has to be at least 2.")

    if ts_len < 1:
        raise ValueError("Time series has to be positive integer.")

    if X.shape[1] % order != 0:
        raise ValueError("The number of columns has been dividable by order!")

    num_vars = int(X.shape[1] / order)

    reverse_embedded = []
    for i in range(num_vars):
        X_i = X[:, order * i : order * i + order]
        tmp_X_i = _revese_time_delay_embed(X_i, ts_len, order)

        tmp_X_i = tmp_X_i.reshape(-1, 1)

        reverse_embedded.append(tmp_X_i)

    ans = np.hstack(reverse_embedded)

    return ans


def perm_entropy(x, order=3, delay=1, normalize=False, ignore_nan=True):
    """
    Computes permutation Entropy.
    Parameters:
        x (numpy array, required): array of numbers.
        order(integer): integer.
        delay(integer): integer.
        normalize(boolean): boolean.
        ignore_nan (boolean, optional): if True, then NaN are ignored and aggregates.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    ran_order = range(order)
    hashmult = np.power(order, ran_order)
    sorted_idx = _time_delay_embed(x, order=order, delay=delay).argsort(
        kind="quicksort"
    )
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    _, c = np.unique(hashval, return_counts=True)
    p = np.true_divide(c, c.sum())
    pe = -np.multiply(p, np.log2(p)).sum()
    if normalize:
        pe /= np.log2(factorial(order))
    return pe


def fft_spectral_entropy(x, fs=2.0, nperseg=None, normalize=False, ignore_nan=True):
    """
    Computes spectral Entropy.
    Parameters:
        x (numpy array, required): array of numbers.
        fs(float): float.
        nperseg(integer): integer.
        normalize(boolean): boolean.
        ignore_nan (boolean, optional): if True, then NaN are ignored and aggregates.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    # Compute and normalize power spectrum
    _, psd = periodogram(x, fs)
    psd_norm = np.divide(psd, psd.sum())
    se = -np.multiply(psd_norm, np.log2(psd_norm)).sum()
    if normalize:
        se /= np.log2(len(psd_norm))
    return se


def welch_spectral_entropy(x, fs=1.0, nperseg=None, normalize=False, ignore_nan=True):
    """
    Computes welch spectral Entropy.
    Parameters:
        x (numpy array, required): array of numbers.
        fs(float): float.
        nperseg(integer): integer.
        normalize(boolean): boolean.
        ignore_nan (boolean, optional): if True, then NaN are ignored and aggregates.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    # Compute and normalize power spectrum
    _, psd = welch(x, fs=0.1, nperseg=nperseg)
    psd_norm = np.divide(psd, psd.sum())
    se = -np.multiply(psd_norm, np.log2(psd_norm)).sum()
    if normalize:
        se /= np.log2(len(psd_norm))
    return se


def svd_entropy(x, order=3, delay=1, normalize=False, ignore_nan=True):
    """
    Computes Singular Value Decomposition entropy.
    Parameters:
        x (numpy array, required): array of numbers.
        order(integer): integer.
        delay(integer): integer.
        normalize(boolean): boolean.
        ignore_nan (boolean, optional): if True, then NaN are ignored and aggregates.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    mat = _time_delay_embed(x, order=order, delay=delay)
    w = np.linalg.svd(mat, compute_uv=False)
    # Normalize the singular values
    w /= sum(w)
    svd_e = -np.multiply(w, np.log2(w)).sum()
    if normalize:
        svd_e /= np.log2(order)
    return svd_e


def discrete_cosine(x, ignore_nan=True):
    """
    Computes discrete cosine transform
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored and aggregates.
    Return:
        numpy array
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return dct(x)


def normalised_count_above_mean(x, ignore_nan=True):
    """
    Computes normalised count above mean.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored and aggregates.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    m = np.mean(x)
    return np.where(x > m)[0].shape[0] / len(x)


def normalised_count_below_mean(x, ignore_nan=True):
    """
    Computes normalised count below mean.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored and aggregates.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    m = np.mean(x)
    return np.where(x < m)[0].shape[0] / len(x)


def sum_log_values(x, ignore_nan=True):
    """
    Computes sum of natural log
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored and aggregates.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    x = remove_zero_single_dimensional_array(x)
    if len(x) < 1:
        return 0
    return np.sum(np.log(x))


def mode_abs_change(x, ignore_nan=True):
    """
    Computes mode absolute change.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored and aggregates.
    Return:
        numpy array.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return mode(abs(np.diff(x)))[0]


def geometric_mean(x, ignore_nan=True):
    """
    Computes geometric_mean of an array.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored and aggregates.
    Return:
        float.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return gmean(x)


def count_nan(x):
    """
    Computes number of nans in an array.
    Parameters:
        x (numpy array, required): array of numbers.
    Return:
        integer.
    """
    x = check_single_dimensional_array(x)
    return len(x[np.isnan(x)])


def z_score(x, ignore_nan=True):
    """
    Computes number of zscore of an array.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored and aggregates.
    Return:
        numpy array.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return zscore(x)


def wavelet_features(x, ignore_nan=True):
    """
    Computes wavelet features of an array.
    Parameters:
        x (numpy array, required): array of numbers.
        ignore_nan (boolean, optional): if True, then NaN are ignored and aggregates.
    Return:
        numpy array.
    """
    x = check_single_dimensional_array(x)
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    return np.array(get_Haar_Wavelet(x))


def identity(x):
    """
    Return the same array as it is
    Parameters:
        x (numpy array, required): array of integers
    Return:
        numpy array
    """
    return np.array(x)


def calculate_wavelet_features_for_band(list_values):
    """
    compute wavelet features for band.
    Parameters:
        list_values (list, required): list of intergers
    Return:
        list
    """
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1] / len(list_values) for elem in counter_values]
    entropy = scipy.stats.entropy(probabilities)

    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))

    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(
        np.diff(np.array(list_values) > np.nanmean(list_values))
    )[0]
    no_mean_crossings = len(mean_crossing_indices)
    [no_zero_crossings, no_mean_crossings]

    return (
        [entropy]
        + [no_zero_crossings, no_mean_crossings]
        + [n5, n25, n75, n95, median, mean, std, var, rms]
    )


def get_Haar_Wavelet(x, ignore_nan=True):
    """
    Extract Haar wavelet
    """
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)

    list_coeff = wavedec(x, "haar")
    features = []
    for coeff in list_coeff:
        features += calculate_wavelet_features_for_band(coeff)
    return features


def get_symbolicaggregation(
    x, n_bins=5, encode="ordinal", ignore_nan=True, return_trained_bins=False
):
    """
    Symbolic Aggregation
    """
    if ignore_nan:
        x = remove_nan_single_dimensional_array(x)
    if len(x) < 1:
        return float(np.nan)
    from sklearn.preprocessing import KBinsDiscretizer
    import itertools

    tmpMdl = KBinsDiscretizer(n_bins=n_bins, encode=encode)
    if len(x.shape) == 1 or x.shape[1] == 1:
        x = x.reshape(-1, 1)
        if return_trained_bins:
            return list(itertools.chain(*tmpMdl.fit(x).transform(x).tolist())), tmpMdl
        else:
            return list(itertools.chain(*tmpMdl.fit(x).transform(x).tolist()))
    else:
        raise Exception("Not Implemented Error")
    

import inspect as _inspect
import math as _math
import sys as _sys
 
# known stubs / non-scalar-features to exclude even though they probe as scalar
_REGISTRY_BLACKLIST = {"ar_coefficient"}
 
# fixed-arg variants of param'd features -> (function_name, kwargs). Names avoid the core-111.
_VARIANTS = {
    "fe_quantile_10": ("quantile", {"q": 0.10}),
    "fe_quantile_90": ("quantile", {"q": 0.90}),
    "fe_c3_lag1": ("c3", {"lag": 1}),
    "fe_c3_lag2": ("c3", {"lag": 2}),
    "fe_ratio_beyond_2sigma": ("ratio_beyond_r_sigma", {"r": 2.0}),
    "fe_ratio_beyond_3sigma": ("ratio_beyond_r_sigma", {"r": 3.0}),
    "fe_approx_entropy": ("approximate_entropy", {"m": 2, "r": 0.2}),
    "fe_energy_ratio_chunk1": ("energy_ratio_by_chunks", {"num_segments": 10, "segment_focus": 1}),
}
 
 
def _probe_signal():
    rng = np.random.default_rng(0)
    n = 256
    return (
        np.sin(2 * np.pi * np.arange(n) / 24) + 0.01 * np.arange(n)
        + 0.05 * rng.standard_normal(n)
    ).astype(float)
 
 
def _finite_scalar(v):
    try:
        a = np.asarray(v)
        if a.ndim == 0 or a.size == 1:
            return _math.isfinite(float(a.reshape(-1)[0]))
    except Exception:
        pass
    return False
 
 
def _wrap_scalar(fn, kwargs=None):
    """1D window -> float; nan/exception-safe (returns 0.0 on failure)."""
    kwargs = kwargs or {}
 
    def _g(w):
        try:
            v = fn(np.asarray(w, dtype=float), **kwargs)
            v = float(np.asarray(v).reshape(-1)[0])
            return v if _math.isfinite(v) else 0.0
        except Exception:
            return 0.0
 
    return _g
 
 
def _first_sentence(fn):
    doc = (fn.__doc__ or "").strip()
    if not doc:
        return f"Feature '{fn.__name__}'."
    line = doc.splitlines()[0].strip()
    return line.split(". ")[0].rstrip(".") + "."
 
 
def build_extractors():
    """Discover scalar features in this module -> (REGISTRY, DESCRIPTIONS)."""
    mod = _sys.modules[__name__]
    x = _probe_signal()
    reg, desc = {}, {}
    for name, fn in _inspect.getmembers(mod, _inspect.isfunction):
        if name.startswith("_") or name in _REGISTRY_BLACKLIST:
            continue
        if getattr(fn, "__module__", None) != __name__:
            continue
        params = list(_inspect.signature(fn).parameters.values())
        if not params:
            continue
        extra_required = [
            p for p in params[1:]
            if p.default is _inspect._empty
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)
        ]
        if extra_required:
            continue                                   # needs params -> use _VARIANTS instead
        try:
            v = fn(x)
        except Exception:
            continue
        if _finite_scalar(v):
            reg[name] = _wrap_scalar(fn)
            desc[name] = _first_sentence(fn)
    for vname, (fnname, kwargs) in _VARIANTS.items():
        fn = getattr(mod, fnname, None)
        if fn is None:
            continue
        reg[vname] = _wrap_scalar(fn, kwargs)
        desc[vname] = (
            _first_sentence(fn)
            + " (" + ", ".join(f"{k}={v}" for k, v in kwargs.items()) + ")"
        )
    return reg, desc
 
 
# built once at import — the consolidated registry other modules extend EXTRACTORS with.
REGISTRY, DESCRIPTIONS = build_extractors()
