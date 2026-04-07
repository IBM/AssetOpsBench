"""Tests for evaluation.metrics."""

from __future__ import annotations


from evaluation.metrics import bootstrap_ci, compute_normalized_gain, compute_pass_rate


def test_pass_rate_all_pass():
    assert compute_pass_rate([True, True, True]) == 1.0


def test_pass_rate_none_pass():
    assert compute_pass_rate([False, False]) == 0.0


def test_pass_rate_mixed():
    assert compute_pass_rate([True, False, True, False]) == 0.5


def test_pass_rate_empty():
    assert compute_pass_rate([]) == 0.0


def test_normalized_gain_positive():
    # 0.8 with skills vs 0.5 without → g = 0.3/0.5 = 0.6
    g = compute_normalized_gain(0.8, 0.5)
    assert abs(g - 0.6) < 1e-9


def test_normalized_gain_perfect_baseline():
    # If baseline is already 1.0, gain is 0.
    assert compute_normalized_gain(1.0, 1.0) == 0.0


def test_normalized_gain_no_improvement():
    assert compute_normalized_gain(0.5, 0.5) == 0.0


def test_bootstrap_ci_deterministic():
    vals = [1.0, 0.0, 1.0, 0.0, 1.0]
    lo, hi = bootstrap_ci(vals, n_bootstrap=1000, ci=0.95, seed=42)
    assert 0.0 <= lo <= hi <= 1.0
    # Mean is 0.6 — CI should straddle it
    assert lo < 0.6 < hi or lo == hi == 0.6


def test_bootstrap_ci_empty():
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_bootstrap_ci_single_value():
    lo, hi = bootstrap_ci([0.7], n_bootstrap=100)
    assert lo == hi == 0.7
