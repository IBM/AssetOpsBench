"""Metrics for the skills evaluation benchmark."""

from __future__ import annotations

import random
from typing import Sequence


def compute_pass_rate(results: Sequence[bool]) -> float:
    """Fraction of ``True`` values in *results*."""
    if not results:
        return 0.0
    return sum(results) / len(results)


def compute_normalized_gain(with_skills: float, without_skills: float) -> float:
    """Normalized gain *g = (with - without) / (1 - without)*.

    Returns 0.0 when *without_skills* is already 1.0 (perfect baseline).
    """
    if without_skills >= 1.0:
        return 0.0
    return (with_skills - without_skills) / (1.0 - without_skills)


def bootstrap_ci(
    values: Sequence[float],
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute a bootstrapped confidence interval for the mean of *values*.

    Args:
        values: Observed values.
        n_bootstrap: Number of bootstrap re-samples.
        ci: Confidence level (e.g. 0.95 for a 95 % CI).
        seed: RNG seed for reproducibility.

    Returns:
        ``(lower, upper)`` bounds of the CI.
    """
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    vals = list(values)
    for _ in range(n_bootstrap):
        sample = [rng.choice(vals) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1.0 - ci) / 2.0
    lo = means[int(alpha * len(means))]
    hi = means[int((1.0 - alpha) * len(means)) - 1]
    return (lo, hi)
