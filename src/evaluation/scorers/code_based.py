"""Code-Based scorers — deterministic, no LLM, no network.

Implements deterministic string and numeric matching for evaluation.
Call install() to register these scorers in the global registry.
"""

from __future__ import annotations

import re
from typing import Any

from ..models import Scenario, ScorerResult
from . import register


def exact_string_match(
    scenario: Scenario, answer: str, trajectory_text: str
) -> ScorerResult:
    """Exact string match scorer with normalization.
    
    Normalization rules:
    - Strip leading/trailing whitespace
    - Case-insensitive comparison
    - Remove extra whitespace within the string
    
    Args:
        scenario: Scenario with expected_answer field
        answer: The agent's answer to score
        trajectory_text: The full trajectory (unused in this scorer)
    
    Returns:
        ScorerResult with passed=True if normalized strings match exactly
    """
    if scenario.expected_answer is None:
        return ScorerResult(
            scorer="exact_string_match",
            passed=False,
            score=0.0,
            rationale="scenario.expected_answer is None",
        )
    
    # Normalize both strings
    def normalize(s: str) -> str:
        # Strip leading/trailing whitespace
        s = s.strip()
        # Collapse multiple whitespace to single space
        s = re.sub(r'\s+', ' ', s)
        # Case-insensitive comparison
        return s.lower()
    
    expected_normalized = normalize(scenario.expected_answer)
    answer_normalized = normalize(answer)
    
    passed = expected_normalized == answer_normalized
    score = 1.0 if passed else 0.0
    
    rationale = (
        f"Expected: {scenario.expected_answer!r}, Got: {answer!r}"
        if not passed
        else "Exact match after normalization"
    )
    
    return ScorerResult(
        scorer="exact_string_match",
        passed=passed,
        score=score,
        rationale=rationale,
        details={
            "expected_normalized": expected_normalized,
            "answer_normalized": answer_normalized,
        },
    )


def numeric_match(
    scenario: Scenario, answer: str, trajectory_text: str
) -> ScorerResult:
    """Numeric match scorer with tolerance support.
    
    Parses numeric values from both expected_answer and answer, then
    compares them within optional tolerance.
    
    Tolerance can be specified via scenario.tolerance field:
    - If a single number: absolute tolerance
    - If a dict with 'relative' and/or 'absolute' keys: combined tolerance
    
    Default tolerance: relative=0.01 (1%) and absolute=1e-9
    
    Args:
        scenario: Scenario with expected_answer and optional tolerance
        answer: The agent's answer to score
        trajectory_text: The full trajectory (unused in this scorer)
    
    Returns:
        ScorerResult with passed=True if numbers match within tolerance
    """
    if scenario.expected_answer is None:
        return ScorerResult(
            scorer="numeric_match",
            passed=False,
            score=0.0,
            rationale="scenario.expected_answer is None",
        )
    
    # Parse expected value
    try:
        expected = _parse_number(scenario.expected_answer)
    except (ValueError, TypeError) as e:
        return ScorerResult(
            scorer="numeric_match",
            passed=False,
            score=0.0,
            rationale=f"Failed to parse expected_answer as number: {e}",
        )
    
    # Parse answer value
    try:
        actual = _parse_number(answer)
    except (ValueError, TypeError) as e:
        return ScorerResult(
            scorer="numeric_match",
            passed=False,
            score=0.0,
            rationale=f"Failed to parse answer as number: {e}",
        )
    
    # Get tolerance from scenario or use defaults
    tolerance = _extract_tolerance(scenario)
    
    # Check if match within tolerance
    passed = _within_tolerance(expected, actual, tolerance)
    score = 1.0 if passed else 0.0
    
    rationale = (
        f"Expected: {expected}, Got: {actual}, Tolerance: {tolerance}"
        if not passed
        else f"Numeric match within tolerance: {expected} ≈ {actual}"
    )
    
    return ScorerResult(
        scorer="numeric_match",
        passed=passed,
        score=score,
        rationale=rationale,
        details={
            "expected": expected,
            "actual": actual,
            "tolerance": tolerance,
            "absolute_diff": abs(expected - actual),
        },
    )


def _parse_number(value: str | float | int) -> float:
    """Parse a numeric value from string or number type.
    
    Handles:
    - Direct numeric types (int, float)
    - String representations of numbers
    - Numbers with units (e.g., "42.5 kg" -> 42.5)
    - Scientific notation
    
    Args:
        value: The value to parse
        
    Returns:
        Parsed float value
        
    Raises:
        ValueError: If value cannot be parsed as a number
    """
    if isinstance(value, (int, float)):
        return float(value)
    
    if not isinstance(value, str):
        raise TypeError(f"Expected str or numeric type, got {type(value)}")
    
    # Try to extract the first number from the string
    # This handles cases like "42.5 kg" or "Temperature: 23.7°C"
    match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', value)
    if not match:
        raise ValueError(f"No numeric value found in: {value!r}")
    
    return float(match.group())


def _extract_tolerance(scenario: Scenario) -> dict[str, float]:
    """Extract tolerance configuration from scenario.
    
    Args:
        scenario: Scenario potentially containing tolerance field
        
    Returns:
        Dict with 'relative' and 'absolute' tolerance values
    """
    # Since Scenario uses extra='allow', tolerance is accessible as an attribute
    tolerance = getattr(scenario, "tolerance", {})
    
    # Default tolerance
    result = {"relative": 0.01, "absolute": 1e-9}
    
    if isinstance(tolerance, (int, float)):
        # Single number means absolute tolerance
        result["absolute"] = float(tolerance)
    elif isinstance(tolerance, dict):
        if "relative" in tolerance:
            result["relative"] = float(tolerance["relative"])
        if "absolute" in tolerance:
            result["absolute"] = float(tolerance["absolute"])
    
    return result


def _within_tolerance(
    expected: float, actual: float, tolerance: dict[str, float]
) -> bool:
    """Check if actual value is within tolerance of expected.
    
    Combined tolerance check:
    - Passes if |expected - actual| <= absolute_tolerance
    - OR if |expected - actual| / max(|expected|, |actual|) <= relative_tolerance
    
    Args:
        expected: The expected value
        actual: The actual value
        tolerance: Dict with 'relative' and 'absolute' tolerance values
        
    Returns:
        True if within tolerance, False otherwise
    """
    abs_diff = abs(expected - actual)
    
    # Check absolute tolerance
    if abs_diff <= tolerance["absolute"]:
        return True
    
    # Check relative tolerance (avoid division by zero)
    max_abs = max(abs(expected), abs(actual))
    if max_abs > 0:
        relative_diff = abs_diff / max_abs
        if relative_diff <= tolerance["relative"]:
            return True
    
    return False


def install() -> None:
    """Register code-based scorers.
    
    This function registers both exact_string_match and numeric_match
    scorers in the global scorer registry.
    """
    register("exact_string_match", exact_string_match)
    register("numeric_match", numeric_match)
