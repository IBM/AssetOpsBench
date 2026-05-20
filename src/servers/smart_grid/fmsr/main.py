"""FMSR MCP server — Failure Mode to Sensor Relation mapping for Smart Grid transformers.

FMSR = Failure Mode Sensor Relation. Given sensor readings (especially dissolved
gas concentrations), this server helps an agent diagnose which fault type is most
likely and understand which sensors are elevated for each failure mode.

Tools exposed to the LLM agent:
  list_failure_modes     — catalogue of all known fault types
  search_failure_modes   — find fault types matching a keyword or gas name
  get_sensor_correlation — which gases/sensors indicate a specific fault
  get_dga_record         — retrieve a transformer's most recent DGA snapshot
  analyze_dga            — classify a set of gas concentrations into a fault type
                           using the IEC 60599 Rogers Ratio method

Data source: ``$SG_DATA_DIR/failure_modes.csv``, ``dga_records.csv``.
"""

from __future__ import annotations

import math

import pandas as pd
from mcp.server.fastmcp import FastMCP

from servers.smart_grid.base import (
    json_safe_record,
    load_dga_records,
    load_failure_modes,
)

mcp = FastMCP("smart-grid-fmsr")

_failure_modes: pd.DataFrame | None = None
_dga_records: pd.DataFrame | None = None


def _get_failure_modes() -> pd.DataFrame:
    global _failure_modes
    if _failure_modes is None:
        _failure_modes = load_failure_modes()
    return _failure_modes


def _get_dga_records() -> pd.DataFrame:
    global _dga_records
    if _dga_records is None:
        _dga_records = load_dga_records()
    return _dga_records


# ---------------------------------------------------------------------------
# Rogers Ratio method (IEC 60599:2022 Table 1)
# ---------------------------------------------------------------------------
# A classic DGA interpretation algorithm. It computes three gas ratios and
# maps them to a fault code via a lookup table.
#
# Ratios (using the JSON's R-numbering convention):
#   R1 = CH4  / H2     (IEC Table 1 middle column)
#   R2 = C2H2 / C2H4   (IEC Table 1 first column)
#   R3 = C2H4 / C2H6   (IEC Table 1 third column)
#
# The lookup table below follows IEC 60599:2022 Table 1 (4th edition, p.13).
# "NS" in the standard ("non-significant whatever the value") is encoded as
# (0, None) on R2 since gas ratios are non-negative.
#
# Order: most-severe first so first-match-wins resolves overlap toward the
# more severe code (e.g., D2 wins over D1 in the overlap region
# R1 ∈ [0.1, 0.5), R2 ∈ [1.0, 2.5), R3 ≥ 2.0). Boundary phrasing here mirrors
# the encoded ranges (min-inclusive, max-exclusive).

_ROGERS_TABLE = [
    # (R1_range, R2_range, R3_range, code, description)
    # Each range is (min_inclusive, max_exclusive); None = no bound.
    ((0.1, 1.0), (0.6, 2.5), (2.0, None), "D2", "Discharges of high energy (arcing)"),
    ((0.1, 0.5), (1.0, None), (1.0, None), "D1", "Discharges of low energy"),
    ((1.0, None), (0, 0.2), (4.0, None), "T3", "Thermal fault, t > 700 °C"),
    ((1.0, None), (0, 0.1), (1.0, 4.0), "T2", "Thermal fault, 300 °C < t < 700 °C"),
    ((1.0, None), (0, None), (0, 1.0), "T1", "Thermal fault, t < 300 °C"),
    ((0, 0.1), (0, None), (0, 0.2), "PD", "Partial discharges"),
]


def _in_range(value: float, lo, hi) -> bool:
    if lo is not None and value < lo:
        return False
    if hi is not None and value >= hi:
        return False
    return True


def _ratio(numerator: float, denominator: float) -> float:
    """Compute a gas ratio with explicit zero-denominator handling.

    - denominator > 0:                   numerator / denominator (finite)
    - denominator == 0 and numerator > 0: math.inf (a real ratio that diverges)
    - denominator == 0 and numerator == 0: 0.0 (genuinely no signal)

    Returning math.inf for a divergent ratio is critical: collapsing it to 0.0
    would silently drop samples into the wrong fault class (e.g., a sample with
    nonzero CH4 and C2H4 but zero C2H6 has R3 = +inf and should match D2 if the
    other ratios fit, not fall through to N).
    """
    if denominator > 0:
        return numerator / denominator
    return math.inf if numerator > 0 else 0.0


def _ratio_field(value: float) -> tuple:
    """Normalize a ratio for JSON-safe outbound serialization.

    Returns (json_safe_value, is_divergent). math.inf becomes None + True,
    so `json.dumps(result, allow_nan=False)` succeeds on the public output.
    Internal table matching keeps the raw float (including math.inf) — this
    helper runs only at the dict-construction boundary.
    """
    if math.isinf(value):
        return None, True
    return round(value, 4), False


def _rogers_ratio(h2: float, ch4: float, c2h2: float, c2h4: float, c2h6: float) -> dict:
    """Apply Rogers Ratio method; return IEC code and description.

    Output ratio fields are JSON-safe: a divergent ratio (zero denominator,
    nonzero numerator) is reported as `null` with a sibling `r{1,2,3}_divergent: true`
    flag rather than `inf`. Internal table matching uses the true infinity so
    classification is correct.
    """
    r1 = _ratio(ch4, h2)
    r2 = _ratio(c2h2, c2h4)
    r3 = _ratio(c2h4, c2h6)

    # All-zero gases → N. IEC 60599 Table 1 does not address the no-detectable-gas
    # case explicitly; PD's R1/R3 ranges include 0 and would otherwise spuriously
    # match. Operationally, no measurable gas means no fault, not partial discharge.
    if h2 == 0 and ch4 == 0 and c2h2 == 0 and c2h4 == 0 and c2h6 == 0:
        return {
            "iec_code": "N",
            "diagnosis": "Normal / Inconclusive",
            "r1_ch4_h2": 0.0,
            "r2_c2h2_c2h4": 0.0,
            "r3_c2h4_c2h6": 0.0,
        }

    for r1_range, r2_range, r3_range, code, description in _ROGERS_TABLE:
        if (
            _in_range(r1, *r1_range)
            and _in_range(r2, *r2_range)
            and _in_range(r3, *r3_range)
        ):
            return _build_result(code, description, r1, r2, r3)

    return _build_result("N", "Normal / Inconclusive", r1, r2, r3)


def _build_result(code: str, description: str, r1: float, r2: float, r3: float) -> dict:
    """Build the public analyze_dga result dict with JSON-safe ratio fields."""
    r1_val, r1_div = _ratio_field(r1)
    r2_val, r2_div = _ratio_field(r2)
    r3_val, r3_div = _ratio_field(r3)
    result = {
        "iec_code": code,
        "diagnosis": description,
        "r1_ch4_h2": r1_val,
        "r2_c2h2_c2h4": r2_val,
        "r3_c2h4_c2h6": r3_val,
    }
    if r1_div:
        result["r1_divergent"] = True
    if r2_div:
        result["r2_divergent"] = True
    if r3_div:
        result["r3_divergent"] = True
    return result


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def list_failure_modes() -> list[dict]:
    """Return the full catalogue of known transformer failure modes.

    Returns:
        List of dicts with keys: failure_mode_id, name, severity, iec_code,
        key_gases, recommended_action.
    """
    df = _get_failure_modes()
    return df[
        [
            "failure_mode_id",
            "name",
            "severity",
            "iec_code",
            "key_gases",
            "recommended_action",
        ]
    ].to_dict(orient="records")


@mcp.tool()
def search_failure_modes(query: str) -> list[dict]:
    """Search failure modes by keyword (name, description, gas, or IEC code).

    Args:
        query: Free-text search string, e.g. "arc", "H2", "PD", "overheating".

    Returns:
        List of matching failure mode dicts (same schema as list_failure_modes).
        Empty list if no matches.
    """
    df = _get_failure_modes()
    q = query.lower()
    mask = (
        df["name"].str.lower().str.contains(q, na=False)
        | df["description"].str.lower().str.contains(q, na=False)
        | df["key_gases"].str.lower().str.contains(q, na=False)
        | df["iec_code"].str.lower().str.contains(q, na=False)
        | df["dga_label"].str.lower().str.contains(q, na=False)
    )
    return df[mask][
        [
            "failure_mode_id",
            "name",
            "severity",
            "iec_code",
            "key_gases",
            "recommended_action",
        ]
    ].to_dict(orient="records")


@mcp.tool()
def get_sensor_correlation(failure_mode_id: str) -> dict:
    """Return the sensors and gases most strongly associated with a failure mode.

    Args:
        failure_mode_id: e.g. "FM-006" (use list_failure_modes to find IDs).

    Returns:
        Dict with keys: failure_mode_id, name, key_gases (list), description,
        iec_code, recommended_action.
        Returns an error dict if the ID is not found.
    """
    df = _get_failure_modes()
    row = df[df["failure_mode_id"] == failure_mode_id]
    if row.empty:
        return {"error": f"Failure mode '{failure_mode_id}' not found."}
    r = row.iloc[0].to_dict()
    r["key_gases"] = [g.strip() for g in r["key_gases"].split(",") if g.strip()]
    return r


@mcp.tool()
def get_dga_record(transformer_id: str) -> dict:
    """Retrieve the most recent dissolved gas analysis (DGA) record for a transformer.

    Args:
        transformer_id: Asset identifier, e.g. "T-016".

    Returns:
        Dict with gas concentrations (ppm) and the recorded fault label:
        transformer_id, sample_date, dissolved_h2_ppm, dissolved_ch4_ppm,
        dissolved_c2h2_ppm, dissolved_c2h4_ppm, dissolved_c2h6_ppm,
        dissolved_co_ppm, dissolved_co2_ppm, fault_label.
        Returns an error dict if not found.
    """
    df = _get_dga_records()
    row = (
        df[df["transformer_id"] == transformer_id]
        # sample_date is stored as ISO YYYY-MM-DD, so lexicographic descending
        # order is chronological.
        .sort_values("sample_date", ascending=False)
    )
    if row.empty:
        return {"error": f"No DGA record found for '{transformer_id}'."}
    record = row.iloc[0].to_dict()
    return json_safe_record(record)


@mcp.tool()
def analyze_dga(
    h2: float,
    ch4: float,
    c2h2: float,
    c2h4: float,
    c2h6: float,
    transformer_id: str | None = None,
) -> dict:
    """Classify a set of dissolved gas concentrations into a fault type using
    the IEC 60599 Rogers Ratio method.

    Given raw gas readings (in ppm), returns the most likely fault classification
    and the three diagnostic ratios.

    Args:
        h2:   Hydrogen concentration (ppm).
        ch4:  Methane concentration (ppm).
        c2h2: Acetylene concentration (ppm).
        c2h4: Ethylene concentration (ppm).
        c2h6: Ethane concentration (ppm).
        transformer_id: Optional — if provided, included in the result for
                        traceability.

    Returns:
        Dict with keys:
          transformer_id (if provided), iec_code, diagnosis,
          r1_ch4_h2, r2_c2h2_c2h4, r3_c2h4_c2h6,
          input_gases (echo of inputs).

        Ratio fields are always JSON-safe: a divergent ratio (zero
        denominator with nonzero numerator) is reported as `null` plus a
        sibling `r{1,2,3}_divergent: true` flag, never as a non-finite float.
        Finite-ratio results omit the `*_divergent` keys entirely.
    """
    # Coerce to float: LLMs sometimes pass numeric args as strings even when
    # the tool schema declares "type": "number".
    try:
        h2, ch4, c2h2, c2h4, c2h6 = (
            float(h2),
            float(ch4),
            float(c2h2),
            float(c2h4),
            float(c2h6),
        )
    except (TypeError, ValueError) as exc:
        return {"error": f"Gas values must be numeric: {exc}"}
    inputs = {
        "h2_ppm": h2,
        "ch4_ppm": ch4,
        "c2h2_ppm": c2h2,
        "c2h4_ppm": c2h4,
        "c2h6_ppm": c2h6,
    }
    negative_inputs = {name: value for name, value in inputs.items() if value < 0}
    if negative_inputs:
        return {
            "error": "Gas concentrations must be non-negative.",
            "invalid_inputs": negative_inputs,
        }

    invalid_number_inputs = {
        name: value for name, value in inputs.items() if not math.isfinite(value)
    }
    if invalid_number_inputs:
        return {
            "error": "Gas concentrations must be finite numbers.",
            "invalid_inputs": invalid_number_inputs,
        }

    result = _rogers_ratio(h2, ch4, c2h2, c2h4, c2h6)
    result["input_gases"] = inputs
    if transformer_id:
        result["transformer_id"] = transformer_id
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point — runs the FastMCP stdio JSON-RPC loop."""
    mcp.run()


if __name__ == "__main__":
    main()
