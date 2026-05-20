"""IEC 60599:2022 + JSON-safe divergent-ratio regression tests for the FMSR port.

These tests target only the gas-ratio classification surface (`analyze_dga`),
which has no dependency on `$SG_DATA_DIR` CSV fixtures. Tests that exercise
`list_failure_modes`, `search_failure_modes`, `get_sensor_correlation`, and
`get_dga_record` are intentionally not ported here because the processed-CSV
data port is deferred (see aob-extraction_deferred.md D2) — they will be
added when SG_DATA_DIR is populated.

DGA contract assumptions (mirrored from the SmartGridBench source project):
  - analyze_dga is fully deterministic.
  - IEC code "N" / "Normal / Inconclusive" is a valid output, not an error.
  - Rogers table follows IEC 60599:2022 Table 1 strictly. D2 ("Discharges of
    high energy / arcing") requires R2 ∈ [0.6, 2.5) AND R3 ≥ 2.0 AND
    R1 ∈ [0.1, 1.0). Samples with R2 ≥ 2.5 fall outside D2 and (if
    R1 ∈ [0.1, 0.5) and R2 ≥ 1.0 and R3 ≥ 1.0) classify as D1 instead.
  - Divergent ratios (zero denominator with nonzero numerator) are reported as
    `null` + sibling `r{1,2,3}_divergent: true`, never as a non-finite float.
"""

from __future__ import annotations

import json as _json

from servers.smart_grid.fmsr import main as fmsr
from servers.smart_grid.fmsr.main import analyze_dga

# T-018 representative profile from the source project's processed DGA records:
# R1 = 6/35 = 0.17, R2 = 482/26 = 18.5, R3 = 26/3 = 8.67.
_T018_GASES = dict(h2=35.0, ch4=6.0, c2h2=482.0, c2h4=26.0, c2h6=3.0)


def test_analyze_dga_returns_required_fields():
    result = analyze_dga(**_T018_GASES, transformer_id="T-018")
    for key in (
        "iec_code",
        "diagnosis",
        "r1_ch4_h2",
        "r2_c2h2_c2h4",
        "r3_c2h4_c2h6",
        "input_gases",
    ):
        assert key in result, f"Missing field: {key}"
    assert result["transformer_id"] == "T-018"


def test_analyze_dga_echoes_inputs():
    result = analyze_dga(**_T018_GASES)
    gases = result["input_gases"]
    assert gases["h2_ppm"] == _T018_GASES["h2"]
    assert gases["c2h2_ppm"] == _T018_GASES["c2h2"]


def test_analyze_dga_deterministic():
    r1 = analyze_dga(**_T018_GASES)
    r2 = analyze_dga(**_T018_GASES)
    assert r1["iec_code"] == r2["iec_code"]
    assert r1["diagnosis"] == r2["diagnosis"]


def test_analyze_dga_high_c2h2_ratio_is_d1_per_iec_strict():
    # T-018 profile (R1=0.17, R2=18.5, R3=8.67) classifies as D1 under
    # IEC 60599:2022 Table 1: R2=18.5 falls outside D2's [0.6, 2.5) cap.
    result = analyze_dga(**_T018_GASES)
    assert result["iec_code"] == "D1"


def test_analyze_dga_all_zeros_no_crash():
    result = analyze_dga(h2=0, ch4=0, c2h2=0, c2h4=0, c2h6=0)
    assert "iec_code" in result
    assert result["iec_code"] == "N"


def test_analyze_dga_zero_c2h6_diverges_r3():
    # Regression: zero denominator must produce a divergent ratio internally
    # (so classification is correct), but the public output normalizes inf →
    # null + r3_divergent: True for JSON safety.
    result = analyze_dga(h2=500, ch4=200, c2h2=120, c2h4=100, c2h6=0)
    assert result["iec_code"] == "D2"
    assert result["r3_c2h4_c2h6"] is None
    assert result.get("r3_divergent") is True
    _json.dumps(result, allow_nan=False)


def test_analyze_dga_zero_c2h4_diverges_r2():
    # c2h4=0, c2h2>0 → R2 diverges; R3 collapses to 0.0 (c2h4=0, c2h6>0).
    # No fault row matches → N. Public output: r2_c2h2_c2h4 → null + flag.
    result = analyze_dga(h2=500, ch4=200, c2h2=120, c2h4=0, c2h6=30)
    assert result["iec_code"] == "N"
    assert result["r2_c2h2_c2h4"] is None
    assert result.get("r2_divergent") is True
    _json.dumps(result, allow_nan=False)


def test_analyze_dga_zero_h2_diverges_r1():
    # h2=0, ch4>0 → R1 diverges. R2=0.025, R3=0.667 → T1.
    result = analyze_dga(h2=0, ch4=200, c2h2=2, c2h4=80, c2h6=120)
    assert result["iec_code"] == "T1"
    assert result["r1_ch4_h2"] is None
    assert result.get("r1_divergent") is True
    _json.dumps(result, allow_nan=False)


def test_analyze_dga_finite_ratios_have_no_divergent_flags():
    # Non-regression: finite-ratio results must NOT carry r{1,2,3}_divergent
    # keys at all.
    result = analyze_dga(**_T018_GASES)
    assert "r1_divergent" not in result
    assert "r2_divergent" not in result
    assert "r3_divergent" not in result
    _json.dumps(result, allow_nan=False)


def test_analyze_dga_without_transformer_id():
    result = analyze_dga(**_T018_GASES)
    assert "transformer_id" not in result


def test_analyze_dga_negative_input_rejected():
    result = analyze_dga(h2=-1, ch4=200, c2h2=2, c2h4=80, c2h6=120)
    assert "error" in result
    assert "invalid_inputs" in result


def test_analyze_dga_string_inputs_coerced():
    # LLM tool clients sometimes stringify numeric args.
    result = analyze_dga(h2="35", ch4="6", c2h2="482", c2h4="26", c2h6="3")
    assert result["iec_code"] == "D1"


def test_get_dga_record_sample_date_is_json_safe(tmp_path, monkeypatch):
    (tmp_path / "dga_records.csv").write_text(
        "\n".join(
            [
                "transformer_id,sample_date,dissolved_h2_ppm,dissolved_ch4_ppm,"
                "dissolved_c2h2_ppm,dissolved_c2h4_ppm,dissolved_c2h6_ppm,"
                "dissolved_co_ppm,dissolved_co2_ppm,fault_label,source_dataset",
                "T-001,2026-01-02,10,20,1,30,40,100,200,T1,unit-test",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SG_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(fmsr, "_dga_records", None)

    result = fmsr.get_dga_record("T-001")

    assert result["sample_date"] == "2026-01-02"
    _json.dumps(result, allow_nan=False)
