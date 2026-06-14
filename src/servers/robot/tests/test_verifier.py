"""Unit tests for MultiReadingVerifier — no CouchDB required."""

import pytest
from servers.robot.verifier import MultiReadingVerifier, VerifierResult


@pytest.fixture
def verifier():
    return MultiReadingVerifier()


class TestBlockedGate:
    def test_blocked_n_1(self, verifier):
        result = verifier.verify(
            readings=[50.0],
            iot_value=50.0,
            gauge_range=[0, 100],
        )
        assert result.status == "BLOCKED"
        assert result.fm_flag == "FM-7b"
        assert result.score == 0.0

    def test_blocked_n_2(self, verifier):
        result = verifier.verify(
            readings=[48.0, 52.0],
            iot_value=50.0,
            gauge_range=[0, 100],
        )
        assert result.status == "BLOCKED"
        assert result.fm_flag == "FM-7b"

    def test_fm7b_flag_in_blocked(self, verifier):
        result = verifier.verify(readings=[], iot_value=0.0, gauge_range=[0, 100])
        assert result.fm_flag == "FM-7b"
        # insufficient readings flag appears in fm_annotations
        assert any("FM-7b" in ann for ann in result.fm_annotations)


class TestFreezeGate:
    def test_zero_variance_triggers_panel_recheck(self, verifier):
        # Three identical readings → std=0.0 → PANEL_RECHECK
        result = verifier.verify(
            readings=[50.0, 50.0, 50.0],
            iot_value=50.0,
            gauge_range=[0, 100],
        )
        assert result.status == "PANEL_RECHECK"
        assert result.fm_flag == "FM-1"

    def test_tiny_variance_still_freezes(self, verifier):
        # std = 0.0001 < 0.001 * 100 = 0.1 → PANEL_RECHECK
        result = verifier.verify(
            readings=[50.0, 50.0001, 50.0002],
            iot_value=50.0,
            gauge_range=[0, 100],
        )
        assert result.status == "PANEL_RECHECK"


class TestCommit:
    def test_commit_consistent_readings(self, verifier):
        # Tight cluster near IoT value → high C, high A → COMMIT
        result = verifier.verify(
            readings=[49.5, 50.0, 50.5],
            iot_value=50.0,
            gauge_range=[0, 100],
        )
        assert result.status == "COMMIT"
        assert result.score >= 0.82

    def test_historical_signal_neutral_when_absent(self, verifier):
        result = verifier.verify(
            readings=[49.5, 50.0, 50.5],
            iot_value=50.0,
            gauge_range=[0, 100],
            historical_baseline=None,
        )
        assert result.H == 0.5
        assert result.status == "COMMIT"


class TestEscalateAndOOD:
    def test_escalate_moderate_contradiction(self, verifier):
        # Mean reading 80, IoT says 20 — large gap → low A
        result = verifier.verify(
            readings=[78.0, 80.0, 82.0],
            iot_value=20.0,
            gauge_range=[0, 100],
        )
        # A = 1 - |80 - 20| / 100 = 0.40 → score roughly 0.35*C + 0.35*0.40 + 0.30*H
        assert result.status in {"ESCALATE", "OOD_FLAG"}

    def test_ood_very_low_score(self, verifier):
        # Readings near top, IoT near bottom, historical near bottom
        result = verifier.verify(
            readings=[88.0, 90.0, 92.0],
            iot_value=5.0,
            gauge_range=[0, 100],
            historical_baseline=5.0,
        )
        # A = 1 - |90-5|/100 = 0.15; H = 1 - |90-5|/100 = 0.15 → score ≈ 0.35*C + 0.35*0.15 + 0.30*0.15
        assert result.status == "OOD_FLAG"


class TestFMAnnotations:
    def test_fm7_annotation_when_a_lt_015(self, verifier):
        # A = 1 - |90 - 3| / 100 = 0.13 < 0.15 → sensor-physical contradiction annotation fires
        result = verifier.verify(
            readings=[88.0, 90.0, 92.0],
            iot_value=3.0,
            gauge_range=[0, 100],
        )
        assert any("FM-7" in ann for ann in result.fm_annotations)

    def test_fm7c_annotation_when_h_lt_015(self, verifier):
        # H = 1 - |90 - 3| / 100 = 0.13 < 0.15 → historical outlier annotation fires; IoT agrees (A fine)
        result = verifier.verify(
            readings=[88.0, 90.0, 92.0],
            iot_value=90.0,
            gauge_range=[0, 100],
            historical_baseline=3.0,
        )
        assert any("FM-7c" in ann for ann in result.fm_annotations)

    def test_no_fm_annotations_for_clean_commit(self, verifier):
        result = verifier.verify(
            readings=[49.5, 50.0, 50.5],
            iot_value=50.0,
            gauge_range=[0, 100],
            historical_baseline=50.0,
        )
        assert result.fm_annotations == []
        assert result.status == "COMMIT"


