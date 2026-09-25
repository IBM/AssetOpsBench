"""Tests for the shared server clock and its ASSETOPS_FIXED_DATETIME override."""

from datetime import datetime, timezone

import pytest
from servers.clock import FIXED_DATETIME_ENV, fixed_datetime, now_utc


class TestUnpinned:
    def test_returns_live_clock_when_unset(self, monkeypatch):
        monkeypatch.delenv(FIXED_DATETIME_ENV, raising=False)
        assert fixed_datetime() is None
        before = datetime.now(timezone.utc)
        assert now_utc() >= before

    def test_blank_value_is_ignored(self, monkeypatch):
        monkeypatch.setenv(FIXED_DATETIME_ENV, "   ")
        assert fixed_datetime() is None


class TestPinned:
    @pytest.mark.parametrize(
        "raw",
        [
            "2025-01-15T09:00:00Z",
            "2025-01-15T09:00:00+00:00",
            "2025-01-15T09:00:00",  # naive → assumed UTC
        ],
    )
    def test_parses_iso_forms_as_utc(self, monkeypatch, raw):
        monkeypatch.setenv(FIXED_DATETIME_ENV, raw)
        assert now_utc() == datetime(2025, 1, 15, 9, 0, tzinfo=timezone.utc)

    def test_offset_is_normalised_to_utc(self, monkeypatch):
        monkeypatch.setenv(FIXED_DATETIME_ENV, "2025-01-15T09:00:00+02:00")
        assert now_utc() == datetime(2025, 1, 15, 7, 0, tzinfo=timezone.utc)

    def test_repeated_calls_are_identical(self, monkeypatch):
        monkeypatch.setenv(FIXED_DATETIME_ENV, "2025-01-15T09:00:00Z")
        assert now_utc() == now_utc()

    def test_invalid_value_raises(self, monkeypatch):
        monkeypatch.setenv(FIXED_DATETIME_ENV, "yesterday")
        with pytest.raises(ValueError, match=FIXED_DATETIME_ENV):
            now_utc()
