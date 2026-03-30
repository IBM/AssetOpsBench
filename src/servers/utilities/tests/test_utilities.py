"""Tests for Utilities MCP server tools."""

import json
import os
import tempfile

import pytest
from servers.utilities.main import mcp
from .conftest import call_tool


# ---------------------------------------------------------------------------
# current_date_time
# ---------------------------------------------------------------------------


class TestCurrentDateTime:
    @pytest.mark.anyio
    async def test_response_structure(self):
        data = await call_tool(mcp, "current_date_time", {})
        assert "currentDateTime" in data
        assert "currentDateTimeDescription" in data

    @pytest.mark.anyio
    async def test_description_format(self):
        data = await call_tool(mcp, "current_date_time", {})
        desc = data["currentDateTimeDescription"]
        assert "Today's date is" in desc
        assert "time is" in desc

    @pytest.mark.anyio
    async def test_iso_format(self):
        data = await call_tool(mcp, "current_date_time", {})
        # Should contain a T separator (ISO 8601)
        assert "T" in data["currentDateTime"]


# ---------------------------------------------------------------------------
# current_time_english
# ---------------------------------------------------------------------------


class TestCurrentTimeEnglish:
    @pytest.mark.anyio
    async def test_response_structure(self):
        data = await call_tool(mcp, "current_time_english", {})
        assert "english" in data
        assert "iso" in data

    @pytest.mark.anyio
    async def test_english_is_readable(self):
        data = await call_tool(mcp, "current_time_english", {})
        # pendulum's to_datetime_string returns "YYYY-MM-DD HH:MM:SS"
        parts = data["english"].split(" ")
        assert len(parts) == 2  # date + time


# ---------------------------------------------------------------------------
# json_reader
# ---------------------------------------------------------------------------


class TestJsonReader:
    @pytest.mark.anyio
    async def test_reads_valid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump({"test": "data"}, tmp)
            tmp_name = tmp.name

        try:
            data = await call_tool(mcp, "json_reader", {"file_name": tmp_name})
            assert data == {"test": "data"}
        finally:
            os.remove(tmp_name)

    @pytest.mark.anyio
    async def test_reads_nested_json(self):
        payload = {"a": [1, 2, 3], "b": {"nested": True}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(payload, tmp)
            tmp_name = tmp.name

        try:
            data = await call_tool(mcp, "json_reader", {"file_name": tmp_name})
            assert data == payload
        finally:
            os.remove(tmp_name)

    @pytest.mark.anyio
    async def test_nonexistent_file(self):
        data = await call_tool(
            mcp, "json_reader", {"file_name": "/tmp/does_not_exist_12345.json"}
        )
        assert "error" in data

    @pytest.mark.anyio
    async def test_invalid_json_content(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp.write("not valid json {{{")
            tmp_name = tmp.name

        try:
            data = await call_tool(mcp, "json_reader", {"file_name": tmp_name})
            assert "error" in data
        finally:
            os.remove(tmp_name)
