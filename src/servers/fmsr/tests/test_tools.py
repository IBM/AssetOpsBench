"""Tests for FMSR MCP server tools."""

import pytest

from servers.fmsr.main import mcp

from .conftest import call_tool, requires_watsonx


class TestGetFailureModes:
    @pytest.mark.anyio
    async def test_reads_failure_modes_from_db(self, fake_fm_db):
        data = await call_tool(mcp, "get_failure_modes", {"asset_class": "pump"})

        assert data["asset_class"] == "pump"
        assert data["failure_modes"] == ["seal leakage", "impeller wear"]
        assert data["exhaustive"] is False
        assert data["source"] == "synthetic sample"

    @pytest.mark.anyio
    async def test_asset_class_case_normalized(self, fake_fm_db):
        data = await call_tool(mcp, "get_failure_modes", {"asset_class": "PUMP"})

        assert data["asset_class"] == "pump"
        assert data["failure_modes"] == ["seal leakage", "impeller wear"]

    @pytest.mark.anyio
    async def test_asset_class_spacing_normalized(self, fake_fm_db):
        data = await call_tool(mcp, "get_failure_modes", {"asset_class": "  PUMP  "})

        assert data["asset_class"] == "pump"
        assert data["failure_modes"] == ["seal leakage", "impeller wear"]

    @pytest.mark.anyio
    async def test_empty_asset_class_returns_error(self, fake_fm_db):
        data = await call_tool(mcp, "get_failure_modes", {"asset_class": ""})

        assert data == {"error": "asset_class is required"}

    @pytest.mark.anyio
    async def test_missing_asset_class_returns_guidance(self, fake_fm_db):
        data = await call_tool(mcp, "get_failure_modes", {"asset_class": "bad-pump-1"})

        assert "error" in data
        assert "no failure_mode record for asset_class 'bad pump'" in data["error"]
        assert "Input was normalized from 'bad-pump-1'" in data["error"]
        assert "Available asset_class values include: pump" in data["error"]

    @pytest.mark.anyio
    async def test_db_unavailable_returns_error(self, monkeypatch):
        monkeypatch.setattr("servers.fmsr.main.fm_db", None)

        data = await call_tool(mcp, "get_failure_modes", {"asset_class": "pump"})

        assert data == {"error": "database not connected"}

    @pytest.mark.anyio
    async def test_database_read_error_returns_error(self, broken_fm_db):
        data = await call_tool(mcp, "get_failure_modes", {"asset_class": "pump"})

        assert data == {
            "error": "database lookup failed for asset_class 'pump': database read failed"
        }


class TestGenerateFailureModes:
    @pytest.mark.anyio
    async def test_extends_failure_modes_from_db(
        self, fake_fm_db, mock_failure_mode_generation
    ):
        data = await call_tool(
            mcp,
            "generate_failure_modes",
            {"asset_class": "Pump", "max_modes": 5},
        )

        assert data["asset_class"] == "pump"
        assert data["known"] == ["seal leakage", "impeller wear"]
        assert data["generated"] == ["bearing wear", "motor overheating"]
        assert data["failure_modes"] == [
            "seal leakage",
            "impeller wear",
            "bearing wear",
            "motor overheating",
        ]
        assert data["source"].startswith("LLM:")
        assert "nothing was persisted" in data["message"]
        mock_failure_mode_generation.assert_called_once_with(
            "pump", ["seal leakage", "impeller wear"], 5
        )

    @pytest.mark.anyio
    async def test_generates_from_scratch_for_missing_db_record(
        self, empty_fm_db, mock_failure_mode_generation
    ):
        data = await call_tool(
            mcp,
            "generate_failure_modes",
            {"asset_class": "compressor", "max_modes": 3},
        )

        assert data["asset_class"] == "compressor"
        assert data["known"] == []
        assert data["generated"] == [
            "bearing wear",
            "seal leakage",
            "motor overheating",
        ]
        mock_failure_mode_generation.assert_called_once_with("compressor", [], 3)

    @pytest.mark.anyio
    async def test_database_read_error_returns_error(
        self, broken_fm_db, mock_failure_mode_generation
    ):
        data = await call_tool(
            mcp,
            "generate_failure_modes",
            {"asset_class": "pump", "max_modes": 3},
        )

        assert data == {
            "error": "database lookup failed for asset_class 'pump': database read failed"
        }
        mock_failure_mode_generation.assert_not_called()

    @pytest.mark.anyio
    async def test_empty_asset_class_returns_error(self, mock_failure_mode_generation):
        data = await call_tool(
            mcp,
            "generate_failure_modes",
            {"asset_class": "", "max_modes": 3},
        )

        assert data == {"error": "asset_class is required"}

    @pytest.mark.anyio
    async def test_invalid_max_modes_returns_error(self, mock_failure_mode_generation):
        data = await call_tool(
            mcp,
            "generate_failure_modes",
            {"asset_class": "pump", "max_modes": 0},
        )

        assert data == {"error": "max_modes must be greater than 0"}

    @pytest.mark.anyio
    async def test_llm_unavailable_returns_error(self, no_llm):
        data = await call_tool(
            mcp,
            "generate_failure_modes",
            {"asset_class": "pump", "max_modes": 3},
        )

        assert data == {"error": "LLM unavailable"}

    @requires_watsonx
    @pytest.mark.anyio
    async def test_integration(self):
        data = await call_tool(
            mcp,
            "generate_failure_modes",
            {
                "asset_class": "pump",
                "max_modes": 2,
            },
        )

        assert "generated" in data
        assert len(data["generated"]) <= 2


class TestAddFailureModes:
    @pytest.mark.anyio
    async def test_merges_with_existing_modes(self, fake_fm_db):
        data = await call_tool(
            mcp,
            "add_failure_modes",
            {
                "asset_class": "Pump-1",
                "failure_modes": ["impeller wear", "bearing wear"],
                "exhaustive": True,
                "source": "unit-test",
            },
        )

        assert data["asset_class"] == "pump"
        assert data["added"] == ["bearing wear"]
        assert data["failure_modes"] == [
            "seal leakage",
            "impeller wear",
            "bearing wear",
        ]
        assert data["total"] == 3
        assert data["exhaustive"] is True
        assert data["source"] == "unit-test"
        assert fake_fm_db.docs["fm:pump"]["failure_modes"] == [
            "seal leakage",
            "impeller wear",
            "bearing wear",
        ]
        assert fake_fm_db.docs["fm:pump"]["exhaustive"] is True
        assert fake_fm_db.docs["fm:pump"]["source"] == "unit-test"

    @pytest.mark.anyio
    async def test_omitted_exhaustive_preserves_existing_value(self, fake_fm_db):
        fake_fm_db.docs["fm:pump"]["exhaustive"] = True

        data = await call_tool(
            mcp,
            "add_failure_modes",
            {
                "asset_class": "pump",
                "failure_modes": ["bearing wear"],
            },
        )

        assert data["exhaustive"] is True
        assert fake_fm_db.docs["fm:pump"]["exhaustive"] is True

    @pytest.mark.anyio
    async def test_creates_new_asset_class_record(self, empty_fm_db):
        data = await call_tool(
            mcp,
            "add_failure_modes",
            {
                "asset_class": "Gearbox-1",
                "failure_modes": ["gear tooth wear", "bearing wear"],
            },
        )

        assert data["asset_class"] == "gearbox"
        assert data["added"] == ["gear tooth wear", "bearing wear"]
        assert data["failure_modes"] == ["gear tooth wear", "bearing wear"]
        assert data["total"] == 2
        assert data["exhaustive"] is False
        assert data["source"] == "user"
        assert empty_fm_db.docs["fm:gearbox"]["asset_class"] == "gearbox"
        assert empty_fm_db.docs["fm:gearbox"]["failure_modes"] == [
            "gear tooth wear",
            "bearing wear",
        ]

    @pytest.mark.anyio
    async def test_empty_asset_class_returns_error(self, fake_fm_db):
        data = await call_tool(
            mcp,
            "add_failure_modes",
            {"asset_class": "", "failure_modes": ["bearing wear"]},
        )

        assert data == {"error": "asset_class is required"}

    @pytest.mark.anyio
    async def test_empty_failure_modes_returns_error(self, fake_fm_db):
        data = await call_tool(
            mcp,
            "add_failure_modes",
            {"asset_class": "pump", "failure_modes": []},
        )

        assert data == {"error": "failure_modes list is required"}

    @pytest.mark.anyio
    async def test_db_unavailable_returns_error(self, monkeypatch):
        monkeypatch.setattr("servers.fmsr.main.fm_db", None)

        data = await call_tool(
            mcp,
            "add_failure_modes",
            {"asset_class": "pump", "failure_modes": ["bearing wear"]},
        )

        assert data == {"error": "database not connected"}

    @pytest.mark.anyio
    async def test_database_read_error_returns_error(self, broken_fm_db):
        data = await call_tool(
            mcp,
            "add_failure_modes",
            {"asset_class": "pump", "failure_modes": ["bearing wear"]},
        )

        assert data == {
            "error": "database lookup failed for asset_class 'pump': database read failed"
        }


class TestToolRegistration:
    @pytest.mark.anyio
    async def test_mapping_tool_is_not_registered(self):
        tools = await mcp.list_tools()

        assert "generate_failure_mode_sensor_mapping" not in {
            tool.name for tool in tools
        }
