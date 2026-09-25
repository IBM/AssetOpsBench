"""Tests for FMSR MCP server tools."""

import pytest

from servers.fmsr.main import mcp

from .conftest import call_tool


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
        assert "no failure_mode record for asset_class 'bad pump 1'" in data["error"]
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


class TestAssetClassMatching:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "query",
        [
            "Hydraulic Pump",
            "hydraulic pump",
            "HYDRAULIC_PUMP",
            "hydraulic-pump",
            "  hydraulic   pump ",
        ],
    )
    async def test_matches_unnormalised_stored_name(self, messy_fm_db, query):
        data = await call_tool(mcp, "get_failure_modes", {"asset_class": query})

        assert data["asset_class"] == "Hydraulic Pump"
        assert data["failure_modes"] == ["cavitation"]

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "query, expected",
        [
            ("co2 compressor", "CO2 Compressor"),
            ("CO2-Compressor", "CO2 Compressor"),
            ("co compressor", "CO Compressor"),
        ],
    )
    async def test_digits_in_name_are_kept(self, messy_fm_db, query, expected):
        data = await call_tool(mcp, "get_failure_modes", {"asset_class": query})

        assert data["asset_class"] == expected

    @pytest.mark.anyio
    async def test_instance_id_is_not_a_class(self, messy_fm_db):
        data = await call_tool(
            mcp, "get_failure_modes", {"asset_class": "Hydraulic Pump 2"}
        )

        assert (
            "no failure_mode record for asset_class 'hydraulic pump 2'"
            in (data["error"])
        )
        assert "Did you mean: Hydraulic Pump?" in data["error"]

    @pytest.mark.anyio
    async def test_does_not_match_on_id(self, messy_fm_db):
        data = await call_tool(mcp, "get_failure_modes", {"asset_class": "fan"})

        assert "error" in data
        assert "no failure_mode record for asset_class 'fan'" in data["error"]

    @pytest.mark.anyio
    async def test_add_updates_matched_record_in_place(self, messy_fm_db):
        data = await call_tool(
            mcp,
            "add_failure_modes",
            {"asset_class": "hydraulic_pump", "failure_modes": ["seal wear"]},
        )

        assert data["asset_class"] == "Hydraulic Pump"
        assert data["failure_modes"] == ["cavitation", "seal wear"]
        assert len(messy_fm_db.docs) == 5
        assert messy_fm_db.docs["fm:Hydraulic Pump"]["failure_modes"] == [
            "cavitation",
            "seal wear",
        ]

    @pytest.mark.anyio
    async def test_add_new_class_keeps_digits(self, empty_fm_db):
        data = await call_tool(
            mcp,
            "add_failure_modes",
            {"asset_class": "CO2-Compressor", "failure_modes": ["valve leakage"]},
        )

        assert data["asset_class"] == "co2 compressor"
        assert empty_fm_db.docs["fm:co2 compressor"]["asset_class"] == "co2 compressor"


class TestDuplicateNormalizedClasses:
    @pytest.mark.anyio
    async def test_lowest_id_wins(self, duplicate_fm_db):
        data = await call_tool(mcp, "get_failure_modes", {"asset_class": "PUMP"})

        assert data["asset_class"] == "pump"

    @pytest.mark.anyio
    async def test_add_and_get_resolve_to_same_record(self, duplicate_fm_db):
        await call_tool(
            mcp,
            "add_failure_modes",
            {"asset_class": "Pump", "failure_modes": ["bearing wear"]},
        )
        data = await call_tool(mcp, "get_failure_modes", {"asset_class": "Pump"})

        assert data["failure_modes"] == ["from pump", "bearing wear"]
        assert duplicate_fm_db.docs["b"]["failure_modes"] == ["from Pump"]


class TestAddFailureModes:
    @pytest.mark.anyio
    async def test_merges_with_existing_modes(self, fake_fm_db):
        data = await call_tool(
            mcp,
            "add_failure_modes",
            {
                "asset_class": "Pump",
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
                "asset_class": "Gearbox",
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
    async def test_new_record_is_found_by_later_lookup(self, empty_fm_db):
        await call_tool(
            mcp,
            "add_failure_modes",
            {"asset_class": "Cooling_Tower", "failure_modes": ["fill fouling"]},
        )
        data = await call_tool(
            mcp, "get_failure_modes", {"asset_class": "Cooling Tower"}
        )

        assert data["asset_class"] == "cooling tower"
        assert data["failure_modes"] == ["fill fouling"]

    @pytest.mark.anyio
    async def test_database_write_error_returns_error(self, write_failing_fm_db):
        data = await call_tool(
            mcp,
            "add_failure_modes",
            {"asset_class": "pump", "failure_modes": ["bearing wear"]},
        )

        assert data == {"error": "database write failed"}
        assert write_failing_fm_db.docs["fm:pump"]["failure_modes"] == ["seal leakage"]

    @pytest.mark.anyio
    async def test_concurrent_update_returns_conflict_error(self, racing_fm_db):
        data = await call_tool(
            mcp,
            "add_failure_modes",
            {"asset_class": "pump", "failure_modes": ["bearing wear"]},
        )

        assert "conflict" in data["error"].lower()
        assert racing_fm_db.docs["fm:pump"]["failure_modes"] == [
            "seal leakage",
            "concurrent edit",
        ]

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
    async def test_only_catalog_tools_are_registered(self):
        tools = await mcp.list_tools()

        assert {tool.name for tool in tools} == {
            "get_failure_modes",
            "add_failure_modes",
        }
