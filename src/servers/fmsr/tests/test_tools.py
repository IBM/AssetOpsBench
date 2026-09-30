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


class TestMissingDatabaseMessage:
    @pytest.mark.anyio
    async def test_missing_database_reports_unavailable(self, monkeypatch):
        from couchdb3.exceptions import NotFoundError

        class MissingDatabase:
            def get(self, *args, **kwargs):
                raise NotFoundError(
                    '{"error":"not_found","reason":"Database does not exist."}'
                )

            find = get

        monkeypatch.setattr("servers.fmsr.main.fm_db", MissingDatabase())

        data = await call_tool(mcp, "get_failure_modes", {"asset_class": "pump"})

        assert "does not exist in this environment" in data["error"]
        assert "failure_mode" not in data["error"]
        assert "no failure_mode record" not in data["error"]


class TestInterpretDGA:
    @pytest.mark.anyio
    async def test_missing_asset_name_returns_error(self):
        data = await call_tool(mcp, "interpret_dga", {
            "asset_name": "",
            "hydrogen": 10.0,
            "methane": 5.0,
            "acetylene": 0.5,
            "ethylene": 1.0,
            "ethane": 0.1,
        })
        assert "error" in data

    @pytest.mark.anyio
    async def test_llm_unavailable_returns_error(self, no_llm):
        data = await call_tool(mcp, "interpret_dga", {
            "asset_name": "Transformer1",
            "hydrogen": 10.0,
            "methane": 5.0,
            "acetylene": 0.5,
            "ethylene": 1.0,
            "ethane": 0.1,
        })
        assert "error" in data

    @pytest.mark.anyio
    async def test_parse_llm_response(self, mock_dga_chain):
        """Test correct parsing of a mocked LLM response."""
        data = await call_tool(mcp, "interpret_dga", {
            "asset_name": "Transformer1",
            "hydrogen": 10.0,
            "methane": 5.0,
            "acetylene": 0.5,
            "ethylene": 1.0,
            "ethane": 0.1,
        })
        assert "fault_type" in data
        assert isinstance(data["r1"], float)
        assert isinstance(data["r2"], float)
        assert isinstance(data["r3"], float)
        assert isinstance(data["confidence"], str)
        mock_dga_chain.assert_called_once()

    @requires_watsonx
    @pytest.mark.anyio
    async def test_integration(self):
        data = await call_tool(mcp, "interpret_dga", {
            "asset_name": "Transformer1",
            "hydrogen": 10.0,
            "methane": 5.0,
            "acetylene": 0.5,
            "ethylene": 1.0,
            "ethane": 0.1,
        })
        assert "fault_type" in data
        assert len(data["reasoning"]) > 0


class TestAssessWindingTemperature:
    @pytest.mark.anyio
    async def test_missing_asset_name_returns_error(self):
        data = await call_tool(mcp, "assess_winding_temperature", {
            "asset_name": "",
            "wti": 80,
            "oti": 90,
            "ati": 85,
            "oti_a": 3,
            "oti_t": 5,
        })
        assert "error" in data

    @pytest.mark.anyio
    async def test_llm_unavailable_returns_error(self, no_llm):
        data = await call_tool(mcp, "assess_winding_temperature", {
            "asset_name": "Transformer1",
            "wti": 80,
            "oti": 90,
            "ati": 85,
            "oti_a": 3,
            "oti_t": 5,
        })
        assert "error" in data

    @pytest.mark.anyio
    async def test_parse_llm_response(self, mock_winding_chain):
        data = await call_tool(mcp, "assess_winding_temperature", {
            "asset_name": "Transformer1",
            "wti": 80,
            "oti": 90,
            "ati": 85,
            "oti_a": 3,
            "oti_t": 5,
        })
        assert "thermal_status" in data
        assert isinstance(data["ageing_rate"], float)
        assert isinstance(data["alarm_active"], bool)
        assert isinstance(data["trip_active"], bool)
        mock_winding_chain.assert_called_once()

    @requires_watsonx
    @pytest.mark.anyio
    async def test_integration(self):
        data = await call_tool(mcp, "assess_winding_temperature", {
            "asset_name": "Transformer1",
            "wti": 80,
            "oti": 90,
            "ati": 85,
            "oti_a": 3,
            "oti_t": 5,
        })
        assert "thermal_status" in data
        assert len(data["recommended_action"]) > 0


class TestAssessLoadProfile:
    @pytest.mark.anyio
    async def test_missing_asset_name_returns_error(self):
        data = await call_tool(mcp, "assess_load_profile", {
            "asset_name": "",
            "vl1": 10, "vl2": 10, "vl3": 10,
            "il1": 5, "il2": 5, "il3": 5,
            "vl12": 20, "vl23": 20, "vl31": 20,
            "inut": 5,
            "rated_mva": 50,
        })
        assert "error" in data

    @pytest.mark.anyio
    async def test_llm_unavailable_returns_error(self, no_llm):
        data = await call_tool(mcp, "assess_load_profile", {
            "asset_name": "Transformer1",
            "vl1": 10, "vl2": 10, "vl3": 10,
            "il1": 5, "il2": 5, "il3": 5,
            "vl12": 20, "vl23": 20, "vl31": 20,
            "inut": 5,
            "rated_mva": 50,
        })
        assert "error" in data

    @pytest.mark.anyio
    async def test_parse_llm_response(self, mock_load_chain):
        data = await call_tool(mcp, "assess_load_profile", {
            "asset_name": "Transformer1",
            "vl1": 10, "vl2": 10, "vl3": 10,
            "il1": 5, "il2": 5, "il3": 5,
            "vl12": 20, "vl23": 20, "vl31": 20,
            "inut": 5,
            "rated_mva": 50,
        })
        assert "load_mva" in data
        assert isinstance(data["load_factor_pct"], float)
        assert isinstance(data["current_imbalance_pct"], float)
        assert isinstance(data["neutral_current_flag"], bool)
        mock_load_chain.assert_called_once()

    @requires_watsonx
    @pytest.mark.anyio
    async def test_integration(self):
        data = await call_tool(mcp, "assess_load_profile", {
            "asset_name": "Transformer1",
            "vl1": 10, "vl2": 10, "vl3": 10,
            "il1": 5, "il2": 5, "il3": 5,
            "vl12": 20, "vl23": 20, "vl31": 20,
            "inut": 5,
            "rated_mva": 50,
        })
        assert "load_mva" in data
        assert len(data["reasoning"]) > 0


class TestTransformerHealthIndexModel:
    VALID_FEATURES = {
        "hydrogen": 100,
        "oxygen": 10,
        "nitrogen": 200,
        "methane": 50,
        "co": 5,
        "co2": 20,
        "ethylene": 15,
        "ethane": 8,
        "acetylene": 3,
        "dbds": 0.5,
        "power_factor": 0.95,
        "interfacial_v": 15,
        "dielectric_rigidity": 30,
        "water_content": 0.2,
    }
    @pytest.mark.anyio
    async def test_missing_asset_name_returns_error(self):
        data = await call_tool(mcp, "predict_health_index", {
            "asset_name": "",
            **self.VALID_FEATURES
        })
        assert "error" in data

    @pytest.mark.anyio
    async def test_predicts_without_llm(self, no_llm):
        data = await call_tool(mcp, "predict_health_index", {
            "asset_name": "Transformer1",
            **self.VALID_FEATURES
        })
        assert "asset_name" in data
        assert "health_index" in data
        assert "condition" in data

    @requires_watsonx
    @pytest.mark.anyio
    async def test_integration(self):
        data = await call_tool(mcp, "predict_health_index", {
            "asset_name": "Transformer1",
            **self.VALID_FEATURES
        })
        assert "asset_name" in data
        assert "health_index" in data
        assert "condition" in data
        assert data["asset_name"] == "Transformer1"
        assert data["condition"] in ["Very Poor", "Poor", "Fair", "Good", "Very Good"]
