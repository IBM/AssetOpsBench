from __future__ import annotations

import unittest
from unittest.mock import patch

from dashboard import server


class DashboardRepositoryTests(unittest.TestCase):
    def test_scenarios_are_loaded_from_repository_files(self):
        scenarios = server.load_scenarios()
        by_id = {scenario["id"]: scenario for scenario in scenarios}

        self.assertGreaterEqual(len(scenarios), 25)
        self.assertIn("304", by_id)
        self.assertIn("scenario_3/question.txt", by_id["3"]["source"])
        self.assertIn("characteristic_form", by_id["304"])

    def test_overview_uses_checked_in_data(self):
        overview = server.load_data_overview()

        self.assertGreater(overview["telemetry"]["records"], 0)
        self.assertGreater(overview["telemetry"]["observations"], 0)
        self.assertEqual(overview["fmea"]["names"], ["seal leakage", "impeller wear"])
        self.assertEqual(overview["workorders"]["records"], 3)
        self.assertEqual(overview["workorders"]["statuses"], {"WAPPR": 1, "APPR": 1, "COMP": 1})
        self.assertEqual(overview["kpis"]["completion_rate"], 33.3)
        self.assertFalse(overview["alerts"]["available"])

    def test_unknown_scenario_fails_before_agent_execution(self):
        with self.assertRaises(LookupError):
            server.run_scenario("not-a-repository-scenario", "plan-execute")

    def test_missing_provider_credentials_are_reported(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "LITELLM_BASE_URL, LITELLM_API_KEY"):
                server.run_scenario("304", "plan-execute", "litellm_proxy/test-model")

    def test_llm_judge_reports_absent_reference_answer(self):
        scenario = {
            "id": "1",
            "text": "A repository task",
            "category": "Repository scenario",
            "type": "",
            "characteristic_form": "",
        }

        result = server.evaluate_result(scenario, "answer", {}, "litellm_proxy/test-model")

        self.assertFalse(result["passed"])
        self.assertIn("no characteristic_form", result["error"])

    def test_dashboard_tool_policy_blocks_work_order_writes(self):
        with self.assertRaisesRegex(ValueError, "not enabled"):
            server.call_repository_tool("wo", "generate_work_order", {})

    def test_dashboard_tool_policy_blocks_persistent_fmsr_edits(self):
        with self.assertRaisesRegex(ValueError, "not enabled"):
            server.call_repository_tool("fmsr", "add_failure_modes", {})

    def test_local_profile_runs_only_scenarios_found_in_repository(self):
        profile = {
            "id": "mini",
            "source": "benchmarks/scenario_suite/mini.yaml",
            "scenario_count": 2,
            "scenario_ids": ["304", "external-scenario"],
            "locally_available": ["304"],
            "requires_data_reset": True,
        }
        with (
            patch.object(server, "benchmark_profiles", return_value=[profile]),
            patch.object(server, "run_scenario", return_value={"scenario_id": "304"}) as run,
        ):
            result = server.run_local_profile("mini", "plan-execute")

        run.assert_called_once_with("304", "plan-execute", None)
        self.assertEqual(result["skipped_scenario_ids"], ["external-scenario"])
        self.assertFalse(result["data_reset_performed"])


if __name__ == "__main__":
    unittest.main()
