"""Tests for the Planner and parse_plan()."""

from workflow.executor import _has_placeholders
from workflow.planner import Planner, parse_plan

_TWO_STEP = """\
#Task1: List all available IoT sites
#Server1: iot
#Tool1: sites
#Args1: {}
#Dependency1: None
#ExpectedOutput1: A list of site names

#Task2: Get assets at site MAIN
#Server2: iot
#Tool2: assets
#Args2: {"site_name": "MAIN"}
#Dependency2: #S1
#ExpectedOutput2: A list of asset IDs"""

_MULTI_DEP = """\
#Task1: Get sites
#Server1: iot
#Tool1: sites
#Args1: {}
#Dependency1: None
#ExpectedOutput1: Sites

#Task2: Get current time
#Server2: utilities
#Tool2: current_date_time
#Args2: {}
#Dependency2: None
#ExpectedOutput2: Current time

#Task3: Combine results
#Server3: utilities
#Tool3: none
#Args3: {}
#Dependency3: #S1, #S2
#ExpectedOutput3: Combined output"""

_NO_TASKS = "No tasks here."


class TestParsePlan:
    def test_two_steps_parsed(self):
        plan = parse_plan(_TWO_STEP)
        assert len(plan.steps) == 2

    def test_step_numbers(self):
        plan = parse_plan(_TWO_STEP)
        assert plan.steps[0].step_number == 1
        assert plan.steps[1].step_number == 2

    def test_task_text(self):
        plan = parse_plan(_TWO_STEP)
        assert "IoT sites" in plan.steps[0].task
        assert "assets" in plan.steps[1].task

    def test_server_names(self):
        plan = parse_plan(_TWO_STEP)
        assert plan.steps[0].server == "iot"
        assert plan.steps[1].server == "iot"

    def test_tool_names(self):
        plan = parse_plan(_TWO_STEP)
        assert plan.steps[0].tool == "sites"
        assert plan.steps[1].tool == "assets"

    def test_tool_args_parsed(self):
        plan = parse_plan(_TWO_STEP)
        assert plan.steps[0].tool_args == {}
        assert plan.steps[1].tool_args == {"site_name": "MAIN"}

    def test_no_dependency(self):
        plan = parse_plan(_TWO_STEP)
        assert plan.steps[0].dependencies == []

    def test_single_dependency(self):
        plan = parse_plan(_TWO_STEP)
        assert plan.steps[1].dependencies == [1]

    def test_multiple_dependencies(self):
        plan = parse_plan(_MULTI_DEP)
        assert set(plan.steps[2].dependencies) == {1, 2}

    def test_raw_preserved(self):
        plan = parse_plan(_TWO_STEP)
        assert plan.raw == _TWO_STEP

    def test_expected_output_captured(self):
        plan = parse_plan(_TWO_STEP)
        assert "site names" in plan.steps[0].expected_output.lower()

    def test_empty_input_yields_empty_plan(self):
        plan = parse_plan("")
        assert plan.steps == []

    def test_no_matching_blocks_yields_empty_plan(self):
        plan = parse_plan(_NO_TASKS)
        assert plan.steps == []

    def test_placeholder_args_preserved_as_string(self):
        """parse_plan stores {step_N} placeholder strings verbatim in tool_args.

        This is the format the LLM actually generates: _PLAN_PROMPT uses
        str.format(), which converts {{step_N}} in the template to {step_N}
        (single braces) in the rendered prompt.  The LLM echoes that back.
        """
        raw = (
            "#Task1: Get sites\n"
            "#Server1: iot\n"
            "#Tool1: sites\n"
            "#Args1: {}\n"
            "#Dependency1: None\n"
            "#ExpectedOutput1: Sites\n\n"
            "#Task2: Get assets\n"
            "#Server2: iot\n"
            "#Tool2: assets\n"
            '#Args2: {"site_name": "{step_1}"}\n'
            "#Dependency2: #S1\n"
            "#ExpectedOutput2: Assets"
        )
        plan = parse_plan(raw)
        assert plan.steps[1].tool_args == {"site_name": "{step_1}"}

    def test_placeholder_in_parsed_args_detected(self):
        """After parse_plan, {step_N} args are detected as placeholders by the executor.

        Regression guard for the bug where _PLACEHOLDER_RE matched {{step_N}}
        (double braces) instead of the {step_N} (single braces) that the LLM
        actually produces.
        """
        raw = (
            "#Task1: Get sites\n"
            "#Server1: iot\n"
            "#Tool1: sites\n"
            "#Args1: {}\n"
            "#Dependency1: None\n"
            "#ExpectedOutput1: Sites\n\n"
            "#Task2: Get assets\n"
            "#Server2: iot\n"
            "#Tool2: assets\n"
            '#Args2: {"site_name": "{step_1}"}\n'
            "#Dependency2: #S1\n"
            "#ExpectedOutput2: Assets"
        )
        plan = parse_plan(raw)
        assert _has_placeholders(plan.steps[1].tool_args) is True

    def test_invalid_args_json_falls_back_to_empty(self):
        raw = (
            "#Task1: Do something\n"
            "#Server1: iot\n"
            "#Tool1: sites\n"
            "#Args1: not-valid-json\n"
            "#Dependency1: None\n"
            "#ExpectedOutput1: result\n"
        )
        plan = parse_plan(raw)
        assert plan.steps[0].tool_args == {}


class TestPlanner:
    def test_generate_plan_uses_llm_output(self, mock_llm):
        llm = mock_llm(_TWO_STEP)
        planner = Planner(llm)
        plan = planner.generate_plan(
            "List all assets",
            {"iot": "  - sites(): List sites\n  - assets(site_name: string): List assets"},
        )
        assert len(plan.steps) == 2
        assert plan.steps[0].server == "iot"
        assert plan.steps[1].tool == "assets"

    def test_generate_plan_prompt_contains_question(self, mock_llm, monkeypatch):
        captured = []
        llm = mock_llm(_TWO_STEP)
        original = llm.generate
        llm.generate = lambda p, **kw: (captured.append(p), original(p))[1]

        Planner(llm).generate_plan(
            "What sensors exist for CH-1?",
            {"iot": "  - sites(): List sites"},
        )
        assert "What sensors exist for CH-1?" in captured[0]

    def test_generate_plan_prompt_contains_agent_names(self, mock_llm, monkeypatch):
        captured = []
        llm = mock_llm(_TWO_STEP)
        original = llm.generate
        llm.generate = lambda p, **kw: (captured.append(p), original(p))[1]

        Planner(llm).generate_plan(
            "Q",
            {"iot": "  - sites(): List sites", "utilities": "  - current_date_time(): Get time"},
        )
        assert "iot" in captured[0]
        assert "utilities" in captured[0]
