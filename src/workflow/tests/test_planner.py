"""Tests for the Planner and parse_plan()."""

from workflow.planner import Planner, parse_plan

_TWO_STEP = """\
#Task1: List all available IoT sites
#Server1: iot
#Tool1: sites
#Dependency1: None
#ExpectedOutput1: A list of site names

#Task2: Get assets at site MAIN
#Server2: iot
#Tool2: assets
#Dependency2: #S1
#ExpectedOutput2: A list of asset IDs"""

_MULTI_DEP = """\
#Task1: Get sites
#Server1: iot
#Tool1: sites
#Dependency1: None
#ExpectedOutput1: Sites

#Task2: Get current time
#Server2: utilities
#Tool2: current_date_time
#Dependency2: None
#ExpectedOutput2: Current time

#Task3: Combine results
#Server3: utilities
#Tool3: none
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

    def test_tool_name_signature_stripped(self):
        """LLM sometimes copies the 'tool(params)' format from server descriptions.

        parse_plan must strip the signature so the bare name reaches _call_tool.
        """
        raw = (
            "#Task1: Get sites\n"
            "#Server1: iot\n"
            "#Tool1: sites()\n"
            "#Dependency1: None\n"
            "#ExpectedOutput1: Sites\n\n"
            "#Task2: Get assets\n"
            "#Server2: iot\n"
            "#Tool2: assets(site_name: string)\n"
            "#Dependency2: #S1\n"
            "#ExpectedOutput2: Assets"
        )
        plan = parse_plan(raw)
        assert plan.steps[0].tool == "sites"
        assert plan.steps[1].tool == "assets"

    def test_tool_args_always_empty(self):
        """Planner no longer generates args — tool_args is always {}."""
        plan = parse_plan(_TWO_STEP)
        assert plan.steps[0].tool_args == {}
        assert plan.steps[1].tool_args == {}

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

    def test_args_lines_in_raw_are_ignored(self):
        """#Args lines left over from old prompts are silently ignored."""
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
            '#Args2: {"site_name": "MAIN"}\n'
            "#Dependency2: #S1\n"
            "#ExpectedOutput2: Assets"
        )
        plan = parse_plan(raw)
        assert plan.steps[0].tool_args == {}
        assert plan.steps[1].tool_args == {}


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

    def test_generate_plan_prompt_does_not_mention_args(self, mock_llm):
        """New prompt must not instruct the LLM to fill in #Args."""
        captured = []
        llm = mock_llm(_TWO_STEP)
        original = llm.generate
        llm.generate = lambda p, **kw: (captured.append(p), original(p))[1]

        Planner(llm).generate_plan("Q", {"iot": "  - sites(): List sites"})
        assert "#Args" not in captured[0]
