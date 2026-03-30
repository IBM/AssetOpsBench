from dataclasses import dataclass


@dataclass
class ScenarioType:
    id: str
    title: str
    description: str


@dataclass
class Scenario:
    id: str
    query: str
    metadata: dict


@dataclass
class ScenarioSet:
    scenarios: list[Scenario]

    def get_scenario(self, sid: str):
        return next((entry for entry in self.scenarios if entry.id == sid), None)


@dataclass
class ScenarioAnswer:
    scenario_id: str
    answer: str


@dataclass
class ScenarioGrade:
    scenario_id: str
    correct: bool
    details: list


@dataclass
class SubmissionSummary:
    name: str
    value: str


@dataclass
class SubmissionResult:
    scenario_set_id: str
    summary: list[SubmissionSummary]
    grades: list[ScenarioGrade]


@dataclass
class Submission:
    experiment_id: str
    run_id: str
    scenario_set_id: str
    submission: list[ScenarioAnswer]
