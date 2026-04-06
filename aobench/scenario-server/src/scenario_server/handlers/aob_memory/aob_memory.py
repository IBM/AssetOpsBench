import json
import logging
from pathlib import Path

from scenario_server.entities import (
    Scenario,
    ScenarioAnswer,
    ScenarioGrade,
    ScenarioType,
    SubmissionResult,
    SubmissionSummary,
)
from scenario_server.grading import evaluation_agent
from scenario_server.handlers.scenario_handler import ScenarioHandler

logger: logging.Logger = logging.getLogger(__name__)
logger.debug(f"debug: {__name__}")

_SCENARIO_FILE = Path(__file__).parent / "memory_scenarios.jsonl"


class AOBMemoryScenarios(ScenarioHandler):
    id = "c7f3a2d1-9e4b-4f8a-b1c5-d2e6f7a8b9c0"
    title = "Asset Operations Bench - Memory"
    description = (
        "Multi-step scenarios that evaluate whether industrial AI agents correctly "
        "maintain and propagate context across reasoning steps. Covers asset identity "
        "persistence, failure mode propagation, temporal context retention, "
        "multi-entity tracking, and cross-agent context flow."
    )

    def __init__(self):
        self.scenario_data = dict()
        try:
            with open(_SCENARIO_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    sd = json.loads(line)
                    self.scenario_data[str(sd["id"])] = sd
        except Exception as e:
            logger.exception(f"failed to init AOBMemoryScenarios: {e=}")

    def _grade_answer(self, entry_id, answer) -> ScenarioGrade:
        try:
            unwrap = json.loads(answer)

            c = self.scenario_data[entry_id]["characteristic_form"]
            q = self.scenario_data[entry_id]["text"]
            r = unwrap["result"]
            t = unwrap["trace"]

            result, details = evaluation_agent(
                actual=r,
                charactistic=c,
                query=q,
                trace=t,
            )

            return ScenarioGrade(
                scenario_id=entry_id,
                correct=result,
                details=details,
            )
        except Exception as e:
            logger.exception(f"failed to grade {entry_id=} : {e=}")
            logger.debug(f"{entry_id=} / {answer=} / {self.scenario_data[entry_id]}")
            return ScenarioGrade(
                scenario_id=entry_id,
                correct=False,
                details=[{"error": f"failed to grade scenario id: {entry_id}"}],
            )

    def scenario_type(self) -> ScenarioType:
        return ScenarioType(id=self.id, title=self.title, description=self.description)

    def fetch_scenarios(self) -> list[Scenario]:
        scenarios = []

        for k, v in self.scenario_data.items():
            try:
                metadata = dict()

                if "category" in v:
                    metadata["category"] = v["category"]

                scenarios.append(
                    Scenario(
                        id=str(k),
                        query=v["text"],
                        metadata=metadata,
                    )
                )
            except Exception as e:
                logger.exception(f"failed to process {k}, {v} : {e=}")

        return scenarios

    async def grade_responses(
        self, submission: list[ScenarioAnswer]
    ) -> SubmissionResult:
        correct = 0
        grades = []
        for entry in submission:
            try:
                entry_id: str = entry.scenario_id
            except Exception:
                logger.exception(f"missing scenario id: {entry=}")
                continue

            if entry_id not in self.scenario_data:
                grades.append(
                    ScenarioGrade(
                        scenario_id=entry_id,
                        correct=False,
                        details=[{"error": f"unknown scenario id: {entry_id}"}],
                    )
                )
                continue

            g: ScenarioGrade = self._grade_answer(entry_id, entry.answer)
            if g.correct:
                correct += 1
            grades.append(g)

        summary: list[SubmissionSummary] = [
            SubmissionSummary(
                name="Correct",
                value=f"{correct}/{len(self.scenario_data)}",
            )
        ]

        return SubmissionResult(
            scenario_set_id=self.id,
            summary=summary,
            grades=grades,
        )


if __name__ == "__main__":
    import asyncio

    aobs = AOBMemoryScenarios()
    print(f"Loaded {len(aobs.scenario_data)} memory scenarios")

    submission: list[ScenarioAnswer] = [
        ScenarioAnswer(
            scenario_id="mem_001",
            answer=json.dumps(
                {
                    "trace": "IoT query returned Chiller-A with sensors S-101, S-102. FMSR returned failure mode FM-3 for Chiller-A.",
                    "result": "Work order created for Chiller-A: inspect sensor S-101 for failure mode FM-3 (bearing wear).",
                }
            ),
        ),
    ]
    grade: SubmissionResult = asyncio.run(aobs.grade_responses(submission=submission))
    print(f"{grade=}")
