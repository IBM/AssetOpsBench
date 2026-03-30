import json
import logging

from huggingface_hub import hf_hub_download
from scenario_server.entities import (
    Scenario,
    ScenarioType,
    ScenarioAnswer,
    ScenarioGrade,
    SubmissionResult,
    SubmissionSummary,
)
from scenario_server.grading import evaluation_agent
from scenario_server.handlers.scenario_handler import ScenarioHandler

logger: logging.Logger = logging.getLogger(__name__)
logger.debug(f"debug: {__name__}")

HUGGINGFACE_REPO = "ibm-research/AssetOpsBench"
HUGGINGFACE_DATA = "data/scenarios/all_utterance.jsonl"


class AOBIoTScenarios(ScenarioHandler):
    id = "b3aa206a-f7dc-43c9-a1f4-dcf984417487"
    title = "Asset Operations Bench - IoT"
    description = "Human-authored evaluation prompts for industrial asset agents."

    def __init__(self):
        self.scenario_data = dict()
        try:
            cache: str = hf_hub_download(
                repo_id=HUGGINGFACE_REPO,
                filename=HUGGINGFACE_DATA,
                repo_type="dataset",
            )

            with open(cache, "r") as f:
                scenario_data = [json.loads(line) for line in f]

            for sd in scenario_data:
                if "type" in sd and sd["type"].lower() == "iot":
                    self.scenario_data[str(sd["id"])] = sd

        except Exception as e:
            logger.exception(f"failed to init AOBScenarios: {e=}")

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
            except Exception as e:
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

    aobs = AOBIoTScenarios()
    submission: list[ScenarioAnswer] = [
        ScenarioAnswer(
            scenario_id="Q.S5",
            answer='[{"scenario_id": "Q.S5.0", "answer": ""}]',
        ),
        ScenarioAnswer(
            scenario_id="2",
            answer="",
        ),
        ScenarioAnswer(
            scenario_id="2",
            answer=json.dumps(
                {
                    "trace": "query database for iot sites",
                    "result": ["Downtown", "Uptown"],
                }
            ),
        ),
    ]
    grade: SubmissionResult = asyncio.run(aobs.grade_responses(submission=submission))
    print(f"{grade=}")
