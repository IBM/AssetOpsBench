from abc import ABC, abstractmethod

import logging
from scenario_server.entities import (
    Scenario,
    ScenarioType,
    ScenarioAnswer,
    SubmissionResult,
)

logger: logging.Logger = logging.getLogger("scenario-server")


class ScenarioHandler(ABC):
    id: str = ""
    title: str = ""
    description: str = ""

    def __init__(self) -> None:
        # configure with environment variables
        ...

    @abstractmethod
    def scenario_type(self) -> ScenarioType: ...

    @abstractmethod
    def fetch_scenarios(self) -> list[Scenario]: ...

    @abstractmethod
    async def grade_responses(
        self, submission: list[ScenarioAnswer]
    ) -> SubmissionResult: ...
