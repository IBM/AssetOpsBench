from langchain_core.tools import BaseTool
from datetime import datetime
from typing import Any


class TimeTool(BaseTool):
    name: str = "time"
    description: str = "Get the current time"

    def _run(self, *args: Any) -> str:
        return "The current time is " + str(datetime.now())


def get_time_tools():
    return [TimeTool()]


def get_time_agent_name():
    return "Time Agent"
