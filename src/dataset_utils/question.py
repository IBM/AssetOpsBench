from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional


class QuestionDifficulty(Enum):
    entry = "Entry-Level"
    intermediary = "Intermediate"
    advance = "Advanced"


@dataclass
class Question:
    subject: str  # d
    id: int  # d
    question_id: str
    rule_id: int  # d
    rule_name: str
    date_detected: str  # d
    asset_type: str  # d
    condition_description: List[str]  # d
    temporal_condition: List[str]
    question: str
    question_prompt: str  # d
    options: List[str]  # d
    option_ids: List[str]  # d
    question_type: str  # d
    correct: List[bool]  # d
    answer_str: str
    rationale: str  # d
    tags: List[str]  # d
    difficulty: QuestionDifficulty  # d
    question_first: Optional[bool] = None  # d
    text_type: Optional[str] = "choice"
    relevancy = None
    verberlized_conditions: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        attrs = vars(self)
        attrs["difficulty"] = (
            str(attrs["difficulty"].value)
            if isinstance(attrs["difficulty"], QuestionDifficulty)
            else attrs["difficulty"]
        )

        return attrs

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Any:
        data["difficulty"] = (
            QuestionDifficulty(data["difficulty"])
            if isinstance(data["difficulty"], str)
            else data["difficulty"]
        )
        return Question(**data)
