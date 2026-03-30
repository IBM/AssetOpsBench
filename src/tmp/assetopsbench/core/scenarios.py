from __future__ import annotations
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class Scenario(BaseModel):
    id: str = Field(..., description="Primary identifier (string).")
    uuid: Optional[str] = Field(None, description="Optional file-unique id.")
    type: Optional[str] = Field(None, description="Type (e.g., knowledge, reasoning, skill, tool skill).")  # Changed to Optional
    text: str = Field(..., description="Prompt / question / utterance text (real question).")
    category: Optional[str] = Field(None, description="Category (e.g., single-agent, multi-agent, multi-turn, dialog).")
    characteristic_form: Optional[str] = Field(
        None,
        description="Descriptive form capturing how the agent should have solved the problem."
    )
    deterministic: bool = Field(False, description="Whether the scenario is deterministic.")
    expected_result: Optional[Any] = Field(None, description="Optional expected result for deterministic scenarios.")
    data: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary key-value data for the scenario.")
    source: Optional[str] = Field(None, description="Optional source (dataset name, file name, etc.).")

    model_config = {
        "extra": "ignore",  # Changed from "forbid" to "ignore"
        "str_strip_whitespace": True,
    }

