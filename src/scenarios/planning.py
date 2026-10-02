"""Explicit scenario quotas shared by the CLI, service, and generator."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from .models import ScenarioTypeKey


class ScenarioCounts(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    positive: int = Field(default=0, ge=0, strict=True)
    negative: int = Field(default=0, ge=0, strict=True)


class ScenarioPlan(RootModel[dict[ScenarioTypeKey, ScenarioCounts]]):
    """Missing focuses and omitted counts request zero scenarios."""

    @model_validator(mode="after")
    def require_scenarios(self) -> Self:
        if self.positive_total + self.negative_total == 0:
            raise ValueError("Scenario plan must request at least one positive or negative scenario")
        return self

    @property
    def positive_counts(self) -> dict[str, int]:
        return {focus: counts.positive for focus, counts in self.root.items() if counts.positive}

    @property
    def negative_counts(self) -> dict[str, int]:
        return {focus: counts.negative for focus, counts in self.root.items() if counts.negative}

    @property
    def positive_total(self) -> int:
        return sum(counts.positive for counts in self.root.values())

    @property
    def negative_total(self) -> int:
        return sum(counts.negative for counts in self.root.values())

    def check_totals(self, positive: int | None, negative: int | None) -> None:
        for field, supplied, requested in (
            ("num_scenarios", positive, self.positive_total),
            ("num_negative_scenarios", negative, self.negative_total),
        ):
            if supplied is not None and supplied != requested:
                raise ValueError(f"{field}={supplied} conflicts with scenario_plan total {requested}")
