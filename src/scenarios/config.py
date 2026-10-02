"""Validated, secret-free run configuration shared by generator interfaces."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from .backends import BACKENDS
from .models import RetrieverMode
from .planning import ScenarioCounts, ScenarioPlan


class GeneratorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    asset_name: str = Field(min_length=1)
    backend: str = "litellm"
    model_id: str | None = Field(default=None, min_length=1)
    scenario_plan: ScenarioPlan | None = None
    scenario_counts: ScenarioCounts | None = None
    num_scenarios: int = Field(default=50, ge=0)
    num_negative_scenarios: int = Field(default=2, ge=0)
    batch_size: int = Field(default=5, gt=0)
    mode: Literal["open", "closed"] = "closed"
    retriever: RetrieverMode = "arxiv"
    research_file: str | None = Field(
        default=None, validation_alias=AliasChoices("research_file", "research_digest"),
    )
    show_workflow: bool = False
    log: bool = False

    @model_validator(mode="before")
    @classmethod
    def resolve_totals(cls, value):
        if not isinstance(value, dict):
            return value
        value = dict(value)
        for legacy in ("live_data", "data_in_couchdb"):
            if legacy in value:
                enabled = value.pop(legacy)
                if not isinstance(enabled, bool):
                    raise ValueError(f"{legacy} must be a boolean")
                mode = "open" if enabled else "closed"
                if "mode" in value and value["mode"] != mode:
                    raise ValueError(f"{legacy} conflicts with mode")
                value["mode"] = mode
        if value.get("scenario_plan") is not None and value.get("scenario_counts") is not None:
            raise ValueError("Use either scenario_plan or scenario_counts")
        if value.get("scenario_plan") is not None:
            plan = ScenarioPlan.model_validate(value["scenario_plan"])
            plan.check_totals(value.get("num_scenarios"), value.get("num_negative_scenarios"))
            return {
                **value, "scenario_plan": plan,
                "num_scenarios": plan.positive_total,
                "num_negative_scenarios": plan.negative_total,
            }
        if value.get("scenario_counts") is not None:
            counts = ScenarioCounts.model_validate(value["scenario_counts"])
            totals = {"num_scenarios": counts.positive, "num_negative_scenarios": counts.negative}
            for field, total in totals.items():
                if field in value and value[field] != total:
                    raise ValueError(f"{field} conflicts with scenario_counts total {total}")
            return {**value, "scenario_counts": counts, **totals}
        return value

    @model_validator(mode="after")
    def require_scenarios(self) -> Self:
        if self.num_scenarios + self.num_negative_scenarios == 0:
            raise ValueError("Request at least one positive or negative scenario")
        return self

    @field_validator("backend")
    @classmethod
    def supported_backend(cls, value: str) -> str:
        if value not in BACKENDS:
            raise ValueError(f"Unknown backend {value!r}; choose from {', '.join(BACKENDS)}")
        return value

    @property
    def live_data(self) -> bool:
        return self.mode == "open"

    @property
    def resolved_model(self) -> str | None:
        return self.model_id or BACKENDS[self.backend].default_model

    def resolved_dict(self) -> dict:
        return {**self.model_dump(), "model_id": self.resolved_model}
