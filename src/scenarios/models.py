"""Pydantic models used by the scenario-generation pipeline."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ScenarioTypeKey = Literal["iot", "fmsr", "tsfm", "wo", "vibration", "multiagent"]

RetrieverMode = Literal["arxiv", "semantic_scholar"]

PdfTextOutcome = Literal["ok", "fetch_failed", "empty_text"]


class KeyDescription(BaseModel):
    """A named item with a short human-readable description."""

    key: str
    description: str


class GroundedTimeRange(BaseModel):
    """Concrete time coverage discovered from a live data source."""

    start: str | None = None
    end: str | None = None
    total_observations: int = 0


class SensorNameDescription(BaseModel):
    """One sensor channel with a human-readable description (from profile synthesis)."""

    name: str
    description: str


class AssetInstance(BaseModel):
    """One live asset id with per-instance coverage (time ranges).

    Sensor inventories are shared across instances and live on ``GroundingBundle`` as
    ``iot_sensors`` / ``vibration_sensors`` (prior data for synthesis only).
    """

    site_name: str
    asset_id: str
    has_iot: bool = False
    has_vibration: bool = False
    iot_time_range: GroundedTimeRange | None = None
    vibration_time_range: GroundedTimeRange | None = None


class GroundingBundle(BaseModel):
    """Deterministic live grounding information collected before profile synthesis."""

    asset_name: str
    requested_open_form: bool = False
    open_form_eligible: bool = False
    iot_sensors: list[str] = Field(default_factory=list)
    vibration_sensors: list[str] = Field(default_factory=list)
    asset_instances: list[AssetInstance] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    failure_sensor_mapping: dict[str, list[str]] = Field(default_factory=dict)
    sensor_failure_mapping: dict[str, list[str]] = Field(default_factory=dict)


class AssetProfile(BaseModel):
    """Structured knowledge about an asset and the tools relevant to it."""

    asset_name: str
    generation_mode: str = "closed_form"
    description: str
    iot_sensors: list[SensorNameDescription] = Field(default_factory=list)
    vibration_sensors: list[SensorNameDescription] = Field(default_factory=list)
    failure_modes: list[KeyDescription] = Field(default_factory=list)
    asset_instances: list[AssetInstance] = Field(default_factory=list)
    failure_sensor_mapping: dict[str, list[str]] = Field(default_factory=dict)
    sensor_failure_mapping: dict[str, list[str]] = Field(default_factory=dict)
    relevant_tools: dict[str, list[dict[str, str]]] = Field(default_factory=dict)
    operator_tasks: list[str] = Field(default_factory=list)
    manager_tasks: list[str] = Field(default_factory=list)

    def instances_for_focus(self, focus: str) -> list[AssetInstance]:
        """Return grounded instances that can support the requested focus."""

        if focus == "iot":
            return [instance for instance in self.asset_instances if instance.has_iot]
        if focus == "vibration":
            return [instance for instance in self.asset_instances if instance.has_vibration]
        if focus in {"fmsr", "tsfm", "wo", "multiagent"}:
            return [
                instance
                for instance in self.asset_instances
                if instance.has_iot or instance.has_vibration
            ]
        return list(self.asset_instances)

    def grounded_sites(self) -> list[str]:
        """Return the distinct grounded site names."""

        return sorted({instance.site_name for instance in self.asset_instances if instance.site_name})

    def grounded_asset_ids(self, focus: str | None = None) -> list[str]:
        """Return grounded asset ids, optionally filtered by focus."""

        instances = self.instances_for_focus(focus) if focus else self.asset_instances
        return sorted({instance.asset_id for instance in instances if instance.asset_id})

    def grounded_sensor_names(self, focus: str | None = None) -> list[str]:
        """Return sensor names from the profile for the requested focus (open-form validation)."""

        if focus == "vibration":
            return sorted(dict.fromkeys(s.name for s in self.vibration_sensors if s.name))
        if focus == "iot":
            return sorted(dict.fromkeys(s.name for s in self.iot_sensors if s.name))
        combined = [*(s.name for s in self.iot_sensors), *(s.name for s in self.vibration_sensors)]
        return sorted(dict.fromkeys(n for n in combined if n))

    def grounded_timestamps(self, focus: str | None = None) -> list[str]:
        """Return explicit timestamp bounds discovered for grounded instances."""

        instances = self.instances_for_focus(focus) if focus else self.asset_instances
        timestamps: set[str] = set()
        for instance in instances:
            ranges = []
            if focus != "vibration" and instance.iot_time_range:
                ranges.append(instance.iot_time_range)
            if focus != "iot" and instance.vibration_time_range:
                ranges.append(instance.vibration_time_range)
            for time_range in ranges:
                if time_range.start:
                    timestamps.add(time_range.start)
                if time_range.end:
                    timestamps.add(time_range.end)
        return sorted(timestamps)


class RetrievalAction(BaseModel):
    """Next-step decision emitted by the retrieval planner."""

    action: str = "search"
    reason: str = ""
    canonical_asset_name: str
    queries: list[str] = Field(default_factory=list)
    selected_ids: list[str] = Field(default_factory=list)


class EvidenceCandidate(BaseModel):
    """Metadata for one potentially relevant paper or evidence source."""

    paper_id: str
    title: str
    summary: str
    query: str
    pdf_url: str | None = None
    published: str | None = None
    judge_score: int = 0
    judge_reason: str = ""


class EvidenceSnippet(BaseModel):
    """Extracted evidence text used to justify an asset profile."""

    paper_id: str
    title: str
    url: str | None = None
    source: str
    text: str


class EvidenceBundle(BaseModel):
    """All evidence collected for one asset profile build."""

    asset_name: str
    canonical_asset_name: str
    query_history: list[str] = Field(default_factory=list)
    selected_candidate_ids: list[str] = Field(default_factory=list)
    candidates: list[EvidenceCandidate] = Field(default_factory=list)
    snippets: list[EvidenceSnippet] = Field(default_factory=list)


class ScenarioBudget(BaseModel):
    """Requested scenario count plus per-subagent allocation."""

    total_scenarios: int = 50
    allocation: dict[str, int] = Field(default_factory=dict)
    reasoning: str = ""


class Scenario(BaseModel):
    """One benchmark scenario emitted by the generator."""

    id: str
    text: str
    category: str
    characteristic_form: str
    type: ScenarioTypeKey
    generation_mode: str | None = Field(default=None, exclude=True)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation for JSON export."""

        return {
            "id": self.id,
            "type": self.type,
            "text": self.text,
            "category": self.category,
            "characteristic_form": self.characteristic_form,
        }


class ScenarioGenerationResult(BaseModel):
    """Final output of one scenario-generation run."""

    scenarios: list[Scenario] = Field(default_factory=list)
    negative_scenarios: list[Scenario] = Field(default_factory=list)


__all__ = [
    "AssetProfile",
    "EvidenceBundle",
    "EvidenceCandidate",
    "EvidenceSnippet",
    "AssetInstance",
    "GroundedTimeRange",
    "GroundingBundle",
    "KeyDescription",
    "RetrievalAction",
    "RetrieverMode",
    "Scenario",
    "ScenarioGenerationResult",
    "ScenarioBudget",
    "ScenarioTypeKey",
    "SensorNameDescription",
]
