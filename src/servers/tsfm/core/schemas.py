from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
class Provenance(str, Enum):
    pretrained = "pretrained"
    finetuned = "finetuned"
    trained = "trained"
    external_hf = "external_hf"
    external_service = "external_service"
    toolkit = "toolkit"


class Status(str, Enum):
    active = "active"
    deprecated = "deprecated"
    experimental = "experimental"
    superseded = "superseded"


class Modality(str, Enum):
    timeseries = "timeseries"
    vision = "vision"
    text = "text"
    multimodal = "multimodal"
    audio = "audio"


class Metric(BaseModel):
    model_config = ConfigDict(extra="allow")
    metric: str
    value: Optional[float] = None


# --------------------------------------------------------------------------- #
class ModelCard(BaseModel):
    """A model-store entry. Pointer index: weights live at one of artifact_path / hf_repo /
    remote_endpoint / model_checkpoint (toolkit)."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())

    model_id: str
    description: str = Field(min_length=3)
    task_ids: List[str] = Field(min_length=1)

    model_checkpoint: Optional[str] = None
    framework: Optional[str] = None
    model_family: Optional[str] = None
    modality: Modality = Modality.timeseries
    provenance: Provenance = Provenance.pretrained
    base_model_id: Optional[str] = None
    usage_modes: List[str] = Field(default_factory=list)
    output_type: Optional[str] = None
    context_length: Optional[int] = None
    prediction_length: Optional[int] = None
    domain: str = "general"
    frequency: Optional[str] = None

    source: Optional[str] = None
    artifact_path: Optional[str] = None
    hf_repo: Optional[str] = None
    remote_endpoint: Optional[str] = None
    pipeline_type: Optional[str] = None

    # sktime resolution + agent-reasoned config (read by resolver / param_space / composition)
    sktime_class: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)
    param_hints: Dict[str, Any] = Field(default_factory=dict)
    training_regime: Optional[str] = None

    metrics: List[Metric] = Field(default_factory=list)
    trained_on: Optional[Any] = None
    tags: List[str] = Field(default_factory=list)
    status: Status = Status.active
    version: str = "1"
    created_by: str = "seed"
    created_at: str = Field(default_factory=_now)

    @field_validator("context_length", "prediction_length")
    @classmethod
    def _non_negative(cls, v):
        if v is not None and v < 0:
            raise ValueError("length must be >= 0")
        return v

    @model_validator(mode="after")
    def _resolvable(self):
        # must be loadable somewhere (else it's a catalog-only stub, allowed but flagged)
        refs = [
            self.artifact_path,
            self.hf_repo,
            self.remote_endpoint,
            self.model_checkpoint,
        ]
        object.__setattr__(self, "resolvable", any(refs) or self.source == "toolkit")
        if self.provenance == Provenance.finetuned and not self.base_model_id:
            raise ValueError("finetuned model requires base_model_id (lineage)")
        return self

    def to_doc(self) -> dict:
        d = self.model_dump(mode="json")
        d["_id"] = f"model:{self.model_id}"
        return d


# --------------------------------------------------------------------------- #
class Interface(str, Enum):
    fit_transform = "fit_transform"
    fit_transform_inverse = "fit_transform_inverse"


class FeatureCard(BaseModel):
    """A feature-store entry: an EFE-style fit/transform program stored as code."""

    model_config = ConfigDict(extra="allow")

    feature_id: str
    interface: Interface
    code: str = Field(min_length=10)
    class_name: str = "Transformation"
    name: Optional[str] = None
    description: Optional[str] = None
    modality: Modality = Modality.timeseries
    invertible: bool = False

    provenance: str = "handwritten"  # handwritten | evolved | library
    method: Optional[str] = None
    parent_feature_id: Optional[str] = None
    generation: int = 0
    target_task: Optional[str] = None
    target_model: Optional[str] = None
    dataset: Optional[str] = None
    output_type: Optional[str] = None
    metrics: List[Metric] = Field(default_factory=list)
    columns_added: List[str] = Field(default_factory=list)
    columns_dropped: List[str] = Field(default_factory=list)
    validity: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    status: Status = Status.active
    version: str = "1"
    created_by: str = "seed"
    created_at: str = Field(default_factory=_now)

    @model_validator(mode="after")
    def _invertible_iface(self):
        if self.invertible and self.interface != Interface.fit_transform_inverse:
            raise ValueError("invertible=True requires interface=fit_transform_inverse")
        if "inverse_transform" in self.code and not self.invertible:
            # allow, but normalize the flag
            object.__setattr__(self, "invertible", True)
        return self

    def to_doc(self) -> dict:
        d = self.model_dump(mode="json")
        d["_id"] = f"feature:{self.feature_id}"
        return d


def validate_model(doc: dict) -> dict:
    return ModelCard(**doc).to_doc()


def validate_feature(doc: dict) -> dict:
    return FeatureCard(**doc).to_doc()