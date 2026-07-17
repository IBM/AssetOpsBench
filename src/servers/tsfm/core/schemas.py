from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
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


def validate_feature(doc: dict) -> dict:
    return FeatureCard(**doc).to_doc()
