"""Pydantic result models for the TSFM feature catalog MCP server."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict


class ErrorResult(BaseModel):
    error: str


class FeaturesResult(BaseModel):
    features: List[dict]
    message: str


class RegisterResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    id: str
    card: Dict[str, Any]
    message: str


class CardResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    message: str


class LineageResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    message: str
