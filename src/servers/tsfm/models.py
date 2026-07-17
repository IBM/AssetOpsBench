"""Pydantic result models for the TSFM feature catalog MCP server."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict


class ErrorResult(BaseModel):
    error: str


class FeaturesResult(BaseModel):
    features: List[dict]


class RegisterResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    id: str


class CardResult(BaseModel):
    model_config = ConfigDict(extra="allow")


class LineageResult(BaseModel):
    model_config = ConfigDict(extra="allow")
