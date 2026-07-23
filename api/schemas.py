from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str


class ServiceResponse(BaseModel):
    service: str
    version: str = "v1"
    data: dict[str, Any] | None = None
    error: ErrorDetail | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    services: list[str] = Field(default_factory=list)
