from typing import Literal

from pydantic import BaseModel, Field


class ServiceStatus(BaseModel):
    status: Literal["healthy", "unhealthy"] = Field(..., description="Service status")
    message: str = Field(..., description="Status details")


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = Field(..., description="Overall status")
    version: str
    environment: str
    service_name: str
    services: dict[str, ServiceStatus] = Field(default_factory=dict)
