"""Pydantic schemas for request/response objects."""

from typing import Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Normalized bounding box [0, 1]."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def as_list(self) -> list[float]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]


class Prediction(BaseModel):
    """Single detection prediction."""
    class_id: int
    class_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox


class DetectionResult(BaseModel):
    """Full inference result for one image."""
    model_version_id: str
    predictions: list[Prediction]
    inference_time_ms: float


class LoadModelRequest(BaseModel):
    """Request to load an ONNX model into the registry."""
    version_id: str
    storage_url: str = Field(..., description="Presigned URL to the ONNX model file")
    task: str = Field(default="object-detection", description="Model task type")
    class_names: list[str] = Field(default_factory=list, description="Ordered class name list")


class UnloadModelResponse(BaseModel):
    version_id: str
    status: str = "unloaded"


class HealthResponse(BaseModel):
    status: str = "ok"
    loaded_models: list[str]
    device: str
