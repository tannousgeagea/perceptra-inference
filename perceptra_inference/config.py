"""Configuration management for perceptra-inference."""

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class RuntimeConfig(BaseModel):
    device: Literal["cuda", "cpu", "tensorrt"] = "cuda"
    precision: Literal["fp32", "fp16"] = "fp32"
    max_loaded_models: int = 5


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    cors_origins: list[str] = ["*"]
    api_keys: list[str] = Field(default_factory=list)
    max_image_size_mb: int = 20
    max_image_dimension: int = 8000


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "text"] = "json"


class InferenceConfig(BaseModel):
    """Complete inference service configuration."""

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "InferenceConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def apply_env_overrides(self) -> None:
        """Apply INFERENCE_* environment variable overrides."""
        prefix = "INFERENCE_"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            parts = key[len(prefix):].lower().split("_", 1)
            if len(parts) < 2:
                continue
            section, field = parts[0], parts[1]
            if hasattr(self, section):
                section_obj = getattr(self, section)
                if hasattr(section_obj, field):
                    setattr(section_obj, field, value)
