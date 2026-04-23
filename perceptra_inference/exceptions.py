"""Custom exceptions for perceptra-inference."""


class InferenceError(Exception):
    """Base exception for all inference errors."""


class ModelNotLoadedError(InferenceError):
    """Raised when inference is requested for a model that is not loaded."""

    def __init__(self, version_id: str) -> None:
        super().__init__(f"Model version '{version_id}' is not loaded. Call /v1/models/load first.")
        self.version_id = version_id


class ModelLoadError(InferenceError):
    """Raised when a model fails to load from the ONNX artifact."""


class ImageLoadError(InferenceError):
    """Raised when an image cannot be decoded or fetched."""


class ConfigError(InferenceError):
    """Raised when configuration is invalid."""
