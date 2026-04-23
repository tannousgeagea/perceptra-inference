"""Base protocol for all detection backends."""

from typing import Protocol, runtime_checkable

import numpy as np

from perceptra_inference.models import Prediction


@runtime_checkable
class BaseDetectionBackend(Protocol):
    """Protocol that all detection backends must satisfy."""

    def load(self) -> None:
        """Load model weights / initialize session."""
        ...

    def predict(
        self,
        image: np.ndarray,
        confidence_threshold: float,
        max_detections: int,
    ) -> list[Prediction]:
        """Run inference on a single RGB image (HxWx3).

        Args:
            image: RGB numpy array
            confidence_threshold: Minimum score to keep a prediction
            max_detections: Maximum number of returned predictions

        Returns:
            List of Prediction objects with normalized bbox coords.
        """
        ...

    def close(self) -> None:
        """Release resources (ONNX session, GPU memory, etc.)."""
        ...
