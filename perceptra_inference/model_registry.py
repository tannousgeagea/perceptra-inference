"""
In-memory model registry with LRU eviction.

Mirrors perceptra-seg's app.state.models pattern but adds:
- Dynamic load/unload via API
- LRU eviction when max_loaded_models is reached
- Thread-safe access for concurrent inference requests
"""

import logging
import threading
from collections import OrderedDict

import numpy as np
import requests

from perceptra_inference.exceptions import ModelLoadError, ModelNotLoadedError
from perceptra_inference.models import Prediction

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Thread-safe LRU registry of loaded detection backends."""

    def __init__(self, max_models: int = 5, device: str = "cpu", precision: str = "fp32") -> None:
        self._models: OrderedDict[str, object] = OrderedDict()
        self._meta: dict[str, dict] = {}   # version_id → {task, class_names}
        self._lock = threading.Lock()
        self.max_models = max_models
        self.device = device
        self.precision = precision

    def load_model(
        self,
        version_id: str,
        storage_url: str,
        task: str = "object-detection",
        class_names: list[str] | None = None,
    ) -> None:
        """Download ONNX bytes from storage_url and load backend.

        If max_models is already reached the least-recently-used model is evicted.
        """
        logger.info("Loading model version_id=%s task=%s", version_id, task)

        onnx_bytes = self._download_onnx(storage_url)
        backend = self._build_backend(onnx_bytes, task, class_names or [])
        backend.load()

        with self._lock:
            if version_id in self._models:
                logger.info("Model %s already loaded — replacing", version_id)
                self._models[version_id].close()
                del self._models[version_id]

            # LRU eviction
            while len(self._models) >= self.max_models:
                oldest_id, oldest = self._models.popitem(last=False)
                logger.info("Evicting LRU model %s to make room", oldest_id)
                try:
                    oldest.close()
                except Exception:
                    pass
                self._meta.pop(oldest_id, None)

            self._models[version_id] = backend
            self._meta[version_id] = {"task": task, "class_names": class_names or []}

        logger.info("Model %s loaded successfully (%d total)", version_id, len(self._models))

    def unload_model(self, version_id: str) -> None:
        with self._lock:
            backend = self._models.pop(version_id, None)
            self._meta.pop(version_id, None)

        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass
            logger.info("Model %s unloaded", version_id)
        else:
            logger.warning("Model %s was not loaded — nothing to unload", version_id)

    def predict(
        self,
        version_id: str,
        image: np.ndarray,
        confidence_threshold: float = 0.25,
        max_detections: int = 100,
    ) -> list[Prediction]:
        with self._lock:
            backend = self._models.get(version_id)
            if backend is None:
                raise ModelNotLoadedError(version_id)
            # Move to end (most recently used)
            self._models.move_to_end(version_id)

        return backend.predict(image, confidence_threshold, max_detections)

    def is_loaded(self, version_id: str) -> bool:
        with self._lock:
            return version_id in self._models

    def loaded_version_ids(self) -> list[str]:
        with self._lock:
            return list(self._models.keys())

    # ------------------------------------------------------------------

    def _download_onnx(self, url: str) -> bytes:
        try:
            response = requests.get(url, timeout=120, stream=True)
            response.raise_for_status()
            data = b"".join(response.iter_content(chunk_size=8192))
            logger.info("Downloaded ONNX model: %.1f MB", len(data) / 1024 / 1024)
            return data
        except Exception as e:
            raise ModelLoadError(f"Failed to download ONNX from {url}: {e}") from e

    def _build_backend(self, onnx_bytes: bytes, task: str, class_names: list[str]) -> object:
        from perceptra_inference.backends.onnx_yolo import YOLOOnnxBackend

        supported = {"object-detection", "segmentation", "classification", "obb"}
        if task not in supported:
            logger.warning("Unknown task '%s', defaulting to object-detection", task)

        return YOLOOnnxBackend(
            onnx_bytes=onnx_bytes,
            class_names=class_names,
            device=self.device,
            precision=self.precision,
        )
