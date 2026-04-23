"""
ONNX Runtime backend for YOLOv8/v11 and RT-DETR exported models.

Both YOLO and RT-DETR export to ONNX with the same output layout when using
the Ultralytics exporter:
    output: (1, num_classes + 4, num_proposals)  — merged box+class tensor

This backend handles both layouts:
    - YOLOv8 / v11:  output shape (1, 4+num_cls, 8400)
    - RT-DETR:       output shape (1, num_queries, 4+num_cls) — transposed variant
"""

import logging
import time
from typing import Optional

import numpy as np

from perceptra_inference.exceptions import ModelLoadError, InferenceError
from perceptra_inference.models import BoundingBox, Prediction
from perceptra_inference.utils.nms import nms

logger = logging.getLogger(__name__)


class YOLOOnnxBackend:
    """
    ONNX Runtime inference backend for YOLO / RT-DETR models.

    Expects the ONNX model exported via:
        model.export(format="onnx", simplify=True)
    """

    def __init__(
        self,
        onnx_bytes: bytes,
        class_names: list[str],
        device: str = "cpu",
        precision: str = "fp32",
    ) -> None:
        self._onnx_bytes = onnx_bytes
        self.class_names = class_names
        self.device = device
        self.precision = precision
        self._session = None
        self._input_name: Optional[str] = None
        self._input_shape: Optional[tuple] = None   # (1, 3, H, W)

    def load(self) -> None:
        try:
            import onnxruntime as ort

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device == "cuda"
                else ["CPUExecutionProvider"]
            )

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._session = ort.InferenceSession(
                self._onnx_bytes,
                sess_options=sess_options,
                providers=providers,
            )
            self._input_name = self._session.get_inputs()[0].name
            self._input_shape = tuple(self._session.get_inputs()[0].shape)
            logger.info(
                "ONNX session loaded. input=%s shape=%s providers=%s",
                self._input_name, self._input_shape,
                self._session.get_providers(),
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load ONNX model: {e}") from e

    def predict(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.25,
        max_detections: int = 100,
        iou_threshold: float = 0.45,
    ) -> list[Prediction]:
        if self._session is None:
            raise InferenceError("Model not loaded. Call load() first.")

        h, w = image.shape[:2]
        input_h, input_w = self._get_input_dims()

        blob, scale, pad = self._preprocess(image, input_h, input_w)

        t0 = time.perf_counter()
        raw_outputs = self._session.run(None, {self._input_name: blob})
        logger.debug("ONNX forward pass: %.1f ms", (time.perf_counter() - t0) * 1000)

        predictions = self._postprocess(
            raw_outputs[0],
            scale=scale,
            pad=pad,
            orig_h=h,
            orig_w=w,
            conf_thresh=confidence_threshold,
            iou_thresh=iou_threshold,
            max_det=max_detections,
        )
        return predictions

    def close(self) -> None:
        self._session = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_input_dims(self) -> tuple[int, int]:
        """Extract (H, W) from input shape (1, 3, H, W)."""
        if self._input_shape and len(self._input_shape) == 4:
            h, w = self._input_shape[2], self._input_shape[3]
            if isinstance(h, int) and isinstance(w, int):
                return h, w
        return 640, 640

    def _preprocess(
        self, image: np.ndarray, input_h: int, input_w: int
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Letterbox resize, normalize to [0,1], add batch dim.

        Returns:
            blob: (1, 3, H, W) float32
            scale: resize ratio (shorter side)
            pad: (pad_top, pad_left) pixels added
        """
        orig_h, orig_w = image.shape[:2]
        scale = min(input_h / orig_h, input_w / orig_w)
        new_h, new_w = int(round(orig_h * scale)), int(round(orig_w * scale))

        import cv2
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_top = (input_h - new_h) // 2
        pad_left = (input_w - new_w) // 2
        canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        canvas[pad_top: pad_top + new_h, pad_left: pad_left + new_w] = resized

        blob = canvas.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)
        return blob, scale, (pad_top, pad_left)

    def _postprocess(
        self,
        output: np.ndarray,
        scale: float,
        pad: tuple[int, int],
        orig_h: int,
        orig_w: int,
        conf_thresh: float,
        iou_thresh: float,
        max_det: int,
    ) -> list[Prediction]:
        """
        Decode raw ONNX output into Prediction objects.

        Supported output layouts:
            (1, 4+num_cls, num_proposals)  — YOLOv8/v11 default
            (1, num_proposals, 4+num_cls)  — some RT-DETR exports
        """
        # Remove batch dim
        out = output[0]  # (4+num_cls, N) or (N, 4+num_cls)

        num_cls = len(self.class_names)

        # Normalise to (N, 4+num_cls)
        if out.shape[0] == 4 + num_cls:
            out = out.T
        # out is now (N, 4+num_cls)

        cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        class_scores = out[:, 4:]  # (N, num_cls)

        if num_cls == 1:
            scores = class_scores[:, 0]
            class_ids = np.zeros(len(scores), dtype=int)
        else:
            class_ids = class_scores.argmax(axis=1)
            scores = class_scores[np.arange(len(class_ids)), class_ids]

        # Confidence filter
        mask = scores >= conf_thresh
        cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
        scores, class_ids = scores[mask], class_ids[mask]

        if len(scores) == 0:
            return []

        # cxcywh → xyxy (pixel coords in letterboxed space)
        pad_top, pad_left = pad
        x1 = cx - bw / 2 - pad_left
        y1 = cy - bh / 2 - pad_top
        x2 = cx + bw / 2 - pad_left
        y2 = cy + bh / 2 - pad_top

        # Rescale to original image
        x1 /= scale; y1 /= scale; x2 /= scale; y2 /= scale

        boxes_px = np.stack([x1, y1, x2, y2], axis=1)
        keep = nms(boxes_px, scores, iou_threshold=iou_thresh)
        keep = keep[:max_det]

        predictions: list[Prediction] = []
        for idx in keep:
            bx1 = float(np.clip(boxes_px[idx, 0] / orig_w, 0.0, 1.0))
            by1 = float(np.clip(boxes_px[idx, 1] / orig_h, 0.0, 1.0))
            bx2 = float(np.clip(boxes_px[idx, 2] / orig_w, 0.0, 1.0))
            by2 = float(np.clip(boxes_px[idx, 3] / orig_h, 0.0, 1.0))

            cid = int(class_ids[idx])
            label = self.class_names[cid] if cid < len(self.class_names) else str(cid)

            predictions.append(
                Prediction(
                    class_id=cid,
                    class_label=label,
                    confidence=float(scores[idx]),
                    bbox=BoundingBox(x_min=bx1, y_min=by1, x_max=bx2, y_max=by2),
                )
            )

        return predictions
