"""API route handlers for perceptra-inference."""

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status

from perceptra_inference.exceptions import ModelNotLoadedError, ModelLoadError
from perceptra_inference.models import (
    DetectionResult,
    HealthResponse,
    LoadModelRequest,
    UnloadModelResponse,
)
from perceptra_inference.utils.image_io import load_image

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------

async def get_registry(request: Request):
    registry = getattr(request.app.state, "registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Model registry not initialised")
    return registry


async def verify_api_key(request: Request) -> None:
    config = request.app.state.config
    if not config.server.api_keys:
        return
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = auth_header[len("Bearer "):]
    if token not in config.server.api_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/healthz", response_model=HealthResponse, tags=["health"])
async def healthz(registry=Depends(get_registry), request: Request = None):
    return HealthResponse(
        status="ok",
        loaded_models=registry.loaded_version_ids(),
        device=registry.device,
    )


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

@router.post(
    "/models/load",
    status_code=status.HTTP_201_CREATED,
    tags=["models"],
    dependencies=[Depends(verify_api_key)],
)
async def load_model(body: LoadModelRequest, registry=Depends(get_registry)):
    """Download an ONNX model from a presigned URL and register it for inference."""
    try:
        registry.load_model(
            version_id=body.version_id,
            storage_url=body.storage_url,
            task=body.task,
            class_names=body.class_names,
        )
    except ModelLoadError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error loading model %s", body.version_id)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    return {"version_id": body.version_id, "status": "loaded"}


@router.delete(
    "/models/{version_id}",
    response_model=UnloadModelResponse,
    tags=["models"],
    dependencies=[Depends(verify_api_key)],
)
async def unload_model(version_id: str, registry=Depends(get_registry)):
    """Unload a model from the registry and free its memory."""
    registry.unload_model(version_id)
    return UnloadModelResponse(version_id=version_id)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@router.post("/infer/{model_version_id}", response_model=DetectionResult, tags=["inference"])
async def infer(
    model_version_id: str,
    file: UploadFile = File(...),
    confidence_threshold: Optional[float] = Query(0.25, ge=0.0, le=1.0),
    max_detections: Optional[int] = Query(100, ge=1, le=1000),
    registry=Depends(get_registry),
):
    """Run object detection on an uploaded image using the specified model version."""
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty file upload")

    try:
        image = load_image(raw_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    t0 = time.perf_counter()
    try:
        predictions = registry.predict(
            version_id=model_version_id,
            image=image,
            confidence_threshold=confidence_threshold,
            max_detections=max_detections,
        )
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Inference failed for model %s", model_version_id)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    inference_ms = (time.perf_counter() - t0) * 1000
    logger.debug("Inference: model=%s detections=%d time=%.1fms",
                 model_version_id, len(predictions), inference_ms)

    return DetectionResult(
        model_version_id=model_version_id,
        predictions=predictions,
        inference_time_ms=round(inference_ms, 2),
    )
