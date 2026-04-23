"""FastAPI application factory — mirrors perceptra-seg/service/main.py."""

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from perceptra_inference.config import InferenceConfig
from perceptra_inference.model_registry import ModelRegistry
from service.middleware import LoggingMiddleware
from service.routes import router

logger = logging.getLogger(__name__)


def _load_config() -> InferenceConfig:
    config_path = Path(os.getenv("INFERENCE_CONFIG_PATH", "config.yaml"))
    if config_path.exists():
        cfg = InferenceConfig.from_yaml(config_path)
    else:
        cfg = InferenceConfig()
    cfg.apply_env_overrides()
    return cfg


def create_app(config: InferenceConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI inference application."""
    cfg = config or _load_config()

    app = FastAPI(
        title="Perceptra Inference API",
        description="Production inference service for YOLO / RT-DETR ONNX models",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(LoggingMiddleware)

    app.state.config = cfg
    app.state.registry = ModelRegistry(
        max_models=cfg.runtime.max_loaded_models,
        device=cfg.runtime.device,
        precision=cfg.runtime.precision,
    )

    app.include_router(router, prefix="/v1")

    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    @app.on_event("startup")
    async def startup_event() -> None:
        logger.info(
            "perceptra-inference started. device=%s precision=%s max_models=%d",
            cfg.runtime.device,
            cfg.runtime.precision,
            cfg.runtime.max_loaded_models,
        )

        # Optionally pre-load models from env: INFERENCE_PRELOAD_MODELS=version_id:url:task:cls1,cls2|...
        preload = os.getenv("INFERENCE_PRELOAD_MODELS", "").strip()
        if preload:
            for entry in preload.split("|"):
                parts = entry.split(":", 3)
                if len(parts) < 2:
                    continue
                version_id, storage_url = parts[0], parts[1]
                task = parts[2] if len(parts) > 2 else "object-detection"
                class_names = parts[3].split(",") if len(parts) > 3 else []
                try:
                    app.state.registry.load_model(version_id, storage_url, task, class_names)
                    logger.info("Pre-loaded model: %s", version_id)
                except Exception:
                    logger.exception("Failed to pre-load model: %s", version_id)

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        for version_id in list(app.state.registry.loaded_version_ids()):
            try:
                app.state.registry.unload_model(version_id)
            except Exception:
                logger.exception("Error unloading model %s on shutdown", version_id)
        logger.info("perceptra-inference shutdown complete")

    return app


# For uvicorn: uvicorn service.main:app
app = create_app()
