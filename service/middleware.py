"""Structured request/response logging middleware — mirrors perceptra-seg."""

import json
import logging
import time
import uuid
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Response:
        request_id = str(uuid.uuid4())
        start_time = time.time()
        request.state.request_id = request_id

        logger.info(json.dumps({
            "event": "request_started",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else None,
        }))

        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Request failed")
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            logger.info(json.dumps({
                "event": "request_completed",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code if "response" in locals() else 500,
                "duration_ms": round(duration_ms, 2),
            }))

        return response
