"""GET /health — liveness + readiness probe."""

import time

from fastapi import APIRouter, Request

from app.core.config import settings
from app.models.schemas import ComponentHealth, HealthResponse

router = APIRouter()
_START_TIME = time.monotonic()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Returns liveness status of the service and its dependencies.",
)
async def health_check(request: Request) -> HealthResponse:
    components = {}

    # ── Qdrant ────────────────────────────────────────────────────────────────
    try:
        qdrant_info = request.app.state.vector_store.health_check()
        components["qdrant"] = ComponentHealth(
            status="ok",
            latency_ms=qdrant_info["latency_ms"],
            detail=f"{qdrant_info['vectors_count']} vectors indexed",
        )
    except Exception as exc:
        components["qdrant"] = ComponentHealth(status="down", detail=str(exc))

    # ── InsightFace ───────────────────────────────────────────────────────────
    face_service_loaded = hasattr(request.app.state, "face_service")
    components["insightface"] = ComponentHealth(
        status="ok" if face_service_loaded else "down",
        detail=f"model={settings.INSIGHTFACE_MODEL}",
    )

    all_ok = all(c.status == "ok" for c in components.values())
    overall = "healthy" if all_ok else "degraded"

    return HealthResponse(
        status=overall,
        version=settings.APP_VERSION,
        uptime_seconds=round(time.monotonic() - _START_TIME, 1),
        components=components,
    )
