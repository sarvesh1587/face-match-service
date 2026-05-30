"""
Face Match Service — Production Entry Point
==========================================
FastAPI application with lifespan management, structured logging,
startup validation, and OpenAPI customization.
"""

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.core.config import settings
from app.core.logger import get_logger
from app.core.startup import run_startup_checks
from app.services.face_service import FaceService
from app.services.vector_store import VectorStore

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Manage application lifecycle.
    Runs startup checks, initialises shared services, then tears down cleanly.
    """
    logger.info("Starting Face Match Service", version=settings.APP_VERSION)

    # Validate environment and dependencies before accepting traffic
    await run_startup_checks()

    # Initialise singletons and attach to app state so routes can access them
    app.state.face_service = FaceService()
    app.state.vector_store = VectorStore()
    await app.state.vector_store.initialize()

    logger.info(
        "Service ready",
        qdrant_host=settings.QDRANT_HOST,
        collection=settings.QDRANT_COLLECTION,
        model=settings.INSIGHTFACE_MODEL,
    )

    yield  # ← application is live here

    # Graceful shutdown
    logger.info("Shutting down Face Match Service")
    await app.state.vector_store.close()


def create_application() -> FastAPI:
    application = FastAPI(
        title="Face Match Service",
        description=(
            "Production-grade face enrollment and vector search service.\n\n"
            "Uses InsightFace ArcFace embeddings stored in Qdrant for "
            "sub-millisecond approximate nearest-neighbour search."
        ),
        version=settings.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        contact={
            "name": "Face Match Service",
            "url": "https://github.com/yourhandle/face-match-service",
        },
        license_info={"name": "MIT"},
    )

    # ── Middleware ────────────────────────────────────────────────────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Per-request latency logging
    @application.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=round(elapsed_ms, 2),
        )
        response.headers["X-Response-Time-Ms"] = str(round(elapsed_ms, 2))
        return response

    # ── Routes ────────────────────────────────────────────────────────────────
    application.include_router(api_router)

    # ── Global exception handler ──────────────────────────────────────────────
    @application.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.error(
            "unhandled_exception",
            path=request.url.path,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    return application


app = create_application()
