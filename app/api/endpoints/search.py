"""
POST /search
============
Accepts a query image, computes an ArcFace embedding, runs Qdrant ANN search,
applies the configured threshold, and returns ranked candidates with
confidence bands and detailed latency telemetry.
"""

import asyncio
import time
import uuid
from typing import List

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.core.config import settings
from app.core.exceptions import EmbeddingError, FaceQualityError, MultipleFacesError, NoFaceDetectedError, VectorStoreError
from app.core.logger import get_logger
from app.models.schemas import ConfidenceBand, ErrorResponse, SearchCandidate, SearchResponse
from app.services.metrics import metrics
from app.utils.confidence import classify

logger = get_logger(__name__)
router = APIRouter()

_SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


@router.post(
    "/search",
    response_model=SearchResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Image validation failed"},
        422: {"model": ErrorResponse, "description": "Request validation error"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    summary="Search for a matching identity",
    description=(
        "Upload a query face image. "
        "The service embeds it with ArcFace and runs Qdrant ANN search. "
        "Returns the top match (if above threshold) and all top-K candidates "
        "with cosine scores and confidence bands."
    ),
)
async def search_face(
    request: Request,
    image: UploadFile = File(..., description="Query face image (JPEG/PNG/WebP)"),
) -> SearchResponse:
    t_total_start = time.perf_counter()
    query_id = str(uuid.uuid4())

    # ── Content-type check ────────────────────────────────────────────────────
    ct = (image.content_type or "").split(";")[0].strip()
    if ct not in _SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image type '{ct}'. Use JPEG, PNG, or WebP.",
        )

    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file.")

    face_service = request.app.state.face_service
    vector_store = request.app.state.vector_store

    try:
        # ── Step 1: Embedding ─────────────────────────────────────────────────
        t_embed_start = time.perf_counter()
        embedding, quality = await asyncio.to_thread(
            face_service.process, image_bytes, True  # allow_multiple=True for search
        )
        embed_ms = (time.perf_counter() - t_embed_start) * 1000

        # ── Step 2: ANN search ────────────────────────────────────────────────
        raw_candidates, search_ms = await asyncio.to_thread(
            vector_store.search, embedding, settings.TOP_K
        )

        # ── Step 3: Enrich with threshold + confidence band ───────────────────
        candidates: List[SearchCandidate] = []
        for c in raw_candidates:
            score = c["cosine_score"]
            candidates.append(
                SearchCandidate(
                    identity_id=c["identity_id"],
                    cosine_score=score,
                    is_match=score >= settings.MATCH_THRESHOLD,
                    confidence_band=classify(score),
                    metadata=c.get("metadata"),
                )
            )

        top_match = candidates[0] if (candidates and candidates[0].is_match) else None
        enrolled_count = await asyncio.to_thread(vector_store.count)

    except (NoFaceDetectedError, MultipleFacesError, FaceQualityError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except (EmbeddingError, VectorStoreError) as exc:
        logger.error("search_pipeline_error", query_id=query_id, error=str(exc))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    total_ms = (time.perf_counter() - t_total_start) * 1000
    matched = top_match is not None
    metrics.record_search(total_ms, matched)

    logger.info(
        "search_complete",
        query_id=query_id,
        top_match=top_match.identity_id if top_match else None,
        top_score=candidates[0].cosine_score if candidates else None,
        is_match=matched,
        embed_ms=round(embed_ms, 2),
        search_ms=round(search_ms, 2),
        total_ms=round(total_ms, 2),
    )

    return SearchResponse(
        query_id=query_id,
        top_match=top_match,
        candidates=candidates,
        threshold_used=settings.MATCH_THRESHOLD,
        search_latency_ms=round(search_ms, 2),
        embedding_latency_ms=round(embed_ms, 2),
        total_latency_ms=round(total_ms, 2),
        quality=quality,
        enrolled_count=enrolled_count,
    )
