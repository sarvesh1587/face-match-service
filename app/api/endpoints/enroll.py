"""
POST /enroll
============
Accepts a face image + identity_id, runs the full ML pipeline,
stores the embedding in Qdrant, returns a quality report.
"""

import asyncio
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status

from app.core.config import settings
from app.core.exceptions import (
    DuplicateFaceError,
    EmbeddingError,
    FaceQualityError,
    MultipleFacesError,
    NoFaceDetectedError,
    VectorStoreError,
)
from app.core.logger import get_logger
from app.models.schemas import EnrollResponse, ErrorResponse
from app.services.metrics import metrics

logger = get_logger(__name__)
router = APIRouter()

_SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


@router.post(
    "/enroll",
    response_model=EnrollResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Image validation failed"},
        409: {"model": ErrorResponse, "description": "Duplicate face detected"},
        422: {"model": ErrorResponse, "description": "Request validation error"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    summary="Enroll a face identity",
    description=(
        "Upload a face image and a stable identity_id. "
        "The service detects and aligns the face, extracts an ArcFace embedding, "
        "checks for duplicates, and stores the vector in Qdrant. "
        "Re-enrolling the same identity_id overwrites the previous embedding."
    ),
)
async def enroll_face(
    request: Request,
    identity_id: str = Form(
        ...,
        description="Stable identifier (alphanumeric + _ -), e.g. 'employee_042'",
        min_length=1,
        max_length=128,
    ),
    image: UploadFile = File(..., description="Face image (JPEG/PNG/WebP)"),
    check_duplicate: bool = Form(
        default=True,
        description=(
            "When True, reject if a very similar face is already enrolled. "
            "Set False to force re-enroll."
        ),
    ),
    metadata: Optional[str] = Form(
        default=None,
        description="Optional JSON string of arbitrary metadata to store with the embedding.",
    ),
) -> EnrollResponse:
    t0 = time.perf_counter()

    # ── Content-type validation ───────────────────────────────────────────────
    ct = (image.content_type or "").split(";")[0].strip()
    if ct not in _SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image type '{ct}'. Use JPEG, PNG, or WebP.",
        )

    # ── Parse optional metadata JSON ──────────────────────────────────────────
    parsed_metadata = {}
    if metadata:
        import json
        try:
            parsed_metadata = json.loads(metadata)
            if not isinstance(parsed_metadata, dict):
                raise ValueError("metadata must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid metadata JSON: {exc}",
            )

    # ── Read image bytes ──────────────────────────────────────────────────────
    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    face_service = request.app.state.face_service
    vector_store = request.app.state.vector_store

    try:
        # Run the blocking ML pipeline in a thread to not block the event loop
        embedding, quality = await asyncio.to_thread(
            face_service.process, image_bytes, False  # allow_multiple=False for enroll
        )

        # ── Duplicate detection ───────────────────────────────────────────────
        if check_duplicate:
            candidates, _ = await asyncio.to_thread(
                vector_store.search, embedding, 1
            )
            if candidates:
                top = candidates[0]
                if (
                    top["cosine_score"] >= settings.MATCH_THRESHOLD
                    and top["identity_id"] != identity_id
                ):
                    raise DuplicateFaceError(
                        existing_id=top["identity_id"],
                        similarity=top["cosine_score"],
                    )

        # ── Store in Qdrant ───────────────────────────────────────────────────
        already_existed = await asyncio.to_thread(
            vector_store.upsert, identity_id, embedding, parsed_metadata
        )

    except (NoFaceDetectedError, MultipleFacesError, FaceQualityError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except DuplicateFaceError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    except (EmbeddingError, VectorStoreError) as exc:
        logger.error("enroll_pipeline_error", error=str(exc), identity_id=identity_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    metrics.record_enroll(elapsed_ms)

    logger.info(
        "enroll_success",
        identity_id=identity_id,
        already_existed=already_existed,
        elapsed_ms=round(elapsed_ms, 2),
    )

    return EnrollResponse(
        identity_id=identity_id,
        embedding_dim=len(embedding),
        quality=quality,
        already_existed=already_existed,
        message="Identity updated" if already_existed else "Identity enrolled successfully",
    )
