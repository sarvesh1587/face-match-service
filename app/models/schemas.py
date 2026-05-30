"""
API Models
==========
Pydantic v2 schemas for all request and response objects.
Explicit field descriptions populate the OpenAPI docs automatically.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Shared ────────────────────────────────────────────────────────────────────

class ConfidenceBand(BaseModel):
    label: str = Field(description="Human-readable confidence tier")
    min_score: float
    max_score: float
    description: str


# ── Enroll ────────────────────────────────────────────────────────────────────

class EnrollRequest(BaseModel):
    """
    Multipart data is handled at the FastAPI layer (UploadFile).
    This schema covers the optional metadata fields sent alongside the image.
    """
    identity_id: str = Field(
        description="Stable identifier for this person (e.g. employee_042).",
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_\-]+$",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Arbitrary key-value pairs stored alongside the embedding.",
    )

    @field_validator("identity_id")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class FaceQualityReport(BaseModel):
    detection_score: float = Field(description="InsightFace face detection confidence [0, 1]")
    blur_score: float = Field(description="Laplacian variance — higher is sharper")
    face_size_px: int = Field(description="Bounding-box diagonal in pixels")
    passed_all_gates: bool


class EnrollResponse(BaseModel):
    identity_id: str
    embedding_dim: int = Field(description="Length of the ArcFace embedding vector")
    quality: FaceQualityReport
    enrolled_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = "Identity enrolled successfully"
    already_existed: bool = Field(
        default=False,
        description="True if the embedding was updated (identity previously existed).",
    )


# ── Search ────────────────────────────────────────────────────────────────────

class SearchCandidate(BaseModel):
    identity_id: str
    cosine_score: float = Field(description="Cosine similarity in [-1, 1]; higher is more similar")
    is_match: bool = Field(description="True when cosine_score ≥ configured MATCH_THRESHOLD")
    confidence_band: ConfidenceBand
    metadata: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    query_id: str = Field(description="Server-generated UUID for this search request")
    top_match: Optional[SearchCandidate] = Field(
        description="Best matching identity, or null if no face was found above threshold."
    )
    candidates: List[SearchCandidate] = Field(
        description="All top-K candidates returned by Qdrant ANN, ranked by score."
    )
    threshold_used: float
    search_latency_ms: float
    embedding_latency_ms: float
    total_latency_ms: float
    quality: FaceQualityReport
    enrolled_count: int = Field(description="Total identities in the store at query time")


# ── Health ────────────────────────────────────────────────────────────────────

class ComponentHealth(BaseModel):
    status: str  # "ok" | "degraded" | "down"
    latency_ms: Optional[float] = None
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str  # "healthy" | "degraded" | "unhealthy"
    version: str
    uptime_seconds: float
    components: Dict[str, ComponentHealth]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Metrics ───────────────────────────────────────────────────────────────────

class LatencyStats(BaseModel):
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    count: int


class MetricsResponse(BaseModel):
    enrolled_identities: int
    total_searches: int
    total_enrollments: int
    search_latency: LatencyStats
    enroll_latency: LatencyStats
    match_rate: float = Field(description="Fraction of searches that returned is_match=True")
    uptime_seconds: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Error ─────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str
    error_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
