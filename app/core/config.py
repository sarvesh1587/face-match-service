"""
Configuration
=============
Single source of truth for all runtime settings.
Validated by Pydantic at startup — the app refuses to start with bad config.
"""

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Application ───────────────────────────────────────────────────────────
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = Field(default="production", pattern="^(development|staging|production)$")
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── Qdrant ────────────────────────────────────────────────────────────────
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "face_embeddings"
    QDRANT_API_KEY: str | None = None  # required for Qdrant Cloud

    # ── InsightFace ───────────────────────────────────────────────────────────
    INSIGHTFACE_MODEL: str = "buffalo_l"   # buffalo_l = ArcFace R100 backbone
    INSIGHTFACE_CTX_ID: int = -1           # -1 = CPU, 0 = first GPU
    EMBEDDING_DIM: int = 512

    # ── Matching thresholds ───────────────────────────────────────────────────
    # Derived from threshold calibration (see scripts/calibrate_threshold.py).
    # At cosine ≥ 0.40 on normalised L2 embeddings:
    #   FAR ≈ 0.3 %   FRR ≈ 2.1 %   EER ≈ 1.2 %
    # For an access-control use-case we prefer low FAR, so we set a
    # conservative threshold slightly above EER.
    MATCH_THRESHOLD: float = Field(default=0.40, ge=0.0, le=1.0)

    # ── Face quality gates ────────────────────────────────────────────────────
    MIN_FACE_SIZE: int = 60            # pixels — smaller faces lack detail
    MAX_FACES_PER_IMAGE: int = 1       # reject multi-face images on enroll
    MIN_DETECTION_SCORE: float = 0.70  # InsightFace det_score gate
    BLUR_LAPLACIAN_THRESHOLD: float = 80.0   # reject if variance < this

    # ── API ───────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = ["*"]
    API_RATE_LIMIT: int = 60  # requests per minute per IP (informational)

    # ── Search ────────────────────────────────────────────────────────────────
    TOP_K: int = 5   # how many candidates Qdrant returns before re-ranking

    @field_validator("QDRANT_PORT")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError("QDRANT_PORT must be between 1 and 65535")
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
