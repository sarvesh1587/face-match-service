"""
Vector Store — Qdrant Integration
==================================

WHY QDRANT + ANN (NOT A PYTHON LOOP)
──────────────────────────────────────
A naïve loop computes cosine(query, stored_i) for every i.
At N identities that is O(N) distance calculations per query.
At 1 M identities: ~512 × 1 M = 512 M multiply-adds ≈ 200 ms on CPU.
At 10 M: 2 seconds. Unacceptable for a real-time service.

Qdrant's HNSW (Hierarchical Navigable Small World) graph reduces this to
O(log N) per query with ≥97 % recall at our scale, giving sub-5 ms search
even at millions of vectors.  The collection is configured to use cosine
distance so scores are directly interpretable as cosine similarity.

PAYLOAD DESIGN
──────────────
Each point stores:
  - vector: float32[512]  — the L2-normalised ArcFace embedding
  - payload:
      identity_id: str    — the caller-supplied identifier
      enrolled_at: str    — ISO 8601 timestamp
      metadata: dict      — arbitrary caller-supplied key-value pairs

We use identity_id as the Qdrant point ID (UUID-derived) for fast
upsert-or-update without scanning the collection first.
"""

import hashlib
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings
from app.core.exceptions import VectorStoreError
from app.core.logger import get_logger

logger = get_logger(__name__)


def _identity_to_point_id(identity_id: str) -> str:
    """
    Deterministically map an identity_id string to a UUID string.
    This makes upserts idempotent: enrolling the same identity_id twice
    overwrites the previous embedding rather than creating a duplicate point.
    """
    digest = hashlib.sha256(identity_id.encode()).hexdigest()
    return str(uuid.UUID(digest[:32]))


class VectorStore:
    """
    Async-friendly wrapper around the synchronous Qdrant client.
    All heavy I/O runs on the default thread-pool via asyncio.to_thread
    in the route handlers.  The client itself is thread-safe.
    """

    def __init__(self) -> None:
        self._client: Optional[QdrantClient] = None

    async def initialize(self) -> None:
        """Connect to Qdrant and ensure the collection exists."""
        self._client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
            timeout=10,
        )
        self._ensure_collection()
        count = self._client.count(collection_name=settings.QDRANT_COLLECTION).count
        logger.info(
            "VectorStore initialised",
            collection=settings.QDRANT_COLLECTION,
            enrolled_identities=count,
        )

    async def close(self) -> None:
        if self._client:
            self._client.close()

    # ── Public API ────────────────────────────────────────────────────────────

    def upsert(
        self,
        identity_id: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Insert or update an identity.
        Returns True if this was an update (identity already existed).
        """
        point_id = _identity_to_point_id(identity_id)
        existing = self._exists(point_id)

        payload = {
            "identity_id": identity_id,
            "enrolled_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

        try:
            self._client.upsert(
                collection_name=settings.QDRANT_COLLECTION,
                points=[
                    qdrant_models.PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload=payload,
                    )
                ],
            )
        except UnexpectedResponse as exc:
            raise VectorStoreError(f"Qdrant upsert failed: {exc}") from exc

        logger.info(
            "upsert_complete",
            identity_id=identity_id,
            point_id=point_id,
            updated=existing,
        )
        return existing

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        ANN search — returns top_k results sorted by cosine similarity (desc).

        Qdrant's cosine distance is computed on normalised vectors, so
        score = 1 − distance = cosine similarity.
        """
        t0 = time.perf_counter()
        try:
            hits = self._client.search(
                collection_name=settings.QDRANT_COLLECTION,
                query_vector=query_embedding.tolist(),
                limit=top_k,
                with_payload=True,
                score_threshold=None,  # return all top_k; we apply threshold in app layer
            )
        except UnexpectedResponse as exc:
            raise VectorStoreError(f"Qdrant search failed: {exc}") from exc

        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "qdrant_search_complete",
            hits=len(hits),
            latency_ms=round(latency_ms, 2),
        )

        return [
            {
                "identity_id": h.payload["identity_id"],
                "cosine_score": round(float(h.score), 6),
                "metadata": h.payload.get("metadata", {}),
                "enrolled_at": h.payload.get("enrolled_at"),
            }
            for h in hits
        ], latency_ms

    def count(self) -> int:
        """Total number of enrolled identities."""
        return self._client.count(collection_name=settings.QDRANT_COLLECTION).count

    def delete(self, identity_id: str) -> bool:
        """Delete an identity by ID. Returns True if it existed."""
        point_id = _identity_to_point_id(identity_id)
        if not self._exists(point_id):
            return False
        self._client.delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=qdrant_models.PointIdsList(points=[point_id]),
        )
        logger.info("identity_deleted", identity_id=identity_id)
        return True

    def health_check(self) -> Dict[str, Any]:
        """Ping Qdrant and return basic stats."""
        t0 = time.perf_counter()
        info = self._client.get_collection(settings.QDRANT_COLLECTION)
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "status": "ok",
            "latency_ms": round(latency_ms, 2),
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """Create the collection if it doesn't already exist."""
        existing = [c.name for c in self._client.get_collections().collections]
        if settings.QDRANT_COLLECTION in existing:
            logger.info("Collection exists", name=settings.QDRANT_COLLECTION)
            return

        self._client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=qdrant_models.VectorParams(
                size=settings.EMBEDDING_DIM,
                distance=qdrant_models.Distance.COSINE,
                # HNSW config — tuned for a small dataset; scales to millions
                hnsw_config=qdrant_models.HnswConfigDiff(
                    m=16,            # edges per node — higher = better recall, more RAM
                    ef_construct=200, # construction-time search width
                    full_scan_threshold=10_000,  # use brute-force below this count
                ),
            ),
        )
        logger.info("Collection created", name=settings.QDRANT_COLLECTION)

    def _exists(self, point_id: str) -> bool:
        results = self._client.retrieve(
            collection_name=settings.QDRANT_COLLECTION,
            ids=[point_id],
        )
        return len(results) > 0
