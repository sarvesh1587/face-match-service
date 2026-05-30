"""
API tests — FastAPI endpoints
==============================
Uses FastAPI's TestClient with mocked services.
These tests verify HTTP contract, status codes, and response schemas.
"""

import io
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import NoFaceDetectedError, MultipleFacesError, FaceQualityError
from app.models.schemas import FaceQualityReport


def _make_jpeg_bytes() -> bytes:
    img = np.ones((224, 224, 3), dtype=np.uint8) * 128
    _, enc = cv2.imencode(".jpg", img)
    return enc.tobytes()


def _make_mock_quality() -> FaceQualityReport:
    return FaceQualityReport(
        detection_score=0.97,
        blur_score=230.5,
        face_size_px=185,
        passed_all_gates=True,
    )


# ── App setup ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_app():
    """Create a FastAPI test app with all services mocked."""
    # Patch startup checks so we don't need real Qdrant / InsightFace
    with patch("app.core.startup.run_startup_checks", new_callable=AsyncMock):
        with patch("app.services.face_service.FaceService._load_model", return_value=MagicMock()):
            from app.main import app
            yield app


@pytest.fixture
def client(mock_app):
    # Inject mocked services into app state
    mock_face_service = MagicMock()
    mock_vector_store = MagicMock()
    mock_vector_store.count.return_value = 5
    mock_vector_store.health_check.return_value = {
        "status": "ok",
        "latency_ms": 1.2,
        "vectors_count": 5,
        "indexed_vectors_count": 5,
    }
    mock_app.state.face_service = mock_face_service
    mock_app.state.vector_store = mock_vector_store
    return TestClient(mock_app), mock_face_service, mock_vector_store


# ── Enroll tests ──────────────────────────────────────────────────────────────

class TestEnrollEndpoint:
    def test_successful_enroll(self, client):
        tc, face_svc, vec_store = client
        embedding = np.random.randn(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        face_svc.process.return_value = (embedding, _make_mock_quality())
        vec_store.search.return_value = ([], 1.0)
        vec_store.upsert.return_value = False  # new identity

        resp = tc.post(
            "/enroll",
            data={"identity_id": "alice_001"},
            files={"image": ("face.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["identity_id"] == "alice_001"
        assert body["embedding_dim"] == 512
        assert body["already_existed"] is False
        assert body["quality"]["detection_score"] == 0.97

    def test_enroll_no_face_returns_400(self, client):
        tc, face_svc, _ = client
        face_svc.process.side_effect = NoFaceDetectedError("No face detected.")
        resp = tc.post(
            "/enroll",
            data={"identity_id": "bob_002"},
            files={"image": ("blank.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 400
        assert "No face" in resp.json()["detail"]

    def test_enroll_multiple_faces_returns_400(self, client):
        tc, face_svc, _ = client
        face_svc.process.side_effect = MultipleFacesError("2 faces found.")
        resp = tc.post(
            "/enroll",
            data={"identity_id": "carol_003"},
            files={"image": ("multi.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 400

    def test_enroll_blurry_face_returns_400(self, client):
        tc, face_svc, _ = client
        face_svc.process.side_effect = FaceQualityError("Image too blurry.")
        resp = tc.post(
            "/enroll",
            data={"identity_id": "dave_004"},
            files={"image": ("blurry.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 400
        assert "blurry" in resp.json()["detail"].lower()

    def test_enroll_unsupported_content_type_returns_400(self, client):
        tc, _, _ = client
        resp = tc.post(
            "/enroll",
            data={"identity_id": "eve_005"},
            files={"image": ("doc.pdf", b"pdf content", "application/pdf")},
        )
        assert resp.status_code == 400
        assert "Unsupported image type" in resp.json()["detail"]

    def test_enroll_empty_identity_id_returns_422(self, client):
        tc, _, _ = client
        resp = tc.post(
            "/enroll",
            data={"identity_id": ""},
            files={"image": ("face.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 422

    def test_enroll_update_existing_identity(self, client):
        tc, face_svc, vec_store = client
        embedding = np.random.randn(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        face_svc.process.return_value = (embedding, _make_mock_quality())
        vec_store.search.return_value = ([], 1.0)
        vec_store.upsert.return_value = True  # identity existed

        resp = tc.post(
            "/enroll",
            data={"identity_id": "alice_001"},
            files={"image": ("face.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 201
        assert resp.json()["already_existed"] is True
        assert "updated" in resp.json()["message"].lower()


# ── Search tests ──────────────────────────────────────────────────────────────

class TestSearchEndpoint:
    def _make_candidate(self, identity_id: str, score: float):
        return {"identity_id": identity_id, "cosine_score": score, "metadata": {}}

    def test_successful_search_with_match(self, client):
        tc, face_svc, vec_store = client
        embedding = np.random.randn(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        face_svc.process.return_value = (embedding, _make_mock_quality())
        vec_store.search.return_value = (
            [self._make_candidate("alice_001", 0.82)],
            2.1,  # latency_ms
        )

        resp = tc.post(
            "/search",
            files={"image": ("query.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["top_match"]["identity_id"] == "alice_001"
        assert body["top_match"]["is_match"] is True
        assert body["top_match"]["cosine_score"] == pytest.approx(0.82, abs=1e-4)
        assert "query_id" in body
        assert body["search_latency_ms"] == pytest.approx(2.1, abs=0.1)

    def test_search_below_threshold_returns_no_match(self, client):
        tc, face_svc, vec_store = client
        embedding = np.random.randn(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        face_svc.process.return_value = (embedding, _make_mock_quality())
        vec_store.search.return_value = (
            [self._make_candidate("bob_002", 0.20)],
            1.5,
        )

        resp = tc.post(
            "/search",
            files={"image": ("query.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["top_match"] is None
        assert body["candidates"][0]["is_match"] is False

    def test_search_no_face_returns_400(self, client):
        tc, face_svc, _ = client
        face_svc.process.side_effect = NoFaceDetectedError("No face.")
        resp = tc.post(
            "/search",
            files={"image": ("empty.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 400

    def test_search_response_includes_latency_breakdown(self, client):
        tc, face_svc, vec_store = client
        embedding = np.random.randn(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
        face_svc.process.return_value = (embedding, _make_mock_quality())
        vec_store.search.return_value = ([], 1.0)

        resp = tc.post(
            "/search",
            files={"image": ("query.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "embedding_latency_ms" in body
        assert "search_latency_ms" in body
        assert "total_latency_ms" in body


# ── Health / Metrics tests ────────────────────────────────────────────────────

class TestOperationsEndpoints:
    def test_health_returns_200(self, client):
        tc, _, _ = client
        resp = tc.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("healthy", "degraded")
        assert "components" in body
        assert "qdrant" in body["components"]

    def test_metrics_returns_200(self, client):
        tc, _, _ = client
        resp = tc.get("/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert "total_searches" in body
        assert "search_latency" in body
        assert "p95_ms" in body["search_latency"]
