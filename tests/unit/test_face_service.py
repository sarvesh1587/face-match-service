"""
Unit tests — Face Service pipeline
====================================
Tests the ML pipeline components in isolation using mocked InsightFace.
These run without a GPU, Qdrant, or downloaded models.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.core.exceptions import (
    EmbeddingError,
    FaceQualityError,
    MultipleFacesError,
    NoFaceDetectedError,
)
from app.models.schemas import FaceQualityReport
from app.services.face_service import FaceService


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_mock_face(det_score: float = 0.95, embedding_dim: int = 512):
    """Create a mock InsightFace Face object."""
    face = MagicMock()
    face.det_score = det_score
    face.normed_embedding = np.random.randn(embedding_dim).astype(np.float32)
    face.normed_embedding /= np.linalg.norm(face.normed_embedding)
    face.bbox = np.array([10, 10, 200, 200], dtype=float)
    return face


def make_jpeg_bytes() -> bytes:
    """Minimal valid JPEG header + footer (won't render but passes cv2.imdecode)."""
    import cv2
    import numpy as np

    img = np.ones((224, 224, 3), dtype=np.uint8) * 128
    _, enc = cv2.imencode(".jpg", img)
    return enc.tobytes()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestFaceServiceNormalization:
    def test_l2_normalize_unit_vector(self):
        vec = np.array([3.0, 4.0], dtype=np.float32)
        normed = FaceService._l2_normalize(vec)
        assert abs(np.linalg.norm(normed) - 1.0) < 1e-6

    def test_l2_normalize_already_unit(self):
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        normed = FaceService._l2_normalize(vec)
        np.testing.assert_array_almost_equal(normed, vec)

    def test_l2_normalize_zero_vector_raises(self):
        with pytest.raises(EmbeddingError, match="Zero-norm"):
            FaceService._l2_normalize(np.zeros(512, dtype=np.float32))

    def test_normalized_cosine_equals_dot(self):
        a = np.random.randn(512).astype(np.float32)
        b = np.random.randn(512).astype(np.float32)
        a = FaceService._l2_normalize(a)
        b = FaceService._l2_normalize(b)
        cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        dot = np.dot(a, b)
        assert abs(cosine - dot) < 1e-5, "After normalisation, cosine must equal dot product"


class TestFaceQualityGates:
    def test_low_detection_score_raises(self):
        from app.services.face_service import FaceService
        quality = FaceQualityReport(
            detection_score=0.50,  # below MIN_DETECTION_SCORE=0.70
            blur_score=200.0,
            face_size_px=120,
            passed_all_gates=False,
        )
        with pytest.raises(FaceQualityError, match="Low detection confidence"):
            FaceService._assert_quality(quality)

    def test_small_face_raises(self):
        quality = FaceQualityReport(
            detection_score=0.95,
            blur_score=200.0,
            face_size_px=30,  # below MIN_FACE_SIZE=60
            passed_all_gates=False,
        )
        with pytest.raises(FaceQualityError, match="Face too small"):
            FaceService._assert_quality(quality)

    def test_blurry_face_raises(self):
        quality = FaceQualityReport(
            detection_score=0.95,
            blur_score=10.0,  # below BLUR_LAPLACIAN_THRESHOLD=80
            face_size_px=120,
            passed_all_gates=False,
        )
        with pytest.raises(FaceQualityError, match="blurry"):
            FaceService._assert_quality(quality)

    def test_good_quality_passes(self):
        quality = FaceQualityReport(
            detection_score=0.95,
            blur_score=200.0,
            face_size_px=120,
            passed_all_gates=True,
        )
        # Should not raise
        FaceService._assert_quality(quality)


class TestFaceDetection:
    @patch("app.services.face_service.FaceService._load_model")
    def test_no_face_raises(self, mock_load):
        mock_app = MagicMock()
        mock_app.get.return_value = []
        mock_load.return_value = mock_app

        svc = FaceService()
        image_bytes = make_jpeg_bytes()

        with pytest.raises(NoFaceDetectedError):
            svc.process(image_bytes)

    @patch("app.services.face_service.FaceService._load_model")
    def test_multiple_faces_raises_on_enroll(self, mock_load):
        mock_app = MagicMock()
        mock_app.get.return_value = [make_mock_face(), make_mock_face()]
        mock_load.return_value = mock_app

        svc = FaceService()
        image_bytes = make_jpeg_bytes()

        with pytest.raises(MultipleFacesError):
            svc.process(image_bytes, allow_multiple=False)

    @patch("app.services.face_service.FaceService._load_model")
    @patch("app.services.face_service.FaceService._compute_quality")
    def test_multiple_faces_allowed_on_search(self, mock_quality, mock_load):
        face_a = make_mock_face(det_score=0.90)
        face_b = make_mock_face(det_score=0.95)  # higher score → picked
        mock_app = MagicMock()
        mock_app.get.return_value = [face_a, face_b]
        mock_load.return_value = mock_app
        mock_quality.return_value = FaceQualityReport(
            detection_score=0.95, blur_score=200.0, face_size_px=150, passed_all_gates=True
        )

        svc = FaceService()
        image_bytes = make_jpeg_bytes()
        embedding, quality = svc.process(image_bytes, allow_multiple=True)

        assert embedding.shape == (512,)
        assert abs(np.linalg.norm(embedding) - 1.0) < 1e-5

    @patch("app.services.face_service.FaceService._load_model")
    @patch("app.services.face_service.FaceService._compute_quality")
    def test_embedding_is_unit_norm(self, mock_quality, mock_load):
        mock_app = MagicMock()
        mock_app.get.return_value = [make_mock_face()]
        mock_load.return_value = mock_app
        mock_quality.return_value = FaceQualityReport(
            detection_score=0.95, blur_score=200.0, face_size_px=150, passed_all_gates=True
        )

        svc = FaceService()
        embedding, _ = svc.process(make_jpeg_bytes())
        assert abs(np.linalg.norm(embedding) - 1.0) < 1e-5, "Embedding must be L2-normalised"

    def test_decode_invalid_bytes_raises(self):
        with pytest.raises(FaceQualityError, match="Could not decode"):
            FaceService._decode_image(b"not_an_image")


class TestConfidenceBands:
    def test_high_confidence(self):
        from app.utils.confidence import classify
        band = classify(0.85)
        assert band.label == "HIGH"

    def test_near_threshold(self):
        from app.utils.confidence import classify
        band = classify(0.42)
        assert band.label == "NEAR-THRESHOLD"

    def test_very_low(self):
        from app.utils.confidence import classify
        band = classify(0.05)
        assert band.label == "VERY LOW"
