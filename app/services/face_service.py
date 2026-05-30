"""
Face Service — ML Pipeline
==========================

Step 1: Decode image bytes → numpy BGR array
Step 2: Detect faces with InsightFace (RetinaFace detector)
Step 3: Reject if 0 faces or >1 face (configurable)
Step 4: Run quality gates (blur, size, detection score)
Step 5: Extract ArcFace embedding (alignment is automatic inside InsightFace)
Step 6: L2-normalise the embedding so cosine ≡ dot product
Step 7: Return embedding + quality report

WHY ALIGNMENT MATTERS
─────────────────────
ArcFace was trained on aligned faces (5-point landmark warp to a canonical
96×112 template).  Feeding a tilted or off-centre face breaks the spatial
assumptions of the convolutional layers — the nose appears where the model
expects eyes — and the resulting embedding can shift by 0.1–0.3 cosine
units, enough to miss genuine matches.  InsightFace bundles alignment
inside its `get` method, so we get it for free.

WHY L2-NORMALISATION MATTERS
─────────────────────────────
Cosine similarity = dot(a, b) / (‖a‖ · ‖b‖).
If vectors are already unit-norm, cosine = dot product, which is what
Qdrant's cosine distance computes internally.  Skipping normalisation
means vectors with different magnitudes get unfair weight, and the
similarity scores are no longer in a stable [−1, 1] range.
"""

import time
from typing import Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.core.exceptions import (
    EmbeddingError,
    FaceQualityError,
    MultipleFacesError,
    NoFaceDetectedError,
)
from app.core.logger import get_logger
from app.models.schemas import FaceQualityReport

logger = get_logger(__name__)


class FaceService:
    """
    Thread-safe face pipeline wrapper around InsightFace.

    The InsightFace model is heavyweight (~300 MB download) so we load it
    exactly once at startup and reuse the same instance across requests.
    InsightFace's `get` method is CPU-thread-safe when CTX_ID=-1.
    """

    def __init__(self) -> None:
        self._app = self._load_model()
        logger.info(
            "InsightFace model loaded",
            model=settings.INSIGHTFACE_MODEL,
            ctx_id=settings.INSIGHTFACE_CTX_ID,
        )

    def _load_model(self):
        """
        Load InsightFace FaceAnalysis app.
        allowed_modules=['detection', 'recognition'] skips the landmark
        and attribute models we don't need, saving ~150 MB of RAM.
        """
        import insightface
        from insightface.app import FaceAnalysis

        fa = FaceAnalysis(
            name=settings.INSIGHTFACE_MODEL,
            allowed_modules=["detection", "recognition"],
        )
        fa.prepare(ctx_id=settings.INSIGHTFACE_CTX_ID, det_size=(640, 640))
        return fa

    # ── Public API ────────────────────────────────────────────────────────────

    def process(
        self,
        image_bytes: bytes,
        allow_multiple: bool = False,
    ) -> Tuple[np.ndarray, FaceQualityReport]:
        """
        Full pipeline: bytes → normalised embedding + quality report.

        Args:
            image_bytes: Raw bytes of the uploaded image (JPEG/PNG/WebP).
            allow_multiple: If True, uses the largest face when >1 detected.
                            If False (enroll mode), raises MultipleFacesError.

        Returns:
            (embedding, quality_report)

        Raises:
            NoFaceDetectedError, MultipleFacesError, FaceQualityError, EmbeddingError
        """
        t0 = time.perf_counter()

        bgr = self._decode_image(image_bytes)
        faces = self._detect_faces(bgr)

        # ── Face count validation ─────────────────────────────────────────────
        if len(faces) == 0:
            raise NoFaceDetectedError("No face detected in the provided image.")

        if len(faces) > 1 and not allow_multiple:
            raise MultipleFacesError(
                f"Image contains {len(faces)} faces. "
                "Enroll one face at a time to avoid ambiguity."
            )

        # Pick the face with the highest detection score (largest if >1 allowed)
        face = max(faces, key=lambda f: f.det_score)

        # ── Quality gates ─────────────────────────────────────────────────────
        quality = self._compute_quality(bgr, face)
        self._assert_quality(quality)

        # ── Embedding + normalisation ─────────────────────────────────────────
        embedding = self._extract_embedding(face)
        embedding = self._l2_normalize(embedding)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "face_pipeline_complete",
            detection_score=round(float(face.det_score), 4),
            blur_score=round(quality.blur_score, 2),
            face_size_px=quality.face_size_px,
            elapsed_ms=round(elapsed_ms, 2),
        )

        return embedding, quality

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        """Decode image bytes to BGR numpy array."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FaceQualityError(
                "Could not decode image. Ensure the file is a valid JPEG, PNG, or WebP."
            )
        return bgr

    def _detect_faces(self, bgr: np.ndarray):
        """Run RetinaFace detection + ArcFace recognition in one call."""
        try:
            faces = self._app.get(bgr)
        except Exception as exc:
            logger.error("InsightFace detection failed", error=str(exc))
            raise EmbeddingError(f"Face detection error: {exc}") from exc
        return faces

    def _compute_quality(self, bgr: np.ndarray, face) -> FaceQualityReport:
        """
        Compute quality metrics from the detected face.

        Blur detection: Laplacian variance.
        A perfectly sharp image has high variance; a blurry one approaches 0.
        Threshold 80 was chosen empirically — faces below this are typically
        motion-blurred or heavily compressed.
        """
        # Bounding box
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), bbox[2], bbox[3]
        face_crop = bgr[y1:y2, x1:x2]

        # Blur: Laplacian variance on greyscale crop
        if face_crop.size == 0:
            blur_score = 0.0
        else:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Face diagonal as proxy for size
        w, h = x2 - x1, y2 - y1
        face_size_px = int((w ** 2 + h ** 2) ** 0.5)

        passed = (
            float(face.det_score) >= settings.MIN_DETECTION_SCORE
            and face_size_px >= settings.MIN_FACE_SIZE
            and blur_score >= settings.BLUR_LAPLACIAN_THRESHOLD
        )

        return FaceQualityReport(
            detection_score=round(float(face.det_score), 4),
            blur_score=round(blur_score, 2),
            face_size_px=face_size_px,
            passed_all_gates=passed,
        )

    @staticmethod
    def _assert_quality(quality: FaceQualityReport) -> None:
        """Raise FaceQualityError with a precise reason if any gate fails."""
        if quality.detection_score < settings.MIN_DETECTION_SCORE:
            raise FaceQualityError(
                f"Low detection confidence ({quality.detection_score:.2f} < "
                f"{settings.MIN_DETECTION_SCORE}). Use a clearer image."
            )
        if quality.face_size_px < settings.MIN_FACE_SIZE:
            raise FaceQualityError(
                f"Face too small ({quality.face_size_px}px diagonal < "
                f"{settings.MIN_FACE_SIZE}px minimum). Move closer or use a higher-res image."
            )
        if quality.blur_score < settings.BLUR_LAPLACIAN_THRESHOLD:
            raise FaceQualityError(
                f"Image too blurry (Laplacian variance={quality.blur_score:.1f} < "
                f"{settings.BLUR_LAPLACIAN_THRESHOLD}). Use a sharper image."
            )

    @staticmethod
    def _extract_embedding(face) -> np.ndarray:
        """Pull the ArcFace embedding from the InsightFace Face object."""
        emb = face.normed_embedding  # InsightFace already L2-normalises this
        if emb is None or emb.size == 0:
            raise EmbeddingError(
                "InsightFace returned an empty embedding. "
                "This may happen if the recognition model was not loaded."
            )
        return emb.astype(np.float32)

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        """
        Explicitly re-normalise.
        InsightFace's normed_embedding is already unit-norm, but we
        re-apply to guard against floating-point drift and make the
        contract explicit in our pipeline.
        """
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            raise EmbeddingError("Zero-norm embedding — cannot normalise.")
        return vec / norm
