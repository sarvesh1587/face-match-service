"""
Domain Exceptions
=================
Typed exceptions allow API handlers to return precise HTTP status codes
and error messages without coupling business logic to HTTP.
"""


class FaceMatchError(Exception):
    """Base class for all Face Match service errors."""


class NoFaceDetectedError(FaceMatchError):
    """Image contains no detectable face."""


class MultipleFacesError(FaceMatchError):
    """Image contains more than one face (ambiguous enroll)."""


class FaceQualityError(FaceMatchError):
    """Face detected but quality gates failed (blur, size, score)."""


class DuplicateFaceError(FaceMatchError):
    """
    The face being enrolled is already present in the store.
    Cosine similarity to the nearest existing embedding exceeded the threshold.
    """

    def __init__(self, existing_id: str, similarity: float):
        self.existing_id = existing_id
        self.similarity = similarity
        super().__init__(
            f"Face is a likely duplicate of enrolled identity '{existing_id}' "
            f"(cosine={similarity:.4f})"
        )


class IdentityNotFoundError(FaceMatchError):
    """Requested identity ID does not exist in the store."""


class VectorStoreError(FaceMatchError):
    """Qdrant operation failed."""


class EmbeddingError(FaceMatchError):
    """InsightFace failed to produce an embedding."""
