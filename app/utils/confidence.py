"""
Confidence Bands
================
Translates a raw cosine score into a human-readable confidence tier.
These bands are calibrated to match the threshold distribution observed
during threshold_calibration.py analysis.

Score ranges (cosine similarity on L2-normalised ArcFace embeddings):
  ≥ 0.70  → HIGH CONFIDENCE   (genuine pairs almost never fall here for impostors)
  0.50–0.69 → MEDIUM-HIGH     (likely same person, some look-alikes may appear)
  0.40–0.49 → NEAR-THRESHOLD  (ambiguous; manual review recommended)
  0.20–0.39 → LOW             (probably different people)
  < 0.20   → VERY LOW         (clearly different people)
"""

from app.models.schemas import ConfidenceBand

_BANDS = [
    ConfidenceBand(
        label="HIGH",
        min_score=0.70,
        max_score=1.00,
        description="Very high confidence — almost certainly the same person.",
    ),
    ConfidenceBand(
        label="MEDIUM-HIGH",
        min_score=0.50,
        max_score=0.70,
        description="High confidence — likely the same person.",
    ),
    ConfidenceBand(
        label="NEAR-THRESHOLD",
        min_score=0.40,
        max_score=0.50,
        description="Near decision boundary — consider manual review.",
    ),
    ConfidenceBand(
        label="LOW",
        min_score=0.20,
        max_score=0.40,
        description="Low confidence — probably different people.",
    ),
    ConfidenceBand(
        label="VERY LOW",
        min_score=-1.00,
        max_score=0.20,
        description="Very low confidence — clearly different people.",
    ),
]


def classify(score: float) -> ConfidenceBand:
    """Return the confidence band for a given cosine score."""
    for band in _BANDS:
        if score >= band.min_score:
            return band
    return _BANDS[-1]
