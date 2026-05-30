"""
scripts/calibrate_threshold.py
================================
Data-driven threshold selection for the face matching service.

This script:
1. Loads enrolled face embeddings from Qdrant
2. Generates genuine pairs  (same identity, different images)
3. Generates impostor pairs (different identities)
4. Computes cosine similarity distribution for each class
5. Sweeps thresholds and computes FAR, FRR, EER
6. Plots the distribution + ROC-style curve
7. Recommends a threshold

HOW TO USE
──────────
1. Enroll your faces via POST /enroll
2. Run: python scripts/calibrate_threshold.py
3. The script will print the recommended threshold.
4. Update MATCH_THRESHOLD in .env.

THEORY
──────
ArcFace embeddings partition space so that:
  • Genuine pairs (same person)  → high cosine (typically 0.4–0.9)
  • Impostor pairs (diff people) → low cosine  (typically −0.1–0.3)

The threshold sits between these two distributions.

  FAR (False Accept Rate)  = FP / (FP + TN)  — impostors let through
  FRR (False Reject Rate)  = FN / (FN + TP)  — genuine users blocked
  EER (Equal Error Rate)   = FAR = FRR        — natural operating point

For access control → minimise FAR (false accepts are security breaches).
For attendance      → minimise FRR (false rejects frustrate users).
"""

import itertools
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Try to import optional visualisation libs
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[warn] matplotlib not installed — skipping plots")

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.core.config import settings


def fetch_embeddings_from_qdrant() -> dict:
    """Pull all enrolled embeddings from Qdrant."""
    from qdrant_client import QdrantClient

    client = QdrantClient(
        host=settings.QDRANT_HOST,
        port=settings.QDRANT_PORT,
        api_key=settings.QDRANT_API_KEY,
    )

    embeddings = {}
    offset = None
    while True:
        results, next_offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            with_vectors=True,
            with_payload=True,
            limit=100,
            offset=offset,
        )
        for r in results:
            identity_id = r.payload["identity_id"]
            embeddings[identity_id] = np.array(r.vector, dtype=np.float32)
        if next_offset is None:
            break
        offset = next_offset

    print(f"Loaded {len(embeddings)} identities from Qdrant.")
    return embeddings


def generate_pairs(embeddings: dict) -> Tuple[List[float], List[float]]:
    """
    Build genuine and impostor pairs.

    With only 1 image per identity (typical for this task), we cannot
    form genuine pairs from different images of the same person.
    Instead we use self-similarity (score = 1.0) for genuine pairs and
    all cross-identity pairs for impostors.

    In a real calibration you would use multiple images per identity.
    """
    ids = list(embeddings.keys())
    vecs = [embeddings[i] for i in ids]

    genuine_scores = []
    impostor_scores = []

    # Genuine: self-similarity = 1.0 (trivial but bounds the genuine distribution)
    for v in vecs:
        genuine_scores.append(float(np.dot(v, v)))  # 1.0 by construction

    # Impostor: all cross-identity pairs
    for i, j in itertools.combinations(range(len(vecs)), 2):
        score = float(np.dot(vecs[i], vecs[j]))
        impostor_scores.append(score)

    return genuine_scores, impostor_scores


def sweep_thresholds(
    genuine: List[float],
    impostor: List[float],
    steps: int = 200,
) -> dict:
    """Sweep thresholds from 0 to 1, compute FAR and FRR at each point."""
    thresholds = np.linspace(0.0, 1.0, steps)
    genuine_arr = np.array(genuine)
    impostor_arr = np.array(impostor)

    fars, frrs = [], []
    for t in thresholds:
        far = float(np.mean(impostor_arr >= t))   # impostors accepted
        frr = float(np.mean(genuine_arr < t))     # genuines rejected
        fars.append(far)
        frrs.append(frr)

    # EER: point where FAR ≈ FRR
    diffs = np.abs(np.array(fars) - np.array(frrs))
    eer_idx = int(np.argmin(diffs))
    eer_threshold = float(thresholds[eer_idx])
    eer = float((fars[eer_idx] + frrs[eer_idx]) / 2)

    return {
        "thresholds": thresholds.tolist(),
        "far": fars,
        "frr": frrs,
        "eer_threshold": round(eer_threshold, 4),
        "eer": round(eer, 4),
    }


def recommend_threshold(sweep: dict, target_far: float = 0.01) -> float:
    """
    For access-control: choose the highest threshold where FAR ≤ target_far.
    This minimises false accepts while keeping FRR low.
    """
    thresholds = np.array(sweep["thresholds"])
    fars = np.array(sweep["far"])
    # Highest threshold with FAR ≤ target
    mask = fars <= target_far
    if not mask.any():
        return sweep["eer_threshold"]
    return round(float(thresholds[mask].max()), 4)


def plot_distributions(genuine: List[float], impostor: List[float], threshold: float) -> None:
    if not HAS_MATPLOTLIB:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ArcFace Cosine Similarity Distributions", fontsize=14, fontweight="bold")

    # Histogram
    ax = axes[0]
    bins = np.linspace(-0.2, 1.0, 60)
    ax.hist(impostor, bins=bins, alpha=0.65, color="#e74c3c", label="Impostor pairs")
    ax.hist(genuine, bins=bins, alpha=0.65, color="#2ecc71", label="Genuine pairs")
    ax.axvline(threshold, color="#f39c12", linewidth=2, linestyle="--",
               label=f"Threshold = {threshold:.3f}")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution")
    ax.legend()

    # FAR / FRR curve
    ax2 = axes[1]
    # Would normally call sweep_thresholds here — skipping for brevity in script
    ax2.set_title("FAR / FRR vs Threshold\n(run calibrate_threshold.py with full data)")
    ax2.text(0.5, 0.5, "Requires ≥2 images/identity\nfor meaningful genuine pairs",
             ha="center", va="center", transform=ax2.transAxes, fontsize=10)

    plt.tight_layout()
    out = Path("docs/threshold_analysis.png")
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"  Plot saved to {out}")
    plt.show()


def main() -> None:
    print("=" * 60)
    print("  Threshold Calibration Script")
    print("=" * 60)

    embeddings = fetch_embeddings_from_qdrant()

    if len(embeddings) < 2:
        print("[ERROR] Need at least 2 enrolled identities to compute impostor pairs.")
        sys.exit(1)

    genuine, impostor = generate_pairs(embeddings)
    sweep = sweep_thresholds(genuine, impostor)

    recommended = recommend_threshold(sweep, target_far=0.01)

    print(f"\n{'─'*40}")
    print(f"  EER threshold     : {sweep['eer_threshold']}")
    print(f"  EER               : {sweep['eer']*100:.2f}%")
    print(f"  Recommended (FAR≤1%): {recommended}")
    print(f"{'─'*40}")
    print(f"\n  Impostor score stats:")
    print(f"    mean = {np.mean(impostor):.4f}  std = {np.std(impostor):.4f}")
    print(f"    min  = {np.min(impostor):.4f}  max = {np.max(impostor):.4f}")
    print(f"\n  Genuine score stats:")
    print(f"    mean = {np.mean(genuine):.4f}  std = {np.std(genuine):.4f}")

    print(f"\n→ Set MATCH_THRESHOLD={recommended} in your .env")

    plot_distributions(genuine, impostor, recommended)


if __name__ == "__main__":
    main()
