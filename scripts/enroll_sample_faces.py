"""
scripts/enroll_sample_faces.py
================================
Bulk-enrolls the downloaded sample faces against a running service.

Usage:
    # 1. Start the service
    docker compose up -d

    # 2. Download sample faces
    python scripts/get_sample_faces.py

    # 3. Enroll + query
    python scripts/enroll_sample_faces.py

    # Or against the deployed URL:
    BASE_URL=https://your-app.onrender.com python scripts/enroll_sample_faces.py
"""

import os
import sys
import time
from pathlib import Path

import requests

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
DATA_DIR = Path("data/sample_faces")

IDENTITIES = [
    ("george_w_bush", DATA_DIR / "george_w_bush" / "George_W_Bush_0001.jpg"),
    ("tony_blair",    DATA_DIR / "tony_blair"    / "Tony_Blair_0001.jpg"),
    ("ariel_sharon",  DATA_DIR / "ariel_sharon"  / "Ariel_Sharon_0001.jpg"),
    ("colin_powell",  DATA_DIR / "colin_powell"  / "Colin_Powell_0001.jpg"),
    ("hugo_chavez",   DATA_DIR / "hugo_chavez"   / "Hugo_Chavez_0001.jpg"),
]

QUERY_IMAGE = DATA_DIR / "query" / "george_w_bush_George_W_Bush_0002.jpg"


def enroll(identity_id: str, image_path: Path) -> dict:
    if not image_path.exists():
        print(f"  [SKIP] {image_path} not found — run get_sample_faces.py first")
        return {}
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/enroll",
            data={"identity_id": identity_id},
            files={"image": (image_path.name, f, "image/jpeg")},
            timeout=30,
        )
    resp.raise_for_status()
    return resp.json()


def search(image_path: Path) -> dict:
    with open(image_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/search",
            files={"image": (image_path.name, f, "image/jpeg")},
            timeout=30,
        )
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    print(f"\nTarget: {BASE_URL}")
    print("=" * 60)

    # ── Health check ──────────────────────────────────────────────────────────
    print("\n[0] Health check")
    h = requests.get(f"{BASE_URL}/health", timeout=10).json()
    print(f"  Status: {h['status']}")

    # ── Enroll 5 identities ───────────────────────────────────────────────────
    print("\n[1] Enrolling 5 identities")
    for identity_id, image_path in IDENTITIES:
        print(f"\n  Enrolling: {identity_id}")
        try:
            result = enroll(identity_id, image_path)
            q = result.get("quality", {})
            print(f"    ✓ enrolled_at: {result.get('enrolled_at')}")
            print(f"    ✓ detection_score: {q.get('detection_score')}")
            print(f"    ✓ blur_score: {q.get('blur_score')}")
            print(f"    ✓ face_size_px: {q.get('face_size_px')}")
        except requests.HTTPError as exc:
            print(f"    ✗ {exc.response.status_code}: {exc.response.text}")

    # ── Query ─────────────────────────────────────────────────────────────────
    print("\n[2] Running query (George W. Bush — different image from enrollment)")
    time.sleep(0.5)
    try:
        result = search(QUERY_IMAGE)
        top = result.get("top_match")
        print(f"\n  Query ID: {result['query_id']}")
        print(f"  Total latency: {result['total_latency_ms']} ms")
        print(f"    • embedding: {result['embedding_latency_ms']} ms")
        print(f"    • qdrant ANN: {result['search_latency_ms']} ms")

        if top:
            print(f"\n  ✓ TOP MATCH: {top['identity_id']}")
            print(f"    cosine_score : {top['cosine_score']}")
            print(f"    is_match     : {top['is_match']}")
            print(f"    confidence   : {top['confidence_band']['label']}")
        else:
            print("\n  ✗ No match above threshold")

        print("\n  All candidates:")
        for c in result["candidates"]:
            marker = "✓" if c["is_match"] else "✗"
            print(f"    {marker} {c['identity_id']:<20} cosine={c['cosine_score']:.4f}  [{c['confidence_band']['label']}]")

    except Exception as exc:
        print(f"  ✗ Search failed: {exc}")

    # ── Hard negative pair ────────────────────────────────────────────────────
    print("\n[3] Hard-negative analysis: george_w_bush ↔ tony_blair")
    george_path = DATA_DIR / "george_w_bush" / "George_W_Bush_0001.jpg"
    tony_path   = DATA_DIR / "tony_blair"    / "Tony_Blair_0001.jpg"

    if george_path.exists() and tony_path.exists():
        # Search tony_blair image against enrolled identities
        result = search(tony_path)
        candidates = {c["identity_id"]: c for c in result["candidates"]}
        george_hit = candidates.get("george_w_bush")
        tony_hit   = candidates.get("tony_blair")

        print(f"  Query: tony_blair image")
        if george_hit:
            print(f"  george_w_bush cosine: {george_hit['cosine_score']:.4f}  match={george_hit['is_match']}")
        if tony_hit:
            print(f"  tony_blair cosine:    {tony_hit['cosine_score']:.4f}  match={tony_hit['is_match']}")

        threshold = result["threshold_used"]
        print(f"\n  Threshold used: {threshold}")
        print(f"  Verdict: {'CORRECTLY REJECTED (FAR avoided)' if not (george_hit and george_hit['is_match']) else 'FALSE ACCEPT — raise threshold'}")

    print("\n✓ Done. Fill RESULTS.md with the numbers above.\n")


if __name__ == "__main__":
    main()
