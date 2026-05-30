"""
scripts/get_sample_faces.py
============================
Downloads 5 identities + 1 query face from the public LFW (Labeled Faces
in the Wild) dataset via the University of Massachusetts mirror.

Usage:
    python scripts/get_sample_faces.py

Output:
    data/sample_faces/
        ├── george_bush/        ← 1 enrollment image
        ├── tony_blair/
        ├── ariel_sharon/
        ├── colin_powell/
        ├── hugo_chavez/
        └── query/
            └── george_bush_query.jpg   ← different image of same person

Hard-negative pair:
    george_bush vs tony_blair  — both caucasian males, similar age, common
    suit-and-tie context. Selected because look-alike pairs stress-test the
    threshold more than obviously different people.
"""

import os
import shutil
import urllib.request
from pathlib import Path

# LFW base URL (public, no auth required)
LFW_BASE = "https://vis-www.cs.umass.edu/lfw/lfw-deepfunneled"

# identity_name → [(filename, use)]
# use: "enroll" or "query"
FACES = {
    "George_W_Bush": [
        ("George_W_Bush_0001.jpg", "enroll"),
        ("George_W_Bush_0002.jpg", "query"),
    ],
    "Tony_Blair": [
        ("Tony_Blair_0001.jpg", "enroll"),
    ],
    "Ariel_Sharon": [
        ("Ariel_Sharon_0001.jpg", "enroll"),
    ],
    "Colin_Powell": [
        ("Colin_Powell_0001.jpg", "enroll"),
    ],
    "Hugo_Chavez": [
        ("Hugo_Chavez_0001.jpg", "enroll"),
    ],
}

DATA_DIR = Path("data/sample_faces")


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return
    print(f"  [download] {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        print(f"  [ERROR] Could not download {url}: {exc}")
        print("  → Try running: pip install requests and use the requests-based fallback below.")
        raise


def main() -> None:
    print("Downloading LFW sample faces...\n")

    query_dir = DATA_DIR / "query"
    query_dir.mkdir(parents=True, exist_ok=True)

    for identity, images in FACES.items():
        identity_dir = DATA_DIR / identity.lower()
        identity_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{identity}]")
        for filename, use in images:
            url = f"{LFW_BASE}/{identity}/{filename}"
            if use == "enroll":
                dest = identity_dir / filename
            else:
                dest = query_dir / f"{identity.lower()}_{filename}"
            download_file(url, dest)

    print("\n✓ Done. Files saved to:", DATA_DIR)
    print("\nEnrollment identities:")
    for identity in FACES:
        print(f"  • {identity.lower()}")
    print("\nQuery image: data/sample_faces/query/george_w_bush_george_w_bush_0002.jpg")
    print("\nHard-negative pair: george_w_bush ↔ tony_blair")
    print("  Both: caucasian male, ~55–65 yrs, suit-and-tie, formal setting")
    print("  ArcFace cosine typically: 0.25–0.35 (below threshold=0.40)")


if __name__ == "__main__":
    main()
