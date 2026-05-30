# RESULTS.md — Face Match Service

> Fill this file with real measured numbers after running `python scripts/enroll_sample_faces.py`.
> Template values shown below — replace with your actual output.

---

## 1. Enrollment Results

| Identity | Detection Score | Blur Score | Face Size (px) | Enrolled At |
|----------|----------------|------------|----------------|-------------|
| george_w_bush | 0.9923 | 318.4 | 202 | 2024-01-15T10:30:01Z |
| tony_blair | 0.9871 | 291.7 | 195 | 2024-01-15T10:30:03Z |
| ariel_sharon | 0.9845 | 274.2 | 188 | 2024-01-15T10:30:05Z |
| colin_powell | 0.9902 | 305.8 | 197 | 2024-01-15T10:30:07Z |
| hugo_chavez | 0.9834 | 261.3 | 182 | 2024-01-15T10:30:09Z |

All 5 identities passed quality gates:
- detection_score ≥ 0.70 ✓
- blur_score ≥ 80.0 ✓
- face_size_px ≥ 60 ✓

---

## 2. Search Results

**Query image:** `data/sample_faces/query/george_w_bush_George_W_Bush_0002.jpg`
(A different photograph of George W. Bush from the enrollment image)

| Rank | Identity | Cosine Score | Is Match | Confidence |
|------|----------|-------------|----------|------------|
| 1 | **george_w_bush** | **0.7823** | ✓ YES | HIGH |
| 2 | colin_powell | 0.2941 | ✗ NO | LOW |
| 3 | tony_blair | 0.2712 | ✗ NO | LOW |
| 4 | ariel_sharon | 0.2384 | ✗ NO | LOW |
| 5 | hugo_chavez | 0.2198 | ✗ NO | VERY LOW |

**Correct top match: `george_w_bush` ✓**

The margin between rank 1 (0.782) and rank 2 (0.294) is 0.488 cosine units — well above the threshold, indicating high confidence.

---

## 3. Latency Table

Measured on: [YOUR MACHINE / CLOUD INSTANCE]

| Metric | Value |
|--------|-------|
| Embedding (ArcFace, CPU) | ~85 ms |
| Qdrant ANN search (5 vectors) | ~1.5 ms |
| End-to-end /search (p50) | ~90 ms |
| End-to-end /search (p95) | ~125 ms |
| End-to-end /enroll (p50) | ~98 ms |

> **Note on Qdrant search latency:** At 5 identities Qdrant uses brute-force
> (configured `full_scan_threshold=10_000`). This is correct behaviour — HNSW
> graph construction is unnecessary overhead for tiny collections. As the
> collection grows beyond 10,000 vectors, Qdrant automatically switches to HNSW.
> The key point is that the search is still executed inside Qdrant — NOT a
> Python-level loop — so the code path scales correctly.

---

## 4. Threshold Analysis

**Selected threshold: 0.40**

| Metric | Value |
|--------|-------|
| EER threshold | 0.38 |
| EER | ~1.2% |
| FAR at t=0.40 | ~0.3% |
| FRR at t=0.40 | ~2.1% |

**Calibration method:**
With only 1 image per identity, genuine pair scores were approximated as self-similarity (1.0), and impostor pairs were all cross-identity cosines. Full calibration requires multiple images per identity — see `scripts/calibrate_threshold.py`.

**Why 0.40 and not the EER threshold (0.38)?**

EER minimises FAR = FRR equally. For access control, a FAR of 1.2% means 12 out of every 1,000 impostor attempts succeed — unacceptably high. By shifting to 0.40:

- FAR drops from 1.2% → 0.3% (4× improvement)
- FRR rises from 1.2% → 2.1% (a genuine user has a 2.1% chance of retry)

This trade-off is intentional: **security failures (FAR) are worse than user friction (FRR)**.

---

## 5. Hard Negative Analysis: George W. Bush ↔ Tony Blair

**Why this pair?**
Both subjects are: white male, ~55–65 years old at time of photographs, commonly photographed in formal attire (suit and tie), under studio or press-conference lighting. Contextually they are among the most visually similar pairs in the LFW dataset.

**Cosine similarity:** ~0.27–0.32 (depending on specific image crop)

**Threshold:** 0.40

**Verdict:** ✓ Correctly rejected as non-match (score < threshold)

**Margin to threshold:** ~0.08–0.13 cosine units

This is a comfortable margin but not a wide one. This pair represents a real stress case for the system.

### Error Type Analysis

**If we incorrectly set threshold too low (e.g. 0.20):**
- FAR increases — more impostors accepted
- Tony Blair might match George W. Bush in some image conditions

**If we incorrectly set threshold too high (e.g. 0.70):**
- FRR increases — genuine users sometimes rejected
- Even genuine george_w_bush images might miss the threshold

### Which error is worse?

| Use Case | Worse Error | Reason |
|----------|-------------|--------|
| **Access Control** | FAR (False Accept) | An impostor entering a restricted area is a security incident. A genuine user denied access just tries again or uses a fallback. |
| **Surveillance / Watchlist** | FRR (False Reject) | Missing a suspect in a crowd defeats the purpose of the system entirely. A false alarm (FAR) wastes investigator time but is safer. |
| **Attendance Tracking** | FRR (False Reject) | Marking a present employee as absent creates payroll errors and requires manual correction. FAR (marking an absent person present) may be collusion risk but is rarer. |

**For this deployment (access control context):** FAR is the worse error. We accept a slightly higher FRR (more retries) to minimise false accepts.

---

## 6. Live URL

`https://face-match-service.onrender.com`

**API docs:** `https://face-match-service.onrender.com/docs`

---

## 7. Sample curl Commands

```bash
# Enroll
curl -X POST https://face-match-service.onrender.com/enroll \
  -F "identity_id=george_w_bush" \
  -F "image=@data/sample_faces/george_w_bush/George_W_Bush_0001.jpg"

# Search
curl -X POST https://face-match-service.onrender.com/search \
  -F "image=@data/sample_faces/query/george_w_bush_George_W_Bush_0002.jpg"

# Health
curl https://face-match-service.onrender.com/health

# Metrics
curl https://face-match-service.onrender.com/metrics
```
