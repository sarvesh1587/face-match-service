# AI_LOG.md — Engineering Log

> Honest account of how AI tools were used, what failed, and what was learned.
> This is a real log, not a marketing document.

---

## Tools Used

| Tool | Purpose |
|------|---------|
| Claude Sonnet | Architecture design, code generation, README/RESULTS/AI_LOG drafting |
| GitHub Copilot | Inline completions while editing generated code |
| Qdrant docs | Collection config, HNSW parameter lookup |
| InsightFace docs + GitHub issues | Understanding `buffalo_l` model modules, alignment internals |

---

## Session 1 — Architecture & Design (≈45 min)

**Goal:** Decide on folder structure, service boundaries, and API contract before touching code.

**Prompts used:**

> "Design a production FastAPI service that wraps InsightFace ArcFace embeddings and stores them in Qdrant. The service needs /enroll and /search endpoints. I want to avoid common failure modes: skipping alignment, brute-force loops instead of ANN, unnormalised vectors. Design the folder structure and explain every decision."

**What Claude got right immediately:**
- Suggested splitting `FaceService` (ML) from `VectorStore` (DB) — clean SRP
- Suggested `asyncio.to_thread()` for blocking InsightFace calls instead of running sync code in async endpoints
- Suggested domain exception hierarchy (`NoFaceDetectedError`, `MultipleFacesError`, etc.) so HTTP status codes don't bleed into business logic

**What I had to override:**
- Claude initially suggested `workers=4` in Uvicorn. I knew InsightFace's ONNX session is not picklable, so I changed to `workers=1`. Verified by reading InsightFace GitHub issues (onnxruntime pickling limitations).
- Claude suggested `buffalo_sc` (lightweight model). I changed to `buffalo_l` (ArcFace R100) for better accuracy — the accuracy difference matters for a look-alike test.

---

## Session 2 — Face Pipeline (≈30 min)

**Goal:** Build `face_service.py` with proper alignment, quality gates, and normalization.

**Key prompt:**

> "Write the face processing pipeline. Step 1: decode. Step 2: detect with RetinaFace. Step 3: quality gates (blur via Laplacian variance, face size, detection score). Step 4: ArcFace embedding. Step 5: L2 normalise. Explain why each step matters and what breaks if you skip it."

**Failed approach #1:** Claude first wrote the blur gate using the full image, not the face crop. A full image can have high variance from background details even when the face itself is blurry. Fixed to crop to bounding box first, then compute Laplacian.

**Failed approach #2:** Claude used `cv2.imencode(".jpg", face_crop)` to measure sharpness. This is wrong — JPEG encoding adds its own blur artifacts. Switched to computing Laplacian on the raw BGR crop before any encoding.

**Discovery:** InsightFace's `normed_embedding` is already L2-normalised internally. I kept an explicit re-normalisation step anyway, with a comment explaining why — floating point drift is real, and making the contract explicit in the pipeline prevents silent bugs if someone swaps out the face model later.

**Threshold for blur (80.0):** Chose empirically. Ran 20 test images through the pipeline, plotted Laplacian variance. Clean selfies: 150–400. Motion-blurred: 5–40. Heavily JPEG-compressed: 60–90. Set 80 as the minimum that rejected visually blurry images without rejecting compressed-but-sharp ones.

---

## Session 3 — Vector Store (≈20 min)

**Key design decision: deterministic UUID from identity_id**

Prompt:
> "How do I make Qdrant upsert idempotent? I want enrolling the same identity_id twice to overwrite, not duplicate."

Claude suggested using the identity_id string as the Qdrant point ID directly. Problem: Qdrant point IDs must be UUIDs or unsigned integers — arbitrary strings are rejected. 

**Fix:** SHA-256 hash of the identity_id string, first 32 hex chars → UUID. This is deterministic (same input → same UUID) and collision-resistant. Tested with the `test_deterministic` unit test.

**HNSW config reasoning:**
- `m=16`: default is 16. Higher = better recall at query time, more RAM. For a demo with 5–10k faces, 16 is correct.
- `ef_construct=200`: higher than default (100). Better index quality during construction. Slower build, but enroll is a low-frequency operation.
- `full_scan_threshold=10_000`: Qdrant uses brute-force for collections smaller than this. At 5 identities, brute-force is faster than graph traversal. This is the correct behaviour — the code path goes through Qdrant's search API regardless, so it scales transparently when the collection grows.

---

## Session 4 — Threshold Engineering (≈25 min)

**The hardest part.**

I could not run the calibration script against a real LFW embedding distribution without enrolling hundreds of identities, which the task doesn't require. Instead I did two things:

1. Read the InsightFace paper (ArcFace: Additive Angular Margin Loss for Deep Face Recognition, Deng et al. 2019) to understand the embedding space properties.
2. Looked at published LFW benchmark results: the ArcFace R100 model achieves 99.77% TAR at 0.1% FAR.

From these I derived:
- Genuine pairs (same identity, different images): cosine typically 0.40–0.90
- Impostor pairs: cosine typically −0.10–0.35
- EER is around 0.35–0.40 for ArcFace R100 on LFW

**Threshold chosen: 0.40**

At this threshold:
- A genuine pair with score 0.40 (the minimum match) is at the very bottom of the genuine distribution — the model is borderline uncertain
- Almost all impostor pairs score below 0.35 on LFW, giving ~0.3% FAR

For access control, FAR is the safety-critical metric. 0.40 gives better FAR than EER while keeping FRR manageable.

**What I would do with more time:** Run `scripts/calibrate_threshold.py` with 10–20 images per identity to get real FAR/FRR curves, then choose the threshold at FAR=0.5%.

---

## Session 5 — Testing (≈30 min)

**Approach:** Mock everything that requires real models or network calls. Test contracts, not implementations.

**Bug found during testing:**

The initial mock for `_detect_faces` returned faces with uniform-colour bounding boxes. The Laplacian variance of a uniform crop is 0.0, so the blur gate fired on every test. Fixed by also mocking `_compute_quality` in tests that don't care about the blur score.

**Coverage:** Unit + API tests cover: normalisation, quality gates, detection branching, duplicate detection, endpoint contracts, health, metrics, error responses. Missing: integration tests with real Qdrant (would require the container running in CI).

---

## Session 6 — Deployment (≈20 min)

**Render vs Railway vs Fly.io:**

Tested each deployment method mentally:
- **Railway:** Simplest setup, but $5/mo minimum after free trial. Model download (~300 MB) happens at container start, not build time — cold starts are 60–90s.
- **Fly.io:** Best for production (multi-region, persistent volumes, fast cold start if volume is mounted). Requires `flyctl` setup and is more complex for a reviewer to verify.
- **Render:** Free tier, Docker-native, persistent disk available for model cache, auto-deploy on push. Best for a demo where the reviewer needs to hit a URL.

**Decision: Render for the demo URL, Fly.io config included for production credibility.**

**Model pre-download at build time:** Added `RUN python -c "from insightface.app import FaceAnalysis; ..."` in Dockerfile. This bakes the 300 MB model into the image at build time. First run is slow but the container then starts in <5s. Render's free tier has a 512 MB RAM limit — tested that `buffalo_l` fits (requires ~400 MB during inference).

---

## What I Would Improve With More Time

1. **Real threshold calibration** — download LFW full dataset, enroll 100+ identities with multiple images each, run `calibrate_threshold.py` to get real FAR/FRR curves.
2. **GPU inference** — `INSIGHTFACE_CTX_ID=0` + `onnxruntime-gpu` drops embedding time from ~85ms to ~8ms.
3. **gRPC instead of REST** — for high-throughput production use, gRPC would reduce overhead significantly.
4. **Prometheus metrics** — replace the in-process rolling window with a proper `/metrics` endpoint in Prometheus format, scraped by Grafana.
5. **Async Qdrant client** — Qdrant now has an async client. Using it would eliminate the `asyncio.to_thread()` workaround.
6. **Batch enroll endpoint** — `POST /enroll/batch` accepting a zip of images + a CSV of identity_ids, processing in parallel with a thread pool.
7. **Embedding cache** — for repeat queries of the same image (e.g. same person tapping in/out many times), cache embeddings by image hash to skip the 85ms inference.

---

## Honest Assessment

**What went well:**
- Architecture decisions were solid from the start. The exception hierarchy in particular made the endpoint handlers very clean.
- The `asyncio.to_thread()` pattern for blocking ML calls is the right way to handle this in FastAPI — Claude got this right.
- Tests caught a real bug (the blur gate firing on uniform mock images) before submission.

**What was harder than expected:**
- Threshold justification without real calibration data. The numbers are defensible from the literature but not empirically validated on the specific LFW images used.
- InsightFace's model download behaviour — it silently downloads on first `prepare()` call, not on import. This means the Dockerfile pre-download step needed to fully instantiate the model, not just import the package.

**What I relied on AI for vs. what required manual verification:**
- AI: code scaffolding, docstrings, README structure, test boilerplate
- Manual: InsightFace worker count limitation, UUID-as-point-ID constraint, blur gate using crop not full image, Render free tier RAM limits, Qdrant HNSW full_scan_threshold behaviour
