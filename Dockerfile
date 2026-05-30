# ═══════════════════════════════════════════════════════════════════
#  Face Match Service — Dockerfile
#  Multi-stage build: builder → runtime
#  Final image size: ~1.2 GB (driven by InsightFace + ONNX Runtime)
# ═══════════════════════════════════════════════════════════════════

# ── Stage 1: dependency builder ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for OpenCV + InsightFace compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps into a prefix we can copy to the runtime stage
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="face-match-service"
LABEL version="1.0.0"
LABEL description="ArcFace + Qdrant face matching service"

# Runtime-only system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# InsightFace model cache dir — writable by appuser
RUN mkdir -p /home/appuser/.insightface && \
    chown -R appuser:appuser /home/appuser/.insightface

# Copy application code
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser scripts/ ./scripts/

# Pre-download the InsightFace model at build time so the container
# starts in seconds rather than downloading ~300 MB on first request.
# This makes the image larger but eliminates cold-start latency.
RUN python -c "
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'])
fa.prepare(ctx_id=-1, det_size=(640, 640))
print('Model downloaded and cached.')
" || echo "Model pre-download failed — will download at runtime."

USER appuser

# Health check — container orchestrators use this
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Uvicorn with sensible production settings
# workers=1 because InsightFace model is not picklable for multi-process
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--access-log"]
