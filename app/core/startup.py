"""
Startup Checks
==============
Validates environment and external dependencies before the application
begins accepting traffic. A hard failure here prevents a misconfigured
service from silently returning wrong results.
"""

import sys

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


async def run_startup_checks() -> None:
    """Run all preflight checks. Raises SystemExit on fatal failure."""
    checks = [
        _check_python_version,
        _check_qdrant_reachable,
        _check_insightface_models,
    ]
    for check in checks:
        await check()

    logger.info("All startup checks passed")


async def _check_python_version() -> None:
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10):
        logger.error("Python ≥ 3.10 required", found=f"{major}.{minor}")
        raise SystemExit(1)
    logger.info("Python version OK", version=f"{major}.{minor}")


async def _check_qdrant_reachable() -> None:
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY,
            timeout=5,
        )
        client.get_collections()
        logger.info(
            "Qdrant reachable",
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )
    except Exception as exc:
        logger.error(
            "Qdrant unreachable — is the container running?",
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            error=str(exc),
        )
        raise SystemExit(1) from exc


async def _check_insightface_models() -> None:
    try:
        import insightface  # noqa: F401

        logger.info("InsightFace available", model=settings.INSIGHTFACE_MODEL)
    except ImportError as exc:
        logger.error("InsightFace not installed", error=str(exc))
        raise SystemExit(1) from exc
