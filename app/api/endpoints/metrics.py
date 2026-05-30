"""GET /metrics — operational metrics snapshot."""

from fastapi import APIRouter, Request

from app.models.schemas import MetricsResponse
from app.services.metrics import metrics

router = APIRouter()


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Service metrics",
    description=(
        "Returns operational metrics: request counts, latency percentiles "
        "(p50/p95/p99), match rate, and uptime. "
        "Data is a rolling window of the last 1,000 requests."
    ),
)
async def get_metrics(request: Request) -> MetricsResponse:
    enrolled_count = request.app.state.vector_store.count()
    return metrics.snapshot(enrolled_count)
