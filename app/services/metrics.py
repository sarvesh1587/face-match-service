"""
Metrics Collector
=================
Lightweight in-process metrics store.
In production you'd wire this to Prometheus/OpenTelemetry.
Here it provides the /metrics endpoint data without external dependencies.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Deque

import numpy as np

from app.models.schemas import LatencyStats, MetricsResponse


@dataclass
class _Sample:
    value_ms: float
    timestamp: float = field(default_factory=time.monotonic)


class MetricsCollector:
    """Thread-safe rolling-window latency tracker."""

    _WINDOW = 1000  # keep last N samples

    def __init__(self) -> None:
        self._lock = Lock()
        self._search_samples: Deque[_Sample] = deque(maxlen=self._WINDOW)
        self._enroll_samples: Deque[_Sample] = deque(maxlen=self._WINDOW)
        self._total_searches = 0
        self._total_enrollments = 0
        self._total_matches = 0
        self._start_time = time.monotonic()

    def record_search(self, latency_ms: float, matched: bool) -> None:
        with self._lock:
            self._search_samples.append(_Sample(latency_ms))
            self._total_searches += 1
            if matched:
                self._total_matches += 1

    def record_enroll(self, latency_ms: float) -> None:
        with self._lock:
            self._enroll_samples.append(_Sample(latency_ms))
            self._total_enrollments += 1

    def snapshot(self, enrolled_count: int) -> MetricsResponse:
        with self._lock:
            return MetricsResponse(
                enrolled_identities=enrolled_count,
                total_searches=self._total_searches,
                total_enrollments=self._total_enrollments,
                search_latency=self._compute_stats(self._search_samples),
                enroll_latency=self._compute_stats(self._enroll_samples),
                match_rate=(
                    self._total_matches / self._total_searches
                    if self._total_searches > 0
                    else 0.0
                ),
                uptime_seconds=round(time.monotonic() - self._start_time, 1),
            )

    @staticmethod
    def _compute_stats(samples: Deque[_Sample]) -> LatencyStats:
        if not samples:
            return LatencyStats(p50_ms=0, p95_ms=0, p99_ms=0, mean_ms=0, count=0)
        vals = np.array([s.value_ms for s in samples])
        return LatencyStats(
            p50_ms=round(float(np.percentile(vals, 50)), 2),
            p95_ms=round(float(np.percentile(vals, 95)), 2),
            p99_ms=round(float(np.percentile(vals, 99)), 2),
            mean_ms=round(float(np.mean(vals)), 2),
            count=len(vals),
        )


# Module-level singleton — imported by route handlers
metrics = MetricsCollector()
