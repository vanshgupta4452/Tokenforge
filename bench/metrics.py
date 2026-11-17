"""Timing and metric helpers. CUDA events when available, perf_counter otherwise."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

try:
    import torch
except ImportError:
    torch = None


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[rank]


class StepTimer:
    def __init__(self, use_cuda: bool | None = None) -> None:
        if use_cuda is None:
            use_cuda = bool(torch is not None and torch.cuda.is_available())
        self.use_cuda = use_cuda
        self._start: object = None

    def start(self) -> None:
        if self.use_cuda:
            self._start = torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            self._start = time.perf_counter()

    def stop_ms(self) -> float:
        if self.use_cuda:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            return float(self._start.elapsed_time(end))
        return (time.perf_counter() - self._start) * 1000.0


@dataclass
class RequestMetrics:
    arrival_s: float
    first_token_s: float | None = None
    token_times_s: list[float] = field(default_factory=list)

    def record_token(self, now: float | None = None) -> None:
        now = time.perf_counter() if now is None else now
        self.token_times_s.append(now)
        if self.first_token_s is None:
            self.first_token_s = now

    @property
    def ttft_ms(self) -> float:
        if self.first_token_s is None:
            return 0.0
        return (self.first_token_s - self.arrival_s) * 1000.0

    @property
    def itl_ms(self) -> list[float]:
        return [
            (later - earlier) * 1000.0
            for earlier, later in zip(self.token_times_s, self.token_times_s[1:])
        ]


def summarize(records: list[RequestMetrics], peak_mem_mb: float = 0.0) -> dict:
    ttfts = [record.ttft_ms for record in records if record.token_times_s]
    itls = [gap for record in records for gap in record.itl_ms]
    tokens = sum(len(record.token_times_s) for record in records)
    span = max(
        (max(record.token_times_s) for record in records if record.token_times_s),
        default=0.0,
    ) - min(
        (record.arrival_s for record in records if record.token_times_s),
        default=0.0,
    )
    return {
        "ttft_p50_ms": percentile(ttfts, 0.50),
        "ttft_p95_ms": percentile(ttfts, 0.95),
        "itl_p50_ms": percentile(itls, 0.50),
        "itl_p95_ms": percentile(itls, 0.95),
        "throughput_tok_s": tokens / span if span > 0 else 0.0,
        "peak_mem_mb": peak_mem_mb,
    }
