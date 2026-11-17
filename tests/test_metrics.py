from bench.metrics import RequestMetrics, percentile, summarize


def make_record(t0: float, tokens: int) -> RequestMetrics:
    record = RequestMetrics(arrival_s=t0)
    for index in range(tokens):
        record.record_token(t0 + 0.05 + index * 0.01)
    return record


def test_percentile_nearest_rank() -> None:
    assert percentile([], 0.5) == 0.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0


def test_request_timings() -> None:
    record = make_record(10.0, 3)
    assert abs(record.ttft_ms - 50.0) < 1e-6
    assert len(record.itl_ms) == 2
    assert all(abs(gap - 10.0) < 1e-6 for gap in record.itl_ms)


def test_summarize() -> None:
    metrics = summarize([make_record(0.0, 5), make_record(0.1, 5)], peak_mem_mb=42.0)
    assert metrics["peak_mem_mb"] == 42.0
    assert metrics["throughput_tok_s"] > 0.0
    assert metrics["ttft_p95_ms"] >= metrics["ttft_p50_ms"] >= 0.0
