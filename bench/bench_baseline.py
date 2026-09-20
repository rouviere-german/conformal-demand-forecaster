#!/usr/bin/env python3
"""Micro-benchmarks for the hot paths, with CI regression thresholds.

Usage:
  uv run python bench/bench_baseline.py          # run, write bench/results/baseline.json
  uv run python bench/bench_baseline.py --check  # also exit 1 if bench/thresholds.json is exceeded

These measure per-call latency of the primitives that run on every request or every ingested
day: the conformal quantile lookup, CQR scoring, the vectorised newsvendor order, the monitor's
binomial test, PSI drift and the shared middleware helpers. Thresholds are deliberately loose
(several times the numbers measured on a laptop) because shared CI runners are noisy; they catch
order-of-magnitude regressions, not small drifts.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from conformal_demand_forecaster.conformal.aci import AlphaState
from conformal_demand_forecaster.conformal.cqr import (
    ScoreBuffer,
    conformal_quantile,
    cqr_scores,
)
from conformal_demand_forecaster.core.ratelimit import RateLimiter
from conformal_demand_forecaster.core.redaction import redact_text
from conformal_demand_forecaster.decisions.newsvendor import order_quantities
from conformal_demand_forecaster.evaluation.report import (
    check_thresholds,
    load_thresholds,
    machine_info,
    markdown_table,
    write_results,
)
from conformal_demand_forecaster.ml.drift import DriftMonitor
from conformal_demand_forecaster.monitoring.coverage import CoverageMonitor, MonitorConfig
from conformal_demand_forecaster.serving.guard import RobustZScoreGuard

BENCH = "baseline"
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / f"{BENCH}.json"
THRESHOLDS = ROOT / "bench" / "thresholds.json"


def _timed(function: Callable[[], object], runs: int, warmup: int = 5) -> dict[str, float]:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        function()
        samples.append(time.perf_counter_ns() - start)
    micros = np.asarray(samples, dtype=np.float64) / 1_000.0
    return {
        "runs": float(runs),
        "p50_us": round(float(np.percentile(micros, 50)), 1),
        "p95_us": round(float(np.percentile(micros, 95)), 1),
    }


def bench_conformal_quantile() -> dict[str, float]:
    scores = np.random.default_rng(0).normal(size=5_000)
    return _timed(lambda: conformal_quantile(scores, 0.05), runs=2_000)


def bench_cqr_scores() -> dict[str, float]:
    rng = np.random.default_rng(1)
    y = rng.poisson(20, size=5_000).astype(np.float64)
    lower = y - rng.uniform(1, 10, size=5_000)
    upper = y + rng.uniform(1, 10, size=5_000)
    return _timed(lambda: cqr_scores(y, lower, upper), runs=1_000)


def bench_score_buffer_append() -> dict[str, float]:
    rng = np.random.default_rng(2)
    days = np.arange(200, dtype=np.int64)
    scores = rng.normal(size=200)
    buffer = ScoreBuffer(window_days=56)
    return _timed(lambda: buffer.append(("smooth/b0", 0.05), days, scores), runs=500)


def bench_newsvendor_batch() -> dict[str, float]:
    rng = np.random.default_rng(3)
    n = 5_000
    median = rng.uniform(0, 50, size=n)
    intervals = {
        0.2: (median * 0.6, median * 1.4),
        0.05: (median * 0.3, median * 1.9),
    }
    return _timed(lambda: order_quantities(median, intervals, 1.0, 4.0), runs=500)


def bench_monitor_status() -> dict[str, float]:
    monitor = CoverageMonitor(MonitorConfig())
    rng = np.random.default_rng(4)
    for day in range(28):
        monitor.record("smooth", 0.2, day, 200, int(rng.binomial(200, 0.2)))
    return _timed(lambda: monitor.status("smooth", 0.2, 27), runs=2_000)


def bench_aci_update() -> dict[str, float]:
    state = AlphaState.initial(0.1, 0.03)
    return _timed(lambda: state.update(0.1), runs=20_000)


def bench_outlier_guard() -> dict[str, float]:
    history = np.random.default_rng(5).poisson(15, size=56).astype(np.float64)
    guard = RobustZScoreGuard()
    return _timed(lambda: guard.check(18.0, history), runs=2_000)


def bench_drift() -> dict[str, float]:
    rng = np.random.default_rng(6)
    monitor = DriftMonitor({"sales": rng.normal(size=10_000)})
    current = {"sales": rng.normal(loc=0.1, size=10_000)}
    return _timed(lambda: monitor.check(current), runs=30, warmup=2)


def bench_redaction() -> dict[str, float]:
    line = (
        "operator jane.doe@example.com approved recalibration from 10.0.0.7; "
        + "lorem ipsum dolor sit amet " * 35
    )
    return _timed(lambda: redact_text(line), runs=2_000)


def bench_rate_limiter() -> dict[str, float]:
    limiter = RateLimiter(rate=1e9, burst=1_000_000_000)
    return _timed(lambda: limiter.check("client"), runs=20_000)


def run() -> dict[str, dict[str, float]]:
    return {
        "conformal_quantile_5k_scores": bench_conformal_quantile(),
        "cqr_scores_5k_rows": bench_cqr_scores(),
        "score_buffer_append_200": bench_score_buffer_append(),
        "newsvendor_orders_5k_rows": bench_newsvendor_batch(),
        "monitor_status_28_days": bench_monitor_status(),
        "aci_update": bench_aci_update(),
        "outlier_guard_check": bench_outlier_guard(),
        "drift_check_10k_vs_10k": bench_drift(),
        "redact_text_1kb": bench_redaction(),
        "rate_limiter_check": bench_rate_limiter(),
    }


def flatten(results: dict[str, dict[str, float]]) -> dict[str, float]:
    """``{"bench": {"p95_us": 1.0}}`` becomes ``{"bench.p95_us": 1.0}`` for the threshold check."""
    return {
        f"{name}.{metric}": value
        for name, stats in results.items()
        for metric, value in stats.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="micro-benchmarks for the hot paths")
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    args = parser.parse_args(argv)

    results = run()
    flat = flatten(results)
    write_results(RESULTS, {"bench": BENCH, "machine": machine_info(), "metrics": flat})
    print(
        markdown_table(
            ["benchmark", "runs", "p50 (us)", "p95 (us)"],
            [[name, s["runs"], s["p50_us"], s["p95_us"]] for name, s in results.items()],
        )
    )
    if args.check:
        problems = check_thresholds(flat, load_thresholds(THRESHOLDS, BENCH))
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all benchmarks within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
