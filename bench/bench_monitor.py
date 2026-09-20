#!/usr/bin/env python3
"""The coverage monitor's operating characteristic: false alarms and detection delay.

Usage:
  uv run python bench/bench_monitor.py           # run, write bench/results/monitor.json
  uv run python bench/bench_monitor.py --check   # also enforce bench/thresholds.json

An alarm opens a recalibration proposal that an operator has to read, so a noisy monitor is not
merely untidy -- it trains people to ignore it. Two numbers decide whether the rule is usable:

- **false-alarm rate**: the share of stationary simulations (coverage exactly at nominal) in
  which the monitor fires at least once over the whole run. The test is applied on every day of
  an overlapping rolling window, so this is a family-wise rate over many correlated tests, not
  the nominal per-test significance.
- **detection delay**: the number of days between a genuine coverage breach and the first alarm,
  reported as the median and the 90th percentile over many simulations, plus the share of runs
  that never detect the breach inside the window.
- the same two numbers for the **over-coverage arm**, which reports bands that are systematically
  wider than their level asks for. Over-coverage is holding cost rather than a broken promise, so
  it never opens a proposal, but a monitor that cannot see it is only half a monitor.

Both are measured by simulating the miss indicator directly (binomial draws), which isolates the
alarm rule from the forecasting model.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from conformal_demand_forecaster.evaluation.report import (
    check_thresholds,
    load_thresholds,
    machine_info,
    markdown_table,
    write_results,
)
from conformal_demand_forecaster.monitoring.coverage import CoverageMonitor, MonitorConfig

BENCH = "monitor"
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / f"{BENCH}.json"
THRESHOLDS = ROOT / "bench" / "thresholds.json"

ALPHA = 0.2
SEGMENT = "smooth"


def _config() -> MonitorConfig:
    return MonitorConfig()  # the shipped defaults


def false_alarm_rate(runs: int, days: int, per_day: int, seed: int) -> dict[str, float]:
    """Share of perfectly calibrated runs in which either arm fires at least once."""
    rng = np.random.default_rng(seed)
    config = _config()
    fired = 0
    over_fired = 0
    alarm_days = 0
    tested_days = 0
    for _ in range(runs):
        monitor = CoverageMonitor(config)
        run_fired = False
        run_over_fired = False
        for day in range(days):
            monitor.record(SEGMENT, ALPHA, day, per_day, int(rng.binomial(per_day, ALPHA)))
            status = monitor.status(SEGMENT, ALPHA, day)
            if status.n >= config.min_observations:
                tested_days += 1
                if status.alarm:
                    alarm_days += 1
                    run_fired = True
                run_over_fired = run_over_fired or status.over_alarm
        fired += int(run_fired)
        over_fired += int(run_over_fired)
    return {
        "false_alarm_rate_per_run": round(fired / runs, 5),
        "false_alarm_rate_per_test": round(alarm_days / max(tested_days, 1), 5),
        "over_false_alarm_rate_per_run": round(over_fired / runs, 5),
        "stationary_runs": float(runs),
        "stationary_days": float(days),
    }


def detection_delay(
    runs: int, burn_in: int, per_day: int, miss_rate: float, seed: int, *, arm: str = "breach"
) -> dict[str, float]:
    """Days from a genuine miscalibration to the first alarm on ``arm``, over many simulations."""
    rng = np.random.default_rng(seed)
    config = _config()
    delays: list[int] = []
    missed_runs = 0
    horizon = config.window_days * 2
    for _ in range(runs):
        monitor = CoverageMonitor(config)
        for day in range(burn_in):
            monitor.record(SEGMENT, ALPHA, day, per_day, int(rng.binomial(per_day, ALPHA)))
        found: int | None = None
        for offset in range(horizon):
            day = burn_in + offset
            monitor.record(SEGMENT, ALPHA, day, per_day, int(rng.binomial(per_day, miss_rate)))
            status = monitor.status(SEGMENT, ALPHA, day)
            if status.alarm if arm == "breach" else status.over_alarm:
                found = offset
                break
        if found is None:
            missed_runs += 1
        else:
            delays.append(found)
    values = np.asarray(delays, dtype=np.float64)
    median = round(float(np.median(values)), 3) if delays else float("nan")
    p90 = round(float(np.percentile(values, 90)), 3) if delays else float("nan")
    prefix = "" if arm == "breach" else "over_"
    return {
        f"{prefix}breach_miss_rate": miss_rate,
        f"{prefix}detection_delay_median_days": median,
        f"{prefix}detection_delay_p90_days": p90,
        f"{prefix}undetected_rate": round(missed_runs / runs, 5),
        f"{prefix}breach_runs": float(runs),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="coverage monitor false alarms and delay")
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--per-day", type=int, default=40)
    parser.add_argument("--burn-in", type=int, default=40)
    parser.add_argument("--breach-miss-rate", type=float, default=0.35)
    parser.add_argument("--over-miss-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    metrics = false_alarm_rate(args.runs, args.days, args.per_day, args.seed)
    metrics.update(
        detection_delay(args.runs, args.burn_in, args.per_day, args.breach_miss_rate, args.seed + 1)
    )
    metrics.update(
        detection_delay(
            args.runs,
            args.burn_in,
            args.per_day,
            args.over_miss_rate,
            args.seed + 2,
            arm="over",
        )
    )
    config = _config()
    metrics["window_days"] = float(config.window_days)
    metrics["tolerance"] = config.tolerance
    metrics["over_tolerance"] = config.over_tolerance
    metrics["significance"] = config.significance
    metrics["seconds"] = round(time.perf_counter() - started, 2)

    write_results(
        RESULTS,
        {"bench": BENCH, "machine": machine_info(), "args": vars(args), "metrics": metrics},
    )
    print(
        markdown_table(
            ["metric", "value"],
            [
                ["false alarms (share of stationary runs)", metrics["false_alarm_rate_per_run"]],
                ["false alarms (share of daily tests)", metrics["false_alarm_rate_per_test"]],
                ["detection delay, median days", metrics["detection_delay_median_days"]],
                ["detection delay, p90 days", metrics["detection_delay_p90_days"]],
                ["breaches never detected", metrics["undetected_rate"]],
                [
                    "over-coverage false alarms (share of runs)",
                    metrics["over_false_alarm_rate_per_run"],
                ],
                [
                    "over-coverage detection delay, median days",
                    metrics["over_detection_delay_median_days"],
                ],
                ["over-coverage never detected", metrics["over_undetected_rate"]],
            ],
        )
    )
    if args.check:
        problems = check_thresholds(metrics, load_thresholds(THRESHOLDS, BENCH))
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all monitor metrics within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
