#!/usr/bin/env python3
"""Rolling-origin backtest: accuracy, calibration and decision quality against the baselines.

Usage:
  uv run python bench/bench_backtest.py            # CI-sized run (20 series), writes results
  uv run python bench/bench_backtest.py --series 200 --origins 12    # the full evaluation
  uv run python bench/bench_backtest.py --check    # also enforce bench/thresholds.json

At every origin the model is retrained and recalibrated on the window before it, then scored on
the next 28 days. Everything is scored against the generator's *latent* demand, so censored
stockout days do not flatter the intervals; coverage against censored sales is reported
separately for comparison.

What the table answers:

- is the global gradient-boosted model actually better than seasonal-naive and ETS (MASE, and
  the M5 weighted scaled pinball loss over all five quantile levels)?
- do the conformalized bands hit their nominal coverage, and at what width and Winkler score?
- does Mondrian calibration close the per-segment coverage gap that a pooled band leaves?
- does ordering from the conformalized quantiles beat ordering the point forecast, at the
  configured holding and stockout costs?
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from conformal_demand_forecaster.data.generator import GeneratorConfig, generate_panel
from conformal_demand_forecaster.data.panel import Panel
from conformal_demand_forecaster.evaluation.backtest import BacktestConfig, run_backtest
from conformal_demand_forecaster.evaluation.report import (
    check_thresholds,
    load_thresholds,
    machine_info,
    markdown_table,
    write_results,
)
from conformal_demand_forecaster.models.gbm import GbmConfig
from conformal_demand_forecaster.training import TrainingConfig

BENCH = "backtest"
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / f"{BENCH}.json"
THRESHOLDS = ROOT / "bench" / "thresholds.json"


def build_config(args: argparse.Namespace) -> BacktestConfig:
    training = TrainingConfig(
        gbm=GbmConfig(num_threads=args.threads),
        train_days=args.train_days,
        calibration_days=args.calibration_days,
    )
    return BacktestConfig(
        n_origins=args.origins,
        origin_stride=args.stride,
        training=training,
        ets_series_limit=args.ets_limit,
        holding_cost=args.holding_cost,
        stockout_cost=args.stockout_cost,
    )


def derived(metrics: dict[str, float]) -> dict[str, float]:
    """Ratios the thresholds are expressed in: beating a baseline is a ratio below one."""
    out = dict(metrics)
    for tag in ("80", "95"):
        nominal = 1.0 - (0.2 if tag == "80" else 0.05)
        for method in ("mondrian", "pooled", "aci"):
            out[f"coverage_error_{tag}_{method}"] = round(
                abs(metrics[f"coverage_{tag}_{method}"] - nominal), 5
            )
        out[f"segment_gap_improvement_{tag}"] = round(
            metrics[f"segment_gap_{tag}_pooled"] - metrics[f"segment_gap_{tag}_mondrian"], 5
        )
    out["mase_ratio_vs_snaive"] = round(metrics["mase_gbm"] / metrics["mase_snaive"], 5)
    out["mase_ratio_vs_ets"] = round(metrics["mase_gbm"] / metrics["mase_ets"], 5)
    out["wspl_ratio_vs_snaive"] = round(metrics["wspl_gbm"] / metrics["wspl_snaive"], 5)
    out["wspl_ratio_vs_ets"] = round(metrics["wspl_gbm"] / metrics["wspl_ets"], 5)
    return out


def _accuracy_table(m: dict[str, float]) -> str:
    return markdown_table(
        ["model", "MASE", "weighted scaled pinball", "coverage 80%", "coverage 95%"],
        [
            [
                "conformal GBM",
                m["mase_gbm"],
                m["wspl_gbm"],
                m["coverage_80_mondrian"],
                m["coverage_95_mondrian"],
            ],
            [
                "seasonal naive",
                m["mase_snaive"],
                m["wspl_snaive"],
                m["coverage_80_snaive"],
                m["coverage_95_snaive"],
            ],
            ["ETS", m["mase_ets"], m["wspl_ets"], m["coverage_80_ets"], m["coverage_95_ets"]],
        ],
    )


def _calibration_table(m: dict[str, float]) -> str:
    return markdown_table(
        ["level", "method", "coverage", "width", "Winkler", "worst-segment gap"],
        [
            [
                f"{tag}%",
                method,
                m[f"coverage_{tag}_{method}"],
                m[f"width_{tag}_{method}"],
                m[f"winkler_{tag}_{method}"],
                m[f"segment_gap_{tag}_{method}"],
            ]
            for tag in ("80", "95")
            for method in ("mondrian", "pooled", "aci")
        ],
    )


def _decision_table(m: dict[str, float]) -> str:
    return markdown_table(
        ["ordering policy", "realized newsvendor cost"],
        [
            ["conformalized quantiles", m["newsvendor_cost_conformal"]],
            ["point forecast", m["newsvendor_cost_point"]],
            ["median forecast", m["newsvendor_cost_median"]],
        ],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="rolling-origin backtest")
    parser.add_argument("--series", type=int, default=20, help="20 in CI, 200 for the full run")
    parser.add_argument("--days", type=int, default=1096)
    parser.add_argument("--origins", type=int, default=12)
    parser.add_argument("--stride", type=int, default=28)
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--calibration-days", type=int, default=56)
    parser.add_argument("--ets-limit", type=int, default=None, help="cap ETS to the first N series")
    parser.add_argument("--holding-cost", type=float, default=1.0)
    parser.add_argument("--stockout-cost", type=float, default=4.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    frame = generate_panel(GeneratorConfig(n_series=args.series, n_days=args.days, seed=args.seed))
    panel = Panel.from_frame(frame, validate=False)
    result = run_backtest(panel, build_config(args))
    metrics = derived(result.metrics)
    metrics["wall_seconds"] = round(time.perf_counter() - started, 2)
    metrics["series"] = float(args.series)

    write_results(
        RESULTS,
        {
            "bench": BENCH,
            "machine": machine_info(),
            "args": vars(args),
            "digest": result.digest(),
            "metrics": metrics,
            "per_origin": result.per_origin,
        },
    )
    print(_accuracy_table(metrics))
    print()
    print(_calibration_table(metrics))
    print()
    print(_decision_table(metrics))
    print()
    print(
        f"rows scored: {int(metrics['rows'])} | origins: {int(metrics['origins'])} | "
        f"series: {args.series} | digest: {result.digest()[:16]} | "
        f"{metrics['wall_seconds']:.1f}s"
    )
    if args.check:
        problems = check_thresholds(metrics, load_thresholds(THRESHOLDS, BENCH))
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all backtest metrics within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
