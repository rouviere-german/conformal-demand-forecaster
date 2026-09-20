#!/usr/bin/env python3
"""The split-conformal coverage guarantee, measured rather than assumed.

Usage:
  uv run python bench/bench_conformal.py           # run, write bench/results/conformal.json
  uv run python bench/bench_conformal.py --check   # also enforce bench/thresholds.json

The theorem (Vovk et al.; Lei et al. 2018) says that for exchangeable data the split-conformal
interval built at the ``ceil((n+1)(1-alpha))``-th smallest calibration score covers a fresh point
with probability in ``[1-alpha, 1-alpha + 1/(n+1)]``. This bench draws ``--splits`` independent
exchangeable splits of a synthetic regression problem, builds a CQR band on each and scores one
held-out point, and checks that the mean coverage lands inside that bracket within binomial
tolerance (``3 * sqrt(p(1-p)/splits)``).

It also measures what Mondrian calibration buys: with two groups of very different noise scale,
a pooled band over-covers one group and under-covers the other, while per-group calibration
holds both near nominal. That gap is the number reported as ``segment_gap``.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt

from conformal_demand_forecaster.conformal.cqr import (
    conformal_quantile,
    conformal_rank,
    coverage_indicator,
    cqr_scores,
)
from conformal_demand_forecaster.evaluation.report import (
    check_thresholds,
    load_thresholds,
    machine_info,
    markdown_table,
    write_results,
)

BENCH = "conformal"
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / f"{BENCH}.json"
THRESHOLDS = ROOT / "bench" / "thresholds.json"

FloatArray = npt.NDArray[np.float64]
ALPHAS = (0.2, 0.05)


Sample = tuple[FloatArray, FloatArray, FloatArray]


def _draw(rng: np.random.Generator, n: int, scale: float) -> Sample:
    """An exchangeable sample plus a deliberately mis-specified quantile pair.

    The raw band is a fixed, biased interval -- exactly the situation CQR is designed for: the
    conformal correction has to repair a badly calibrated quantile model.
    """
    y = rng.gamma(shape=3.0, scale=scale, size=n)
    lower = np.full(n, 1.0 * scale)
    upper = np.full(n, 4.0 * scale)
    return y, lower, upper


def split_conformal_coverage(splits: int, n_calibration: int, seed: int) -> dict[str, float]:
    """Mean coverage of a fresh point over many independent exchangeable splits."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    for alpha in ALPHAS:
        hits = 0
        widths = []
        for _ in range(splits):
            y, lower, upper = _draw(rng, n_calibration + 1, scale=2.0)
            q = conformal_quantile(cqr_scores(y[:-1], lower[:-1], upper[:-1]), alpha)
            hits += int(coverage_indicator(y[-1:], lower[-1:] - q, upper[-1:] + q)[0])
            widths.append(float((upper[-1] + q) - (lower[-1] - q)))
        coverage = hits / splits
        tag = f"{round((1 - alpha) * 100):d}"
        lower_bound = 1.0 - alpha
        upper_bound = 1.0 - alpha + 1.0 / (n_calibration + 1)
        tolerance = 3.0 * math.sqrt(lower_bound * alpha / splits)
        out[f"coverage_{tag}"] = round(coverage, 5)
        out[f"coverage_{tag}_lower_bound"] = round(lower_bound - tolerance, 5)
        out[f"coverage_{tag}_upper_bound"] = round(upper_bound + tolerance, 5)
        # Signed slack: >= 0 means the measured coverage sits inside the bracket.
        out[f"coverage_{tag}_slack"] = round(
            min(coverage - (lower_bound - tolerance), (upper_bound + tolerance) - coverage), 5
        )
        out[f"width_{tag}"] = round(float(np.mean(widths)), 4)
        out[f"rank_{tag}"] = float(conformal_rank(n_calibration, alpha))
    return out


def mondrian_versus_pooled(splits: int, n_calibration: int, seed: int) -> dict[str, float]:
    """Two groups with very different spread: does per-group calibration close the gap?"""
    rng = np.random.default_rng(seed)
    scales = {"tight": 1.0, "wide": 6.0}
    alpha = 0.2
    hits = {("pooled", name): 0 for name in scales}
    hits.update({("mondrian", name): 0 for name in scales})

    for _ in range(splits):
        samples = {name: _draw(rng, n_calibration + 1, scale) for name, scale in scales.items()}
        per_group = {
            name: cqr_scores(y[:-1], lo[:-1], hi[:-1]) for name, (y, lo, hi) in samples.items()
        }
        pooled_q = conformal_quantile(np.concatenate(list(per_group.values())), alpha)
        for name, (y, lo, hi) in samples.items():
            group_q = conformal_quantile(per_group[name], alpha)
            hits[("pooled", name)] += int(
                coverage_indicator(y[-1:], lo[-1:] - pooled_q, hi[-1:] + pooled_q)[0]
            )
            hits[("mondrian", name)] += int(
                coverage_indicator(y[-1:], lo[-1:] - group_q, hi[-1:] + group_q)[0]
            )

    out: dict[str, float] = {}
    for method in ("pooled", "mondrian"):
        gaps = []
        for name in scales:
            coverage = hits[(method, name)] / splits
            out[f"{method}_coverage_{name}"] = round(coverage, 5)
            gaps.append(abs(coverage - (1.0 - alpha)))
        out[f"{method}_segment_gap"] = round(max(gaps), 5)
    out["mondrian_gap_improvement"] = round(
        out["pooled_segment_gap"] - out["mondrian_segment_gap"], 5
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="split-conformal coverage guarantee")
    parser.add_argument("--splits", type=int, default=500)
    parser.add_argument("--calibration", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    metrics = split_conformal_coverage(args.splits, args.calibration, args.seed)
    metrics.update(mondrian_versus_pooled(args.splits, args.calibration, args.seed + 1))
    metrics["splits"] = float(args.splits)
    metrics["n_calibration"] = float(args.calibration)
    metrics["seconds"] = round(time.perf_counter() - started, 2)

    write_results(
        RESULTS,
        {"bench": BENCH, "machine": machine_info(), "args": vars(args), "metrics": metrics},
    )
    print(
        markdown_table(
            ["level", "measured coverage", "lower bound", "upper bound", "mean width"],
            [
                [
                    f"{tag}%",
                    metrics[f"coverage_{tag}"],
                    metrics[f"coverage_{tag}_lower_bound"],
                    metrics[f"coverage_{tag}_upper_bound"],
                    metrics[f"width_{tag}"],
                ]
                for tag in ("80", "95")
            ],
        )
    )
    print()
    print(
        markdown_table(
            ["calibration", "tight group", "wide group", "worst-group gap"],
            [
                [
                    method,
                    metrics[f"{method}_coverage_tight"],
                    metrics[f"{method}_coverage_wide"],
                    metrics[f"{method}_segment_gap"],
                ]
                for method in ("pooled", "mondrian")
            ],
        )
    )
    if args.check:
        problems = check_thresholds(metrics, load_thresholds(THRESHOLDS, BENCH))
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all conformal metrics within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
