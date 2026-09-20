#!/usr/bin/env python3
"""Behaviour under a regime shift: frozen split conformal versus Adaptive Conformal Inference.

Usage:
  uv run python bench/bench_shift.py           # run, write bench/results/shift.json
  uv run python bench/bench_shift.py --check   # also enforce bench/thresholds.json

The setup is the one the repository exists for. A model is trained and calibrated *before* a
level shift and then frozen -- no retraining, no recalibration, which is what happens between
scheduled training runs in production. Every following day it issues forecasts and the realized
demand is scored two ways:

- ``static``: the frozen split-conformal band at the nominal alpha;
- ``aci``: the same raw quantiles, with the conformal quantile looked up at the ACI-adjusted
  alpha, which is updated once per day per Mondrian group from that day's realized miss rate.

Reported: coverage before the shift (both should be near nominal), coverage after it, and the
ACI long-run coverage error over the whole post-shift stretch. Widths are reported too, because
ACI buys its coverage back by widening -- that cost belongs in the same table as the benefit.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt

from conformal_demand_forecaster.conformal.aci import AciController
from conformal_demand_forecaster.data.generator import GeneratorConfig, RegimeShift, generate_panel
from conformal_demand_forecaster.data.panel import Panel
from conformal_demand_forecaster.evaluation.metrics import coverage, mean_width
from conformal_demand_forecaster.evaluation.report import (
    check_thresholds,
    load_thresholds,
    machine_info,
    markdown_table,
    write_results,
)
from conformal_demand_forecaster.features.builder import FeatureBuilder
from conformal_demand_forecaster.models.gbm import GbmConfig
from conformal_demand_forecaster.training import TrainingConfig, forecast_rows, train_at

BENCH = "shift"
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / f"{BENCH}.json"
THRESHOLDS = ROOT / "bench" / "thresholds.json"

FloatArray = npt.NDArray[np.float64]
ALPHAS = (0.2, 0.05)
HORIZON = 7
# The shipped Settings.aci_gamma; bench/bench_shift.py --sweep is the evidence for it.
DEFAULT_GAMMA = 0.08


def _tag(alpha: float) -> str:
    return f"{round((1 - alpha) * 100):d}"


def _training(gamma: float) -> TrainingConfig:
    return TrainingConfig(
        gbm=GbmConfig(
            horizon_buckets=((1, HORIZON),),
            num_iterations=60,
            num_leaves=15,
            min_data_in_leaf=20,
            num_threads=4,
        ),
        alphas=ALPHAS,
        train_days=300,
        train_stride=7,
        calibration_days=42,
        min_group_size=30,
        aci_gamma=gamma,
    )


class _Tally:
    """Hits, totals and widths for one method, split into the pre- and post-shift stretches."""

    def __init__(self) -> None:
        self.y: dict[str, list[float]] = {"pre": [], "post": []}
        self.lower: dict[str, list[float]] = {"pre": [], "post": []}
        self.upper: dict[str, list[float]] = {"pre": [], "post": []}

    def add(self, phase: str, y: FloatArray, lower: FloatArray, upper: FloatArray) -> None:
        self.y[phase].extend(y.tolist())
        self.lower[phase].extend(lower.tolist())
        self.upper[phase].extend(upper.tolist())

    def coverage(self, phase: str) -> float:
        return coverage(self.y[phase], self.lower[phase], self.upper[phase])

    def width(self, phase: str) -> float:
        return mean_width(self.lower[phase], self.upper[phase])


def run_shift(
    *, n_series: int, shift_day: int, post_days: int, multiplier: float, gamma: float, seed: int
) -> dict[str, float]:
    config = GeneratorConfig(
        n_series=n_series,
        n_days=shift_day + post_days + HORIZON + 1,
        seed=seed,
        shifts=(RegimeShift(day=shift_day, fraction=1.0, multiplier=multiplier),),
    )
    panel = Panel.from_frame(generate_panel(config), validate=False)
    pre_days = 28
    origin0 = shift_day - pre_days - 1  # the last day the frozen model ever saw

    training = _training(gamma)
    bundle = train_at(panel, origin0, training)
    calibrator = bundle.calibrator
    builder = FeatureBuilder(panel)
    aci = AciController([g.label for g in calibrator.all_groups()], ALPHAS, gamma)

    tallies = {("static", a): _Tally() for a in ALPHAS}
    tallies.update({("aci", a): _Tally() for a in ALPHAS})
    demand = panel.demand if panel.demand is not None else panel.sales

    for day in range(origin0, shift_day + post_days):
        rows = builder.rows([day], [1])  # one-step forecasts, scored the next day
        _, static = forecast_rows(bundle.model, calibrator, bundle.scores, panel, rows)
        _, adaptive = forecast_rows(
            bundle.model,
            calibrator,
            bundle.scores,
            panel,
            rows,
            alpha_effective=aci.effective_alphas(),
        )
        y = demand[rows.series_index, rows.target_day]
        observed = ~np.isnan(y) & (rows.censored < 0.5)
        if not observed.any():
            continue
        phase = "pre" if day < shift_day else "post"
        labels = np.array(
            calibrator.group_labels([panel.segments[s] for s in rows.series_index], rows.horizon)
        )
        for alpha in ALPHAS:
            for name, intervals in (("static", static), ("aci", adaptive)):
                lower, upper = intervals.lower[alpha], intervals.upper[alpha]
                tallies[(name, alpha)].add(phase, y[observed], lower[observed], upper[observed])
            # ACI sees only its own band's misses, once per group per day.
            lower, upper = adaptive.lower[alpha], adaptive.upper[alpha]
            missed = (y < lower) | (y > upper)
            for label in np.unique(labels[observed]):
                mask = observed & (labels == label)
                aci.update(str(label), alpha, int(missed[mask].sum()), int(mask.sum()))

    metrics: dict[str, float] = {
        "n_series": float(n_series),
        "shift_multiplier": multiplier,
        "post_days": float(post_days),
        "aci_gamma": gamma,
    }
    for alpha in ALPHAS:
        tag = _tag(alpha)
        nominal = 1.0 - alpha
        for name in ("static", "aci"):
            tally = tallies[(name, alpha)]
            for phase in ("pre", "post"):
                metrics[f"coverage_{tag}_{name}_{phase}"] = round(tally.coverage(phase), 5)
                metrics[f"width_{tag}_{name}_{phase}"] = round(tally.width(phase), 4)
            metrics[f"error_{tag}_{name}_post"] = round(
                abs(tallies[(name, alpha)].coverage("post") - nominal), 5
            )
        metrics[f"aci_improvement_{tag}"] = round(
            metrics[f"error_{tag}_static_post"] - metrics[f"error_{tag}_aci_post"], 5
        )
        metrics[f"aci_width_cost_{tag}"] = round(
            metrics[f"width_{tag}_aci_post"] / max(metrics[f"width_{tag}_static_post"], 1e-9), 4
        )
    return metrics


SWEEP_GAMMAS = (0.03, 0.05, 0.08, 0.12)


def run_sweep(args: argparse.Namespace) -> list[list[object]]:
    """Adaptation speed against interval width, for the ADR that picks the shipped gamma."""
    rows: list[list[object]] = []
    for gamma in SWEEP_GAMMAS:
        result = run_shift(
            n_series=args.series,
            shift_day=args.shift_day,
            post_days=args.post_days,
            multiplier=args.multiplier,
            gamma=gamma,
            seed=args.seed,
        )
        rows.append(
            [
                gamma,
                result["error_80_aci_post"],
                result["error_95_aci_post"],
                result["width_80_aci_post"],
                result["width_95_aci_post"],
            ]
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="static split conformal vs ACI under a shift")
    parser.add_argument("--series", type=int, default=25)
    parser.add_argument("--shift-day", type=int, default=420)
    parser.add_argument("--post-days", type=int, default=200)
    parser.add_argument("--multiplier", type=float, default=2.2)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument(
        "--sweep", action="store_true", help="also report the gamma trade-off table"
    )
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    metrics = run_shift(
        n_series=args.series,
        shift_day=args.shift_day,
        post_days=args.post_days,
        multiplier=args.multiplier,
        gamma=args.gamma,
        seed=args.seed,
    )
    metrics["seconds"] = round(time.perf_counter() - started, 2)
    write_results(
        RESULTS,
        {"bench": BENCH, "machine": machine_info(), "args": vars(args), "metrics": metrics},
    )
    print(
        markdown_table(
            ["level", "method", "coverage before", "coverage after", "width after"],
            [
                [
                    f"{_tag(alpha)}%",
                    name,
                    metrics[f"coverage_{_tag(alpha)}_{name}_pre"],
                    metrics[f"coverage_{_tag(alpha)}_{name}_post"],
                    metrics[f"width_{_tag(alpha)}_{name}_post"],
                ]
                for alpha in ALPHAS
                for name in ("static", "aci")
            ],
        )
    )
    if args.sweep:
        print()
        print(
            markdown_table(
                ["gamma", "80% error", "95% error", "80% width", "95% width"], run_sweep(args)
            )
        )
    if args.check:
        problems = check_thresholds(metrics, load_thresholds(THRESHOLDS, BENCH))
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all shift metrics within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
