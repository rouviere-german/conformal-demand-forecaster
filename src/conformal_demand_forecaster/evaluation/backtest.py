"""Rolling-origin backtest: retrain, recalibrate and forecast at every origin.

At each origin the model is retrained and CQR is recalibrated on the window before it (see
:mod:`conformal_demand_forecaster.training`). Three interval constructions are scored from the
same raw quantiles:

- ``mondrian``: per (segment, horizon bucket) conformal quantiles at the nominal alpha;
- ``pooled``: one conformal quantile per alpha over all groups (the non-Mondrian comparator);
- ``aci``: Mondrian quantiles at the ACI-adjusted alpha, where the alpha state is updated once
  per realized day between origins from that day's miss rate.

Baselines (seasonal naive, ETS) are scored on the same rows. Everything is scored against latent
demand; the coverage against censored sales is reported separately.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from conformal_demand_forecaster.conformal.aci import AciController
from conformal_demand_forecaster.conformal.cqr import MondrianCalibrator
from conformal_demand_forecaster.core.canonical import canonical_json, sha256_hex
from conformal_demand_forecaster.data.panel import Panel
from conformal_demand_forecaster.decisions.newsvendor import order_quantities
from conformal_demand_forecaster.evaluation.metrics import (
    coverage,
    mean_width,
    newsvendor_cost,
    winkler,
)
from conformal_demand_forecaster.features.builder import FeatureBuilder
from conformal_demand_forecaster.ml.contracts import SEGMENTS
from conformal_demand_forecaster.models.baselines import (
    EtsBaseline,
    SeasonalNaiveBaseline,
    mase_scale,
)
from conformal_demand_forecaster.training import (
    TrainedBundle,
    TrainingConfig,
    forecast_rows,
    train_at,
)

__all__ = ["BacktestConfig", "BacktestResult", "run_backtest"]

FloatArray = npt.NDArray[np.float64]
METHODS = ("mondrian", "pooled", "aci")
# Metrics that measure the machine rather than the model; excluded from the digest.
TIMING_METRICS = frozenset({"seconds"})


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    n_origins: int = 12
    origin_stride: int = 28
    training: TrainingConfig = field(default_factory=TrainingConfig)
    baselines: bool = True
    ets_series_limit: int | None = None
    holding_cost: float = 1.0
    stockout_cost: float = 4.0

    def __post_init__(self) -> None:
        if self.n_origins < 1 or self.origin_stride < 1:
            raise ValueError("n_origins and origin_stride must be >= 1")


@dataclass(slots=True)
class BacktestResult:
    metrics: dict[str, float]
    per_origin: list[dict[str, float]]
    rows: pd.DataFrame
    seconds: float

    def digest(self) -> str:
        """Hash of the rounded metrics: identical runs must reproduce it exactly.

        Wall-clock timings are excluded -- they are diagnostics, not results, and including
        them would make the digest differ on every run.
        """
        rounded = {
            key: round(value, 6)
            for key, value in sorted(self.metrics.items())
            if key not in TIMING_METRICS
        }
        return sha256_hex(canonical_json(rounded))


def _tag(alpha: float) -> str:
    return f"{round((1 - alpha) * 100):d}"


def _baseline_columns(
    panel: Panel,
    origin: int,
    horizons: npt.NDArray[np.int64],
    *,
    levels: tuple[float, ...],
    ets: EtsBaseline | None,
    ets_limit: int | None,
) -> dict[str, FloatArray]:
    naive = SeasonalNaiveBaseline(levels)
    n_h = horizons.size
    out: dict[str, list[FloatArray]] = {"snaive_point": [], "ets_point": []}
    naive_q: list[FloatArray] = []
    ets_q: list[FloatArray] = []
    scales: list[float] = []
    for index in range(panel.n_series):
        history = panel.sales[index, : origin + 1]
        naive_forecast = naive.forecast(history, horizons)
        out["snaive_point"].append(naive_forecast.point)
        naive_q.append(naive_forecast.quantiles)
        if ets is not None and (ets_limit is None or index < ets_limit):
            ets_forecast = ets.forecast(history, horizons, series_key=panel.series_ids[index])
        else:
            ets_forecast = naive_forecast
        out["ets_point"].append(ets_forecast.point)
        ets_q.append(ets_forecast.quantiles)
        scales.extend([mase_scale(history)] * n_h)
    columns: dict[str, FloatArray] = {key: np.concatenate(v) for key, v in out.items()}
    columns["scale"] = np.asarray(scales)
    naive_matrix = np.concatenate(naive_q)
    ets_matrix = np.concatenate(ets_q)
    for j, level in enumerate(levels):
        columns[f"snaive_q{level:.3f}"] = naive_matrix[:, j]
        columns[f"ets_q{level:.3f}"] = ets_matrix[:, j]
    return columns


def _update_aci_daily(
    aci: AciController, rows: pd.DataFrame, alphas: tuple[float, ...], labels: np.ndarray
) -> None:
    """Replay the realized days of one origin's forecasts through the ACI recursion."""
    y = rows["y_sales"].to_numpy()
    observed = ~np.isnan(y) & (rows["censored"].to_numpy() < 0.5)
    for day in sorted(rows.loc[observed, "target_day"].unique()):
        today = observed & (rows["target_day"].to_numpy() == day)
        for label in np.unique(labels[today]):
            mask = today & (labels == label)
            for alpha in alphas:
                tag = _tag(alpha)
                lo = rows[f"aci_lower_{tag}"].to_numpy()[mask]
                hi = rows[f"aci_upper_{tag}"].to_numpy()[mask]
                misses = int(np.sum((y[mask] < lo) | (y[mask] > hi)))
                aci.update(str(label), alpha, misses, int(mask.sum()))


def _summarize(
    rows: pd.DataFrame,
    config: BacktestConfig,
    levels: tuple[float, ...],
    pairs: Mapping[float, tuple[float, float]],
) -> dict[str, float]:
    alphas = config.training.alphas
    y = rows["y_demand"].to_numpy()
    y_sales = rows["y_sales"].to_numpy()
    scale = rows["scale"].to_numpy()
    metrics: dict[str, float] = {}

    def per_series_mean(values: FloatArray) -> float:
        frame = pd.DataFrame(
            {"key": rows["series_id"].astype(str) + rows["origin_day"].astype(str)}
        )
        frame["v"] = values
        grouped = frame.groupby("key")["v"].mean()
        return float(grouped[np.isfinite(grouped)].mean())

    for name, point in (
        ("gbm", "point"),
        ("gbm_median", "q0.500"),
        ("snaive", "snaive_point"),
        ("ets", "ets_point"),
    ):
        errors = np.abs(y - rows[point].to_numpy()) / np.where(scale > 0, scale, np.nan)
        metrics[f"mase_{name}"] = per_series_mean(errors)
    errors = np.abs(y_sales - rows["point"].to_numpy()) / np.where(scale > 0, scale, np.nan)
    metrics["mase_gbm_vs_sales"] = per_series_mean(errors)

    weights = rows["weight"].to_numpy()
    for name, prefix in (("gbm", "q"), ("snaive", "snaive_q"), ("ets", "ets_q")):
        losses = np.zeros(len(rows))
        for level in levels:
            diff = y - rows[f"{prefix}{level:.3f}"].to_numpy()
            losses += np.maximum(level * diff, (level - 1.0) * diff) / len(levels)
        scaled = losses / np.where(scale > 0, scale, np.nan)
        valid = np.isfinite(scaled)
        metrics[f"wspl_{name}"] = float(
            np.sum(scaled[valid] * weights[valid]) / np.sum(weights[valid])
        )

    segments = rows["segment"].to_numpy()
    for alpha in alphas:
        tag = _tag(alpha)
        nominal = 1.0 - alpha
        for method in METHODS:
            lo = rows[f"{method}_lower_{tag}"].to_numpy()
            hi = rows[f"{method}_upper_{tag}"].to_numpy()
            metrics[f"coverage_{tag}_{method}"] = coverage(y, lo, hi)
            metrics[f"width_{tag}_{method}"] = mean_width(lo, hi)
            metrics[f"winkler_{tag}_{method}"] = winkler(y, lo, hi, alpha)
            gaps = [
                abs(
                    coverage(y[segments == seg], lo[segments == seg], hi[segments == seg]) - nominal
                )
                for seg in SEGMENTS
                if np.any(segments == seg)
            ]
            metrics[f"segment_gap_{tag}_{method}"] = max(gaps) if gaps else float("nan")
            for seg in SEGMENTS:
                mask = segments == seg
                if np.any(mask):
                    metrics[f"coverage_{tag}_{method}_{seg}"] = coverage(
                        y[mask], lo[mask], hi[mask]
                    )
        lo = rows[f"mondrian_lower_{tag}"].to_numpy()
        hi = rows[f"mondrian_upper_{tag}"].to_numpy()
        metrics[f"coverage_{tag}_mondrian_vs_sales"] = coverage(y_sales, lo, hi)
        lo_level, hi_level = pairs[alpha]
        for baseline in ("snaive", "ets"):
            lo_b = rows[f"{baseline}_q{lo_level:.3f}"].to_numpy()
            hi_b = rows[f"{baseline}_q{hi_level:.3f}"].to_numpy()
            metrics[f"coverage_{tag}_{baseline}"] = coverage(y, lo_b, hi_b)
        metrics[f"aci_long_run_error_{tag}"] = abs(metrics[f"coverage_{tag}_aci"] - nominal)

    intervals = {
        alpha: (
            rows[f"mondrian_lower_{_tag(alpha)}"].to_numpy(),
            rows[f"mondrian_upper_{_tag(alpha)}"].to_numpy(),
        )
        for alpha in alphas
    }
    orders = order_quantities(
        rows["q0.500"].to_numpy(), intervals, config.holding_cost, config.stockout_cost
    )
    metrics["newsvendor_cost_conformal"] = newsvendor_cost(
        y, orders, config.holding_cost, config.stockout_cost
    )
    metrics["newsvendor_cost_point"] = newsvendor_cost(
        y, rows["point"].to_numpy(), config.holding_cost, config.stockout_cost
    )
    metrics["newsvendor_cost_median"] = newsvendor_cost(
        y, rows["q0.500"].to_numpy(), config.holding_cost, config.stockout_cost
    )
    metrics["newsvendor_cost_ratio"] = (
        metrics["newsvendor_cost_conformal"] / metrics["newsvendor_cost_point"]
    )
    metrics["rows"] = float(len(rows))
    return metrics


@dataclass(frozen=True, slots=True)
class _OriginContext:
    """Everything one origin of the backtest needs besides the panel."""

    config: BacktestConfig
    builder: FeatureBuilder
    calibrator: MondrianCalibrator
    aci: AciController
    ets: EtsBaseline | None
    horizons: npt.NDArray[np.int64]
    levels: tuple[float, ...]


def _forecast_origin(
    panel: Panel, origin: int, ctx: _OriginContext
) -> tuple[pd.DataFrame, TrainedBundle]:
    """Train at ``origin`` and score the three interval constructions on its horizon."""
    training = ctx.config.training
    bundle = train_at(panel, origin, training)
    test = ctx.builder.rows([origin], ctx.horizons)
    model, scores = bundle.model, bundle.scores
    prediction, mondrian = forecast_rows(model, ctx.calibrator, scores, panel, test)
    _, pooled = forecast_rows(model, ctx.calibrator, scores, panel, test, mondrian=False)
    _, adaptive = forecast_rows(
        model, ctx.calibrator, scores, panel, test, alpha_effective=ctx.aci.effective_alphas()
    )
    segments = [panel.segments[s] for s in test.series_index]
    labels = np.array(ctx.calibrator.group_labels(segments, test.horizon))
    columns: dict[str, Any] = {
        "series_id": np.array(panel.series_ids)[test.series_index],
        "segment": np.array(segments),
        "group": labels,
        "origin_day": test.origin,
        "target_day": test.target_day,
        "horizon": test.horizon,
        "y_demand": test.y_demand if test.y_demand is not None else test.y_sales,
        "y_sales": test.y_sales,
        "censored": test.censored,
        "point": prediction.point,
    }
    for level in ctx.levels:
        columns[f"q{level:.3f}"] = prediction.column(level)
    for alpha in training.alphas:
        tag = _tag(alpha)
        for method, intervals in (("mondrian", mondrian), ("pooled", pooled), ("aci", adaptive)):
            columns[f"{method}_lower_{tag}"] = intervals.lower[alpha]
            columns[f"{method}_upper_{tag}"] = intervals.upper[alpha]
        columns[f"aci_alpha_{tag}"] = adaptive.alpha_effective[alpha]
    columns.update(
        _baseline_columns(
            panel,
            origin,
            ctx.horizons,
            levels=ctx.levels,
            ets=ctx.ets,
            ets_limit=ctx.config.ets_series_limit,
        )
    )
    recent = np.nanmean(panel.sales[:, max(0, origin - 27) : origin + 1], axis=1)
    columns["weight"] = np.repeat(np.maximum(recent, 1e-3), len(ctx.horizons))
    frame = pd.DataFrame(columns)
    _update_aci_daily(ctx.aci, frame, training.alphas, labels)
    return frame, bundle


def _origin_summary(
    frame: pd.DataFrame, origin: int, bundle: TrainedBundle, seconds: float
) -> dict[str, float]:
    summary: dict[str, float] = {
        "origin_day": float(origin),
        "seconds": round(seconds, 2),
        "training_rows": float(bundle.training_rows),
        "calibration_rows": float(bundle.calibration_rows),
    }
    y = frame["y_demand"].to_numpy()
    for alpha in bundle.config.alphas:
        tag = _tag(alpha)
        for method in ("mondrian", "aci"):
            summary[f"coverage_{tag}_{method}"] = coverage(
                y, frame[f"{method}_lower_{tag}"], frame[f"{method}_upper_{tag}"]
            )
    return summary


def run_backtest(panel: Panel, config: BacktestConfig | None = None) -> BacktestResult:
    """Retrain, recalibrate and score every origin; returns metrics, rows and per-origin timings."""
    cfg = config or BacktestConfig()
    training = cfg.training
    horizon = training.horizon
    horizons = np.arange(1, horizon + 1, dtype=np.int64)
    levels = training.gbm.quantiles
    last = panel.last_observed_day
    first_origin = last - horizon - (cfg.n_origins - 1) * cfg.origin_stride
    if first_origin < training.calibration_days + horizon + training.train_days // 2:
        raise ValueError("panel too short for the requested origins and training window")
    origins = [first_origin + j * cfg.origin_stride for j in range(cfg.n_origins)]
    calibrator = training.calibrator()
    ctx = _OriginContext(
        config=cfg,
        builder=FeatureBuilder(panel),
        calibrator=calibrator,
        aci=AciController(
            [g.label for g in calibrator.all_groups()], training.alphas, training.aci_gamma
        ),
        ets=EtsBaseline(levels) if cfg.baselines else None,
        horizons=horizons,
        levels=levels,
    )
    started = time.perf_counter()
    frames: list[pd.DataFrame] = []
    per_origin: list[dict[str, float]] = []
    for origin in origins:
        t0 = time.perf_counter()
        frame, bundle = _forecast_origin(panel, origin, ctx)
        frames.append(frame)
        per_origin.append(_origin_summary(frame, origin, bundle, time.perf_counter() - t0))

    rows = pd.concat(frames, ignore_index=True)
    observed = ~np.isnan(rows["y_demand"].to_numpy())
    rows = rows.loc[observed].reset_index(drop=True)
    metrics = _summarize(rows, cfg, levels, calibrator.pairs)
    metrics["origins"] = float(len(origins))
    for alpha in training.alphas:
        tag = _tag(alpha)
        metrics[f"aci_alpha_final_{tag}"] = float(
            np.mean([s.effective for (_, a), s in ctx.aci.states.items() if a == alpha])
        )
    seconds = time.perf_counter() - started
    metrics["seconds"] = round(seconds, 2)
    return BacktestResult(metrics=metrics, per_origin=per_origin, rows=rows, seconds=seconds)


def describe(metrics: Mapping[str, float]) -> list[tuple[str, float]]:
    """Ordered (name, value) pairs of the headline metrics for reports."""
    order = [key for key in metrics if not key.startswith(("coverage_", "width_", "winkler_"))]
    order += [key for key in metrics if key.startswith(("coverage_", "width_", "winkler_"))]
    return [(key, metrics[key]) for key in order]
