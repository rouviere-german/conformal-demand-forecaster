"""Train the global quantile model at an origin and calibrate it on a disjoint window.

Timeline for an origin ``T`` (the last observed day used), horizon ``H`` and calibration window
``C``::

    training targets            calibration targets       forecasts
    ... <= T - C]               (T - C, T]                (T, T + H]

Training rows come from origins whose every target lies at or before ``T - C``; calibration rows
from later origins whose targets fall inside the window. The two target sets are disjoint, which
is what split conformal calibration needs. Calibration scores are computed against observed
sales with stockout-censored targets excluded, because a censored sale is only a lower bound.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

from conformal_demand_forecaster.conformal.cqr import IntervalSet, MondrianCalibrator, ScoreBuffer
from conformal_demand_forecaster.core.canonical import sha256_hex
from conformal_demand_forecaster.data.panel import Panel
from conformal_demand_forecaster.features.builder import FEATURE_NAMES, FeatureBuilder, FeatureRows
from conformal_demand_forecaster.ml.contracts import SEGMENTS
from conformal_demand_forecaster.models.gbm import GbmConfig, QuantileGbm, QuantilePrediction

__all__ = [
    "TrainedBundle",
    "TrainingConfig",
    "forecast_frame",
    "forecast_rows",
    "model_text_digest",
    "train_at",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
REFERENCE_SAMPLE = 2_000


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    gbm: GbmConfig = field(default_factory=GbmConfig)
    alphas: tuple[float, ...] = (0.2, 0.05)
    train_days: int = 365
    train_stride: int = 7
    calibration_days: int = 56
    calibration_stride: int = 1
    min_group_size: int = 50
    aci_gamma: float = 0.08  # see bench/bench_shift.py --sweep

    def __post_init__(self) -> None:
        if self.train_days < 28 or self.calibration_days < 7:
            raise ValueError("train_days must be >= 28 and calibration_days >= 7")
        if self.train_stride < 1 or self.calibration_stride < 1:
            raise ValueError("strides must be >= 1")
        if not self.alphas or any(not 0.0 < a < 1.0 for a in self.alphas):
            raise ValueError("alphas must lie strictly inside (0, 1)")

    @property
    def horizon(self) -> int:
        return self.gbm.max_horizon

    def calibrator(self) -> MondrianCalibrator:
        return MondrianCalibrator(
            self.alphas,
            self.gbm.horizon_buckets,
            self.gbm.quantiles,
            min_group_size=self.min_group_size,
        )


@dataclass(slots=True)
class TrainedBundle:
    model: QuantileGbm
    calibrator: MondrianCalibrator
    scores: ScoreBuffer
    origin: int
    config: TrainingConfig
    model_text_sha256: str
    reference: dict[str, FloatArray]
    training_rows: int
    calibration_rows: int


def model_text_digest(texts: dict[str, str]) -> str:
    """SHA-256 over the sorted model texts (stable across processes)."""
    joined = "\n".join(f"{key}\n{texts[key]}" for key in sorted(texts))
    return sha256_hex(joined)


def _origins(last: int, first: int, stride: int) -> IntArray:
    """Origins from ``last`` backwards to ``first`` (inclusive) with ``stride``."""
    if last < first:
        return np.zeros(0, dtype=np.int64)
    return np.arange(last, first - 1, -stride, dtype=np.int64)[::-1]


def _reference_samples(panel: Panel, first: int, last: int) -> dict[str, FloatArray]:
    """Deterministic per-segment samples of observed sales for drift monitoring."""
    reference: dict[str, FloatArray] = {}
    codes = panel.segment_codes
    for index, segment in enumerate(SEGMENTS):
        values = panel.sales[codes == index, first : last + 1].ravel()
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue
        if values.size > REFERENCE_SAMPLE:
            step = values.size / REFERENCE_SAMPLE
            values = values[(np.arange(REFERENCE_SAMPLE) * step).astype(np.int64)]
        reference[f"sales__{segment}"] = values
    return reference


def train_at(panel: Panel, origin: int, config: TrainingConfig) -> TrainedBundle:
    """Fit the quantile model and fill a calibration buffer for forecasts issued at ``origin``."""
    if not 0 <= origin <= panel.last_observed_day:
        raise ValueError("origin must be an observed day of the panel")
    horizon = config.horizon
    calibration_start = origin - config.calibration_days  # exclusive lower bound on targets
    train_last = calibration_start - horizon
    train_first = max(0, train_last - config.train_days + 1)
    if train_last < train_first + 28:
        raise ValueError("not enough history before the calibration window to train")
    # The calibration rows are the cartesian product (origin-1 back to the window start) x
    # horizons, so the widest target lands `horizon` days after the last calibration origin --
    # past the end of the panel when training at the last observed day. Build the rows on a
    # panel extended by that much; the rows with unobserved targets are filtered out below.
    overhang = max(0, (origin - 1) + horizon + 1 - panel.n_days)
    builder = FeatureBuilder(panel.extend(overhang))
    horizons = np.arange(1, horizon + 1, dtype=np.int64)

    train = builder.rows(_origins(train_last, train_first, config.train_stride), horizons)
    train = train.subset(train.observed)
    model = QuantileGbm(config.gbm, FEATURE_NAMES).fit(
        train.X, train.y_sales, train.horizon, censored=train.censored
    )

    calibration = builder.rows(
        _origins(origin - 1, max(0, calibration_start + 1 - horizon), config.calibration_stride),
        horizons,
    )
    keep = (
        calibration.observed
        & (calibration.target_day > calibration_start)
        & (calibration.target_day <= origin)
        & (calibration.censored < 0.5)
    )
    calibration = calibration.subset(keep)
    calibrator = config.calibrator()
    scores = ScoreBuffer(config.calibration_days)
    if len(calibration):
        prediction = model.predict(calibration.X, calibration.horizon)
        labels = np.array(
            calibrator.group_labels(
                [panel.segments[s] for s in calibration.series_index], calibration.horizon
            )
        )
        for alpha, values in calibrator.scores(prediction, calibration.y_sales).items():
            for label in np.unique(labels):
                mask = labels == label
                scores.append((str(label), alpha), calibration.target_day[mask], values[mask])
    return TrainedBundle(
        model=model,
        calibrator=calibrator,
        scores=scores,
        origin=origin,
        config=config,
        model_text_sha256=model_text_digest(model.to_text()),
        reference=_reference_samples(panel, train_first, origin),
        training_rows=len(train),
        calibration_rows=len(calibration),
    )


def forecast_rows(
    model: QuantileGbm,
    calibrator: MondrianCalibrator,
    scores: ScoreBuffer,
    panel: Panel,
    rows: FeatureRows,
    *,
    alpha_effective: Mapping[tuple[str, float], float] | None = None,
    mondrian: bool = True,
) -> tuple[QuantilePrediction, IntervalSet]:
    """Raw quantiles and conformalized intervals for prepared feature rows."""
    prediction = model.predict(rows.X, rows.horizon)
    segments = [panel.segments[s] for s in rows.series_index]
    intervals = calibrator.intervals(
        prediction,
        segments,
        rows.horizon,
        scores,
        alpha_effective=alpha_effective,
        mondrian=mondrian,
    )
    return prediction, intervals


def forecast_frame(
    calibrator: MondrianCalibrator,
    panel: Panel,
    rows: FeatureRows,
    prediction: QuantilePrediction,
    intervals: IntervalSet,
) -> pd.DataFrame:
    """Long frame with one row per (series, origin, target): raw quantiles and intervals."""
    frame = pd.DataFrame(
        {
            "series_id": np.array(panel.series_ids)[rows.series_index],
            "segment": np.array(panel.segments)[rows.series_index],
            "origin_date": panel.dates[rows.origin],
            "target_date": panel.dates[rows.target_day],
            "horizon": rows.horizon,
            "point": prediction.point,
        }
    )
    for level in prediction.levels:
        frame[f"q{level:.3f}"] = prediction.column(level)
    for alpha in calibrator.alphas:
        tag = f"{round((1 - alpha) * 100):d}"
        frame[f"lower_{tag}"] = intervals.lower[alpha]
        frame[f"upper_{tag}"] = intervals.upper[alpha]
        frame[f"alpha_eff_{tag}"] = intervals.alpha_effective[alpha]
        frame[f"saturated_{tag}"] = intervals.saturated[alpha]
    return frame
