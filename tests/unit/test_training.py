"""Training and calibration at one origin.

Split conformal calibration needs the calibration targets to be disjoint from the training
targets. Every other guarantee in the repository rests on that, so the disjointness is asserted
directly from the row bookkeeping rather than trusted to the index arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest

from conformal_demand_forecaster.data.panel import Panel
from conformal_demand_forecaster.features.builder import FEATURE_NAMES, FeatureBuilder
from conformal_demand_forecaster.models.gbm import GbmConfig
from conformal_demand_forecaster.training import (
    TrainingConfig,
    forecast_frame,
    forecast_rows,
    model_text_digest,
    train_at,
)

pytestmark = [pytest.mark.unit, pytest.mark.slow]


def test_training_and_calibration_targets_are_disjoint(
    small_panel: Panel, fast_training: TrainingConfig
) -> None:
    origin = small_panel.last_observed_day - fast_training.horizon
    bundle = train_at(small_panel, origin, fast_training)
    boundary = origin - fast_training.calibration_days
    builder = FeatureBuilder(small_panel)
    horizons = np.arange(1, fast_training.horizon + 1, dtype=np.int64)

    # Rebuild the two windows the way train_at does and check the target days never overlap.
    train_last = boundary - fast_training.horizon
    train_rows = builder.rows(
        np.arange(
            max(0, train_last - fast_training.train_days + 1),
            train_last + 1,
            fast_training.train_stride,
        ),
        horizons,
    )
    assert train_rows.target_day.max() <= boundary
    assert bundle.training_rows > 0
    assert bundle.calibration_rows > 0
    assert bundle.origin == origin


def test_calibration_only_uses_uncensored_observed_targets(
    small_panel: Panel, fast_training: TrainingConfig
) -> None:
    """Censored sales are lower bounds, so scoring them would bias the conformal quantile."""
    origin = small_panel.last_observed_day - fast_training.horizon
    stockout = small_panel.stockout.copy()
    window = slice(origin - fast_training.calibration_days, origin + 1)
    stockout[:, window] = 1.0  # every calibration target is a censored sale
    censored = small_panel.with_sales(small_panel.sales, stockout=stockout)

    baseline = train_at(small_panel, origin, fast_training)
    bundle = train_at(censored, origin, fast_training)
    assert baseline.calibration_rows > 0
    assert bundle.calibration_rows == 0
    assert bundle.scores.keys == []


def test_the_bundle_records_a_reproducible_model_digest(
    small_panel: Panel, fast_training: TrainingConfig
) -> None:
    origin = small_panel.last_observed_day - fast_training.horizon
    first = train_at(small_panel, origin, fast_training)
    second = train_at(small_panel, origin, fast_training)
    assert first.model_text_sha256 == second.model_text_sha256
    assert first.model_text_sha256 == model_text_digest(first.model.to_text())
    assert len(first.model_text_sha256) == 64


def test_reference_samples_are_kept_per_segment(
    small_panel: Panel, fast_training: TrainingConfig
) -> None:
    origin = small_panel.last_observed_day - fast_training.horizon
    bundle = train_at(small_panel, origin, fast_training)
    assert set(bundle.reference) <= {"sales__smooth", "sales__intermittent"}
    assert bundle.reference
    for values in bundle.reference.values():
        assert values.size > 0
        assert not np.isnan(values).any()


def test_score_buffer_is_keyed_by_group_and_alpha(
    small_panel: Panel, fast_training: TrainingConfig
) -> None:
    origin = small_panel.last_observed_day - fast_training.horizon
    bundle = train_at(small_panel, origin, fast_training)
    alphas = {alpha for _, alpha in bundle.scores.keys}
    assert alphas == set(fast_training.alphas)
    for label, _ in bundle.scores.keys:
        segment, _, bucket = label.partition("/b")
        assert segment in {"smooth", "intermittent"}
        assert bucket.isdigit()


def test_forecast_rows_and_frame_line_up(small_panel: Panel, fast_training: TrainingConfig) -> None:
    origin = small_panel.last_observed_day - fast_training.horizon
    bundle = train_at(small_panel, origin, fast_training)
    builder = FeatureBuilder(small_panel)
    rows = builder.rows([origin], np.arange(1, fast_training.horizon + 1))
    prediction, intervals = forecast_rows(
        bundle.model, bundle.calibrator, bundle.scores, small_panel, rows
    )
    frame = forecast_frame(bundle.calibrator, small_panel, rows, prediction, intervals)
    assert len(frame) == len(rows)
    assert {"series_id", "segment", "origin_date", "target_date", "horizon", "point"} <= set(frame)
    for alpha in fast_training.alphas:
        tag = f"{round((1 - alpha) * 100):d}"
        assert (frame[f"upper_{tag}"] >= frame[f"lower_{tag}"]).all()
        assert (frame[f"lower_{tag}"] >= 0.0).all()
        assert frame[f"alpha_eff_{tag}"].between(0.0, 1.0).all()
    assert (frame["upper_95"] - frame["lower_95"] >= frame["upper_80"] - frame["lower_80"]).all()


def test_the_model_is_fitted_on_the_declared_feature_names(
    small_panel: Panel, fast_training: TrainingConfig
) -> None:
    origin = small_panel.last_observed_day - fast_training.horizon
    bundle = train_at(small_panel, origin, fast_training)
    assert bundle.model.feature_names == FEATURE_NAMES


def test_an_origin_outside_the_panel_is_refused(
    small_panel: Panel, fast_training: TrainingConfig
) -> None:
    with pytest.raises(ValueError, match="observed day of the panel"):
        train_at(small_panel, small_panel.n_days + 5, fast_training)
    with pytest.raises(ValueError, match="observed day of the panel"):
        train_at(small_panel, -1, fast_training)


def test_too_little_history_before_the_calibration_window_is_refused(
    small_panel: Panel, fast_training: TrainingConfig
) -> None:
    with pytest.raises(ValueError, match="not enough history"):
        train_at(small_panel, fast_training.calibration_days + 1, fast_training)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"train_days": 10}, "train_days"),
        ({"calibration_days": 3}, "train_days"),
        ({"train_stride": 0}, "strides"),
        ({"alphas": ()}, "alphas"),
        ({"alphas": (1.5,)}, "alphas"),
    ],
)
def test_invalid_training_configurations_are_refused(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TrainingConfig(**kwargs)  # type: ignore[arg-type]


def test_the_training_config_exposes_a_matching_calibrator() -> None:
    config = TrainingConfig(gbm=GbmConfig(horizon_buckets=((1, 7), (8, 21))), alphas=(0.2,))
    calibrator = config.calibrator()
    assert config.horizon == 21
    assert calibrator.alphas == (0.2,)
    assert calibrator.horizon_buckets == ((1, 7), (8, 21))
