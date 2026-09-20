"""Robustness of the pipeline to the data pathologies and attacks the threat model names.

Each test takes a clean seeded panel, injects one pathology, and asserts the pipeline either
absorbs it or refuses it loudly -- never that it silently produces intervals it cannot back.

- missing days: the contract refuses a gapped panel rather than training on a misaligned lag
- sudden zero demand: accepted everywhere, because a closure is real news for the monitor
- extreme promotions: features stay finite, intervals stay ordered
- adversarial spikes: quarantined by the guard before they can move the calibration
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_demand_forecaster.conformal.cqr import ScoreBuffer, cqr_scores
from conformal_demand_forecaster.data.generator import GeneratorConfig, generate_panel
from conformal_demand_forecaster.data.panel import Panel
from conformal_demand_forecaster.features.builder import FeatureBuilder
from conformal_demand_forecaster.ml.contracts import DataContractError, validate_panel
from conformal_demand_forecaster.serving.guard import RobustZScoreGuard
from conformal_demand_forecaster.training import TrainingConfig, forecast_rows, train_at

pytestmark = [pytest.mark.unit, pytest.mark.slow]


@pytest.fixture(scope="module")
def base_frame() -> pd.DataFrame:
    return generate_panel(GeneratorConfig(n_series=6, n_days=300, seed=2024, shifts=()))


def _forecast(panel: Panel, config: TrainingConfig) -> tuple[np.ndarray, np.ndarray]:
    origin = panel.last_observed_day - config.horizon
    bundle = train_at(panel, origin, config)
    rows = FeatureBuilder(panel).rows([origin], np.arange(1, config.horizon + 1))
    _, intervals = forecast_rows(bundle.model, bundle.calibrator, bundle.scores, panel, rows)
    return intervals.lower[0.2], intervals.upper[0.2]


# --- missing days ---------------------------------------------------------------------------


def test_a_gapped_panel_is_refused_by_the_contract(base_frame: pd.DataFrame) -> None:
    gapped = base_frame.drop(index=base_frame.index[150]).reset_index(drop=True)
    with pytest.raises(DataContractError):
        validate_panel(gapped)


def test_a_panel_with_a_dropped_series_day_is_refused_by_the_container(
    base_frame: pd.DataFrame,
) -> None:
    ragged = base_frame.drop(index=base_frame.index[150]).reset_index(drop=True)
    with pytest.raises(ValueError, match="same contiguous date range"):
        Panel.from_frame(ragged, validate=False)


# --- sudden zero demand ---------------------------------------------------------------------


def test_a_series_that_drops_to_zero_still_produces_ordered_finite_intervals(
    base_frame: pd.DataFrame, fast_training: TrainingConfig
) -> None:
    panel = Panel.from_frame(base_frame, validate=False)
    sales = panel.sales.copy()
    sales[0, 200:] = 0.0  # a store closes
    zeroed = panel.with_sales(sales)
    lower, upper = _forecast(zeroed, fast_training)
    assert np.all(np.isfinite(lower))
    assert np.all(np.isfinite(upper))
    assert np.all(upper >= lower)
    assert np.all(lower >= 0.0)


def test_a_collapse_to_zero_is_never_quarantined_as_an_outlier() -> None:
    """The guard is one-sided on purpose: the coverage monitor must see genuine collapses."""
    guard = RobustZScoreGuard()
    history = np.full(40, 25.0)
    for _ in range(10):
        assert guard.check(0.0, history).accepted is True


# --- extreme promotions ---------------------------------------------------------------------


def test_extreme_price_moves_keep_the_features_finite(base_frame: pd.DataFrame) -> None:
    panel = Panel.from_frame(base_frame, validate=False)
    price = panel.price.copy()
    promo = panel.promo.copy()
    price[:, 250:] *= 0.01  # a 99% discount
    promo[:, 250:] = 1.0
    shocked = Panel(
        series_ids=panel.series_ids,
        segments=panel.segments,
        dates=panel.dates,
        sales=panel.sales,
        price=price,
        promo=promo,
        stockout=panel.stockout,
        demand=panel.demand,
    )
    rows = FeatureBuilder(shocked).rows([260], [1, 7, 28])
    values = rows.X.to_numpy()
    assert np.isfinite(values[~np.isnan(values)]).all()
    assert rows.X["price_ratio_target"].max() < 1.0


def test_extreme_promotions_do_not_break_interval_ordering(
    base_frame: pd.DataFrame, fast_training: TrainingConfig
) -> None:
    panel = Panel.from_frame(base_frame, validate=False)
    promo = np.ones_like(panel.promo)
    shocked = Panel(
        series_ids=panel.series_ids,
        segments=panel.segments,
        dates=panel.dates,
        sales=panel.sales,
        price=panel.price * 0.5,
        promo=promo,
        stockout=panel.stockout,
        demand=panel.demand,
    )
    lower, upper = _forecast(shocked, fast_training)
    assert np.all(upper >= lower)


# --- adversarial spikes ---------------------------------------------------------------------


def test_a_poisoned_spike_would_widen_calibration_if_the_guard_let_it_through() -> None:
    """Quantifies what the guard prevents: one spike moves the 95% conformal quantile a lot."""
    rng = np.random.default_rng(0)
    y = rng.poisson(20, size=200).astype(np.float64)
    lower, upper = np.full(200, 12.0), np.full(200, 28.0)

    clean = ScoreBuffer(window_days=400)
    clean.append(("smooth/b0", 0.05), np.arange(200), cqr_scores(y, lower, upper))
    poisoned_y = y.copy()
    poisoned_y[:10] = 5_000.0
    dirty = ScoreBuffer(window_days=400)
    dirty.append(("smooth/b0", 0.05), np.arange(200), cqr_scores(poisoned_y, lower, upper))

    from conformal_demand_forecaster.conformal.cqr import conformal_quantile

    clean_q = conformal_quantile(clean.scores(("smooth/b0", 0.05)), 0.05)
    dirty_q = conformal_quantile(dirty.scores(("smooth/b0", 0.05)), 0.05)
    assert dirty_q > 10 * max(clean_q, 1.0)

    guard = RobustZScoreGuard()
    assert all(guard.check(5_000.0, y[:60]).accepted is False for _ in range(3))


def test_the_guard_rejects_a_sustained_poisoning_campaign() -> None:
    """Even after a few spikes slip into the history, the median/MAD statistic still flags more."""
    guard = RobustZScoreGuard()
    history = list(np.full(30, 20.0))
    rejected = 0
    for _ in range(10):
        decision = guard.check(4_000.0, np.asarray(history))
        rejected += int(not decision.accepted)
        if decision.accepted:
            history.append(4_000.0)
        history = history[-30:]
    assert rejected == 10


def test_values_beyond_the_contract_cap_never_reach_the_statistic() -> None:
    guard = RobustZScoreGuard()
    decision = guard.check(1e12, np.full(40, 10.0))
    assert decision.accepted is False
    assert decision.reason == "out_of_range"
