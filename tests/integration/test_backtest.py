"""Rolling-origin backtest: determinism, the metrics it reports, and ACI versus static conformal.

The backtest is the repository's evidence generator, so it must be reproducible bit for bit
(``BacktestResult.digest`` is asserted across runs) and it must actually compare the three
interval constructions on the same raw quantiles: Mondrian, pooled and ACI-adjusted.

The heavier claims -- accuracy against baselines, coverage at nominal levels, newsvendor cost --
are measured in ``bench/bench_backtest.py``; here they are only smoke-checked so a broken
pipeline fails fast in CI.
"""

from __future__ import annotations

import pandas as pd
import pytest

from conformal_demand_forecaster.data.generator import GeneratorConfig, RegimeShift, generate_panel
from conformal_demand_forecaster.data.panel import Panel
from conformal_demand_forecaster.evaluation.backtest import (
    METHODS,
    TIMING_METRICS,
    BacktestConfig,
    run_backtest,
)
from conformal_demand_forecaster.models.gbm import GbmConfig
from conformal_demand_forecaster.training import TrainingConfig

pytestmark = [pytest.mark.integration, pytest.mark.slow]

TRAINING = TrainingConfig(
    gbm=GbmConfig(
        horizon_buckets=((1, 7), (8, 14)),
        num_iterations=25,
        num_leaves=7,
        min_data_in_leaf=10,
        num_threads=2,
    ),
    train_days=180,
    train_stride=7,
    calibration_days=42,
    min_group_size=20,
)
CONFIG = BacktestConfig(n_origins=3, origin_stride=14, training=TRAINING, ets_series_limit=3)


@pytest.fixture(scope="module")
def panel() -> Panel:
    frame = generate_panel(GeneratorConfig(n_series=10, n_days=340, seed=808, shifts=()))
    return Panel.from_frame(frame, validate=False)


@pytest.fixture(scope="module")
def result(panel: Panel):  # type: ignore[no-untyped-def]
    return run_backtest(panel, CONFIG)


def test_the_backtest_runs_every_origin_and_scores_every_row(result, panel: Panel) -> None:  # type: ignore[no-untyped-def]
    assert result.metrics["origins"] == 3
    assert len(result.per_origin) == 3
    assert result.metrics["rows"] > 0
    assert len(result.rows) == result.metrics["rows"]
    assert set(result.rows["series_id"]) <= set(panel.series_ids)
    assert result.rows["horizon"].between(1, TRAINING.horizon).all()


def test_the_digest_is_reproducible(panel: Panel, result) -> None:  # type: ignore[no-untyped-def]
    """Same panel, same config, same numbers: the reproducibility claim in the model card."""
    repeat = run_backtest(panel, CONFIG)
    assert repeat.digest() == result.digest()
    scores = {k: v for k, v in result.metrics.items() if k not in TIMING_METRICS}
    assert {k: v for k, v in repeat.metrics.items() if k not in TIMING_METRICS} == pytest.approx(
        scores
    )


def test_a_different_panel_changes_the_digest(result) -> None:  # type: ignore[no-untyped-def]
    other = Panel.from_frame(
        generate_panel(GeneratorConfig(n_series=10, n_days=340, seed=809, shifts=())),
        validate=False,
    )
    assert run_backtest(other, CONFIG).digest() != result.digest()


def test_all_three_interval_constructions_are_scored(result) -> None:  # type: ignore[no-untyped-def]
    for method in METHODS:
        for tag in ("80", "95"):
            assert f"coverage_{tag}_{method}" in result.metrics
            assert f"width_{tag}_{method}" in result.metrics
            assert f"winkler_{tag}_{method}" in result.metrics
            assert 0.0 <= result.metrics[f"coverage_{tag}_{method}"] <= 1.0
            assert result.metrics[f"width_{tag}_{method}"] >= 0.0


def test_wider_nominal_levels_give_wider_bands_and_more_coverage(result) -> None:  # type: ignore[no-untyped-def]
    for method in METHODS:
        assert result.metrics[f"width_95_{method}"] >= result.metrics[f"width_80_{method}"]
        assert result.metrics[f"coverage_95_{method}"] >= result.metrics[f"coverage_80_{method}"]


def test_coverage_on_a_stationary_panel_lands_near_nominal(result) -> None:  # type: ignore[no-untyped-def]
    """No regime shift here, so split conformal should be close to its nominal level."""
    assert result.metrics["coverage_80_mondrian"] == pytest.approx(0.8, abs=0.12)
    assert result.metrics["coverage_95_mondrian"] == pytest.approx(0.95, abs=0.08)


def test_per_segment_coverage_and_the_mondrian_gap_are_reported(result) -> None:  # type: ignore[no-untyped-def]
    for segment in ("smooth", "intermittent"):
        assert 0.0 <= result.metrics[f"coverage_80_mondrian_{segment}"] <= 1.0
    assert result.metrics["segment_gap_80_mondrian"] >= 0.0
    assert result.metrics["segment_gap_80_pooled"] >= 0.0


def test_accuracy_metrics_cover_the_model_and_both_baselines(result) -> None:  # type: ignore[no-untyped-def]
    for name in ("gbm", "gbm_median", "snaive", "ets"):
        assert result.metrics[f"mase_{name}"] > 0.0
    for name in ("gbm", "snaive", "ets"):
        assert result.metrics[f"wspl_{name}"] > 0.0
    assert result.metrics["mase_gbm_vs_sales"] > 0.0


def test_censored_scoring_is_reported_separately(result) -> None:  # type: ignore[no-untyped-def]
    """Scoring against censored sales is not the same as scoring against latent demand."""
    against_sales = result.metrics["coverage_80_mondrian_vs_sales"]
    assert 0.0 <= against_sales <= 1.0
    # Censored days make sales a lower bound on demand, so the two can differ; on a panel with
    # few stockouts in the scored window they may coincide, which is why they are reported apart.
    assert against_sales >= result.metrics["coverage_80_mondrian"] - 0.2


def test_the_newsvendor_layer_is_scored_against_point_ordering(result) -> None:  # type: ignore[no-untyped-def]
    assert result.metrics["newsvendor_cost_conformal"] > 0.0
    assert result.metrics["newsvendor_cost_point"] > 0.0
    assert result.metrics["newsvendor_cost_ratio"] == pytest.approx(
        result.metrics["newsvendor_cost_conformal"] / result.metrics["newsvendor_cost_point"]
    )


def test_aci_alpha_state_is_reported_and_bounded(result) -> None:  # type: ignore[no-untyped-def]
    for tag in ("80", "95"):
        assert 0.0 <= result.metrics[f"aci_alpha_final_{tag}"] <= 1.0
        assert result.metrics[f"aci_long_run_error_{tag}"] >= 0.0


def test_per_origin_summaries_carry_timings_and_row_counts(result) -> None:  # type: ignore[no-untyped-def]
    for summary in result.per_origin:
        assert summary["training_rows"] > 0
        assert summary["calibration_rows"] > 0
        assert summary["seconds"] >= 0.0
        assert 0.0 <= summary["coverage_80_mondrian"] <= 1.0


def test_aci_holds_coverage_better_than_static_conformal_after_a_shift() -> None:
    """The central claim of the repo, at smoke scale (bench/bench_shift.py measures it properly)."""
    frame = generate_panel(
        GeneratorConfig(
            n_series=10,
            n_days=420,
            seed=4242,
            shifts=(RegimeShift(day=250, fraction=1.0, multiplier=2.5),),
        )
    )
    shifted = Panel.from_frame(frame, validate=False)
    config = BacktestConfig(n_origins=4, origin_stride=28, training=TRAINING, baselines=False)
    result = run_backtest(shifted, config)
    static_error = abs(result.metrics["coverage_80_mondrian"] - 0.8)
    adaptive_error = abs(result.metrics["coverage_80_aci"] - 0.8)
    assert adaptive_error <= static_error + 0.02


def test_a_panel_that_is_too_short_is_refused(panel: Panel) -> None:
    with pytest.raises(ValueError, match="panel too short"):
        run_backtest(panel.truncate(200), BacktestConfig(n_origins=12, training=TRAINING))


@pytest.mark.parametrize(
    ("kwargs", "message"), [({"n_origins": 0}, "n_origins"), ({"origin_stride": 0}, "n_origins")]
)
def test_invalid_backtest_configurations_are_refused(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        BacktestConfig(**kwargs)  # type: ignore[arg-type]


def test_baselines_can_be_switched_off(panel: Panel) -> None:
    result = run_backtest(
        panel, BacktestConfig(n_origins=2, origin_stride=14, training=TRAINING, baselines=False)
    )
    # Without ETS the baseline columns fall back to seasonal naive rather than disappearing.
    assert result.metrics["mase_ets"] == pytest.approx(result.metrics["mase_snaive"])


def test_the_row_frame_is_usable_for_ad_hoc_analysis(result) -> None:  # type: ignore[no-untyped-def]
    rows: pd.DataFrame = result.rows
    expected = {
        "series_id",
        "segment",
        "group",
        "origin_day",
        "target_day",
        "horizon",
        "y_demand",
        "y_sales",
        "point",
        "weight",
    }
    assert expected <= set(rows.columns)
    assert rows["y_demand"].notna().all()
    assert (rows["mondrian_upper_80"] >= rows["mondrian_lower_80"]).all()
