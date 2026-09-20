"""Prometheus metrics for the forecasting runtime (per-app registry)."""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge

__all__ = ["ForecastMetrics", "create_forecast_metrics"]


@dataclass(frozen=True, slots=True)
class ForecastMetrics:
    registry: CollectorRegistry
    rolling_coverage: Gauge
    interval_width: Gauge
    alpha_effective: Gauge
    coverage_alarm_total: Counter
    coverage_over_alarm_total: Counter
    forecast_freshness_seconds: Gauge
    feature_psi: Gauge
    actuals_accepted_total: Counter
    actuals_quarantined_total: Counter
    actuals_rejected_total: Counter
    newsvendor_fractile_clipped_total: Counter
    calibration_saturated_total: Counter
    recalibration_proposals_total: Counter


def create_forecast_metrics(registry: CollectorRegistry, namespace: str = "app") -> ForecastMetrics:
    return ForecastMetrics(
        registry=registry,
        rolling_coverage=Gauge(
            "rolling_coverage",
            "Empirical interval coverage over the monitor window.",
            ["level", "segment"],
            namespace=namespace,
            registry=registry,
        ),
        interval_width=Gauge(
            "interval_width",
            "Mean conformalized interval width of the latest forecast batch.",
            ["level", "segment"],
            namespace=namespace,
            registry=registry,
        ),
        alpha_effective=Gauge(
            "alpha_effective",
            "ACI-adjusted miscoverage level per group.",
            ["level", "segment", "bucket"],
            namespace=namespace,
            registry=registry,
        ),
        coverage_alarm_total=Counter(
            "coverage_alarm_total",
            "Coverage monitor alarms (binomial test breaches).",
            ["level", "segment"],
            namespace=namespace,
            registry=registry,
        ),
        coverage_over_alarm_total=Counter(
            "coverage_over_alarm_total",
            "Coverage monitor over-coverage findings (bands wider than the level asks for).",
            ["level", "segment"],
            namespace=namespace,
            registry=registry,
        ),
        forecast_freshness_seconds=Gauge(
            "forecast_freshness_seconds",
            "Seconds since the latest forecast origin.",
            namespace=namespace,
            registry=registry,
        ),
        feature_psi=Gauge(
            "feature_psi",
            "Population stability index of recent actuals against the training reference.",
            ["feature"],
            namespace=namespace,
            registry=registry,
        ),
        actuals_accepted_total=Counter(
            "actuals_accepted_total",
            "Ingested actuals that passed the contract and the outlier guard.",
            namespace=namespace,
            registry=registry,
        ),
        actuals_quarantined_total=Counter(
            "actuals_quarantined_total",
            "Ingested actuals quarantined by the outlier guard.",
            ["reason"],
            namespace=namespace,
            registry=registry,
        ),
        actuals_rejected_total=Counter(
            "actuals_rejected_total",
            "Ingested actuals refused before the guard (a day the panel already holds).",
            ["reason"],
            namespace=namespace,
            registry=registry,
        ),
        newsvendor_fractile_clipped_total=Counter(
            "newsvendor_fractile_clipped_total",
            "Orders served at the outermost calibrated quantile instead of the asked fractile.",
            namespace=namespace,
            registry=registry,
        ),
        calibration_saturated_total=Counter(
            "calibration_saturated_total",
            "Forecast rows whose conformal quantile saturated at the widest calibration score.",
            ["level"],
            namespace=namespace,
            registry=registry,
        ),
        recalibration_proposals_total=Counter(
            "recalibration_proposals_total",
            "Recalibration proposals by outcome.",
            ["outcome"],
            namespace=namespace,
            registry=registry,
        ),
    )
