"""Offline demo: generate, train, backtest, then stream shifted actuals through the runtime.

The panel has a level shift on a known day. The runtime is installed on the data *before* the
shift, then every post-shift day is ingested as actuals (one adversarial spike included). Each
day the demo records how the frozen split-conformal intervals and the ACI-adjusted intervals
covered the realized demand, so the summary shows static coverage degrading while ACI recovers,
the binomial coverage alarm firing and a recalibration proposal waiting for an operator.

Nothing here reaches the network: the panel comes from the seeded generator and the models are
trained locally in a few seconds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import JsonValue

from conformal_demand_forecaster.core.config import Settings
from conformal_demand_forecaster.data.generator import GeneratorConfig, RegimeShift, generate_panel
from conformal_demand_forecaster.data.panel import Panel
from conformal_demand_forecaster.evaluation.backtest import (
    BacktestConfig,
    BacktestResult,
    run_backtest,
)
from conformal_demand_forecaster.evaluation.metrics import coverage
from conformal_demand_forecaster.models.gbm import GbmConfig
from conformal_demand_forecaster.serving.runtime import ActualRecord, ForecastRuntime, RuntimePaths
from conformal_demand_forecaster.training import TrainingConfig

__all__ = ["DEMO_SEED", "run_demo"]

DEMO_SEED = 20260914
_N_SERIES = 24
_SHIFT_DAY = 560
_POST_DAYS = 70
_SPIKE_DAY = 12  # days after the shift
_SPIKE_FACTOR = 60
_DEMO_TAG = "80"

DemandLookup = dict[tuple[str, pd.Timestamp], float]


def _demo_generator() -> GeneratorConfig:
    return GeneratorConfig(
        n_series=_N_SERIES,
        n_days=_SHIFT_DAY + _POST_DAYS,
        seed=DEMO_SEED,
        shifts=(RegimeShift(day=_SHIFT_DAY, fraction=0.7, multiplier=1.8),),
    )


def _demo_gbm() -> GbmConfig:
    return GbmConfig(num_iterations=80, num_leaves=15, min_data_in_leaf=20, num_threads=2)


def _coverage_of(frame: pd.DataFrame, demand: DemandLookup, tag: str) -> tuple[int, int]:
    """(hits, n) of one forecast frame against latent demand, for the rows we know the truth of."""
    keys = list(zip(frame["series_id"], pd.to_datetime(frame["target_date"]), strict=True))
    known = [key in demand for key in keys]
    if not any(known):
        return 0, 0
    y = np.array([demand[key] for key, ok in zip(keys, known, strict=True) if ok])
    lower = frame.loc[known, f"lower_{tag}"].to_numpy()
    upper = frame.loc[known, f"upper_{tag}"].to_numpy()
    return round(coverage(y, lower, upper) * y.size), int(y.size)


@dataclass(slots=True)
class _Replay:
    """Running tally of the post-shift replay."""

    daily: list[dict[str, float | int | str]] = field(default_factory=list)
    proposals: list[str] = field(default_factory=list)
    static_hits: int = 0
    static_n: int = 0
    aci_hits: int = 0
    aci_n: int = 0
    quarantined: int = 0
    first_alarm_day: int | None = None

    @property
    def static_coverage(self) -> float:
        return round(self.static_hits / max(self.static_n, 1), 4)

    @property
    def aci_coverage(self) -> float:
        return round(self.aci_hits / max(self.aci_n, 1), 4)


def _records_for_day(panel: Panel, day: int, offset: int) -> list[ActualRecord]:
    """One actuals record per series, with an adversarial spike injected on one known day."""
    records: list[ActualRecord] = []
    target = panel.dates[day].date()
    for index, series_id in enumerate(panel.series_ids):
        sales = int(panel.sales[index, day])
        if offset == _SPIKE_DAY and index == 0:
            sales = max(1, sales) * _SPIKE_FACTOR  # spike from a compromised upstream feed
        records.append(
            ActualRecord(
                series_id=series_id,
                date=target,
                sales=sales,
                stockout=int(panel.stockout[index, day]),
            )
        )
    return records


def _horizon1(batch_frame: pd.DataFrame) -> pd.DataFrame:
    return batch_frame[batch_frame["horizon"] == 1]


def _replay_post_shift(
    runtime: ForecastRuntime,
    panel: Panel,
    demand: DemandLookup,
    clock_day: dict[str, pd.Timestamp],
) -> _Replay:
    """Ingest every post-shift day, scoring the previous day's one-step intervals as we go."""
    state = _Replay()
    batch = runtime.run_forecast()
    pending = {"aci": _horizon1(batch.frame), "static": _horizon1(batch.static_frame)}

    for offset in range(_POST_DAYS):
        day = _SHIFT_DAY + offset
        current = panel.dates[day]
        clock_day["value"] = current
        summary = runtime.ingest(_records_for_day(panel, day, offset), actor="demo-ingestor")
        state.quarantined += summary.quarantined
        if summary.alarms and state.first_alarm_day is None:
            state.first_alarm_day = offset
        state.proposals.extend(summary.proposals_opened)

        hits, n = _coverage_of(pending["static"], demand, _DEMO_TAG)
        state.static_hits, state.static_n = state.static_hits + hits, state.static_n + n
        hits, n = _coverage_of(pending["aci"], demand, _DEMO_TAG)
        state.aci_hits, state.aci_n = state.aci_hits + hits, state.aci_n + n
        state.daily.append(
            {
                "day": str(current.date()),
                "static_cov_80_cum": state.static_coverage,
                "aci_cov_80_cum": state.aci_coverage,
                "alarms": ",".join(summary.alarms),
            }
        )
        batch = runtime.run_forecast()
        pending = {"aci": _horizon1(batch.frame), "static": _horizon1(batch.static_frame)}
    return state


def _pre_shift_backtest(pre: pd.DataFrame, training: TrainingConfig) -> BacktestResult:
    return run_backtest(
        Panel.from_frame(pre, validate=False),
        BacktestConfig(n_origins=2, origin_stride=28, training=training, ets_series_limit=6),
    )


def run_demo(workdir: Path) -> dict[str, JsonValue]:
    """Run the whole story end to end and return a JSON-serializable summary."""
    started = time.perf_counter()
    workdir.mkdir(parents=True, exist_ok=True)
    generator = _demo_generator()
    full = generate_panel(generator)
    full_panel = Panel.from_frame(full, validate=False)
    shift_date = full_panel.dates[_SHIFT_DAY]
    pre = full[full["date"] < shift_date].copy()

    training = TrainingConfig(gbm=_demo_gbm(), train_days=300, calibration_days=42)
    backtest = _pre_shift_backtest(pre, training)

    settings = Settings(
        _env_file=None,
        data_dir=workdir,
        calibration_window_days=42,
        coverage_window_days=14,
        coverage_min_observations=40,
    )
    clock_day = {"value": shift_date - pd.Timedelta(days=1)}
    runtime = ForecastRuntime(
        settings,
        RuntimePaths(workdir / "runtime"),
        clock=lambda: datetime.combine(clock_day["value"].date(), datetime.min.time(), tzinfo=UTC),
    )
    install = runtime.train_and_install(
        pre,
        seed=generator.seed,
        generator_params={
            "n_series": generator.n_series,
            "n_days": generator.n_days,
            "shift_day": _SHIFT_DAY,
        },
        gbm=_demo_gbm(),
        run_name="demo",
    )

    demand: DemandLookup = {
        (str(sid), pd.Timestamp(day)): float(value)
        for sid, day, value in zip(full["series_id"], full["date"], full["demand"], strict=True)
    }
    replay = _replay_post_shift(runtime, full_panel, demand, clock_day)

    calibration = runtime.calibration_summary()
    audit = runtime.audit_status()
    example = runtime.get_forecast(full_panel.series_ids[1], 7)
    pending = runtime.proposals("pending")
    runtime.close()

    result: dict[str, Any] = {
        "series": generator.n_series,
        "days": generator.n_days,
        "shift_day": str(shift_date.date()),
        "backtest_coverage_80_mondrian_pre_shift": round(
            backtest.metrics["coverage_80_mondrian"], 4
        ),
        "backtest_coverage_95_mondrian_pre_shift": round(
            backtest.metrics["coverage_95_mondrian"], 4
        ),
        "backtest_mase_gbm": round(backtest.metrics["mase_gbm"], 4),
        "backtest_mase_snaive": round(backtest.metrics["mase_snaive"], 4),
        "model_version": install["model_version"],
        "calibration_version": install["calibration_version"],
        "mlflow_run_id": install["mlflow_run_id"],
        "post_shift_days": _POST_DAYS,
        "static_coverage_80_post_shift": replay.static_coverage,
        "aci_coverage_80_post_shift": replay.aci_coverage,
        "aci_alpha_effective": calibration["alpha_effective"],
        "first_alarm_day_after_shift": replay.first_alarm_day,
        "proposals_opened": replay.proposals,
        "pending_proposals": [item.id for item in pending],
        "quarantined_actuals": replay.quarantined,
        "audit_chain_ok": audit["ok"],
        "audit_entries": audit["entries"],
        "example_forecast": example,
        "daily_tail": replay.daily[-5:],
        "seconds": round(time.perf_counter() - started, 1),
    }
    return result
