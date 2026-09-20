"""End-to-end runtime: install, forecast, ingest, alarm, approve, promote.

This is where the pieces have to agree: the calibration artifact must match the registered model,
ingested actuals must be screened before they touch the ACI state, a coverage breach must open a
proposal that only an operator can apply, and every step must land in the hash chain.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import JsonValue

from conformal_demand_forecaster.conformal.artifacts import SCORES_NPZ, CalibrationIntegrityError
from conformal_demand_forecaster.core.config import Settings
from conformal_demand_forecaster.data.generator import GeneratorConfig, RegimeShift, generate_panel
from conformal_demand_forecaster.data.panel import Panel
from conformal_demand_forecaster.ml.contracts import DataContractError
from conformal_demand_forecaster.models.gbm import GbmConfig
from conformal_demand_forecaster.monitoring.audit import verify_chain
from conformal_demand_forecaster.monitoring.recalibration import WorkflowError
from conformal_demand_forecaster.serving.runtime import (
    ActualRecord,
    ForecastRuntime,
    ForecastRuntimeError,
    RuntimePaths,
    load_install,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

SHIFT_DAY = 220
POST_DAYS = 30
SEED = 4321
GBM = GbmConfig(
    horizon_buckets=((1, 7),), num_iterations=25, num_leaves=7, min_data_in_leaf=10, num_threads=2
)


@pytest.fixture(scope="module")
def full_frame() -> pd.DataFrame:
    return generate_panel(
        GeneratorConfig(
            n_series=8,
            n_days=SHIFT_DAY + POST_DAYS,
            seed=SEED,
            shifts=(RegimeShift(day=SHIFT_DAY, fraction=1.0, multiplier=3.0),),
        )
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path,
        forecast_horizon=7,
        calibration_window_days=30,
        coverage_window_days=10,
        coverage_min_observations=20,
    )


def make_runtime(settings: Settings, tmp_path: Path, today: date) -> ForecastRuntime:
    return ForecastRuntime(
        settings,
        RuntimePaths(tmp_path / "runtime"),
        clock=lambda: datetime.combine(today, datetime.min.time(), tzinfo=UTC),
    )


@pytest.fixture
def installed(
    full_frame: pd.DataFrame, settings: Settings, tmp_path: Path
) -> Iterator[tuple[ForecastRuntime, dict[str, JsonValue], pd.DataFrame, Panel]]:
    panel = Panel.from_frame(full_frame, validate=False)
    shift_date = panel.dates[SHIFT_DAY]
    pre = full_frame[full_frame["date"] < shift_date].copy()
    runtime = make_runtime(settings, tmp_path, (shift_date - pd.Timedelta(days=1)).date())
    summary = runtime.train_and_install(
        pre, seed=SEED, generator_params={"n_series": 8}, gbm=GBM, run_name="test"
    )
    try:
        yield runtime, summary, full_frame, panel
    finally:
        runtime.close()


def records_for(panel: Panel, day: int, *, spike_index: int | None = None) -> list[ActualRecord]:
    target = panel.dates[day].date()
    out = []
    for index, series_id in enumerate(panel.series_ids):
        sales = int(panel.sales[index, day])
        if index == spike_index:
            sales = max(1, sales) * 80
        out.append(
            ActualRecord(
                series_id=series_id,
                date=target,
                sales=sales,
                stockout=int(panel.stockout[index, day]),
            )
        )
    return out


# --- install --------------------------------------------------------------------------------


def test_install_registers_a_model_a_calibration_and_a_first_batch(
    installed: tuple[ForecastRuntime, dict[str, JsonValue], pd.DataFrame, Panel],
) -> None:
    runtime, summary, _, _ = installed
    assert len(str(summary["model_version"])) == 64
    assert len(str(summary["calibration_version"])) == 64
    assert len(str(summary["data_sha256"])) == 64
    assert int(str(summary["training_rows"])) > 0
    assert int(str(summary["calibration_rows"])) > 0
    assert int(str(summary["forecast_rows"])) == 8 * 7
    assert runtime.installed is True
    assert runtime.calibration_loaded() is True
    assert runtime.store.counts()["forecasts"] == 56


def test_the_run_manifest_records_the_seed_data_hash_and_lockfile(
    installed: tuple[ForecastRuntime, dict[str, JsonValue], pd.DataFrame, Panel],
) -> None:
    runtime, summary, _, _ = installed
    manifest = json.loads(runtime.paths.manifest.read_text(encoding="utf-8"))
    assert manifest["seed"] == SEED
    assert manifest["params"]["uv_lock_sha256"]
    assert manifest["params"]["generator"] == {"n_series": 8}
    assert any(item["sha256"] == summary["data_sha256"] for item in manifest["datasets"])
    packages = manifest["packages"]
    assert {"numpy", "pandas", "lightgbm", "scipy"} <= set(packages)


def test_the_install_record_is_readable_and_names_both_versions(
    installed: tuple[ForecastRuntime, dict[str, JsonValue], pd.DataFrame, Panel],
) -> None:
    runtime, summary, _, _ = installed
    install = load_install(runtime.paths)
    assert install["model_version"] == summary["model_version"]
    assert install["calibration_version"] == summary["calibration_version"]
    assert install["seed"] == SEED


def test_install_is_audited(
    installed: tuple[ForecastRuntime, dict[str, JsonValue], pd.DataFrame, Panel],
) -> None:
    runtime, summary, _, _ = installed
    entries = runtime.audit.entries()
    assert [entry.action for entry in entries] == ["model_installed"]
    assert entries[0].data["model_version"] == summary["model_version"]
    assert runtime.audit_status()["ok"] is True


# --- reload ---------------------------------------------------------------------------------


def test_a_reloaded_runtime_serves_the_same_forecasts(
    installed: tuple[ForecastRuntime, dict[str, JsonValue], pd.DataFrame, Panel],
    settings: Settings,
    tmp_path: Path,
) -> None:
    runtime, summary, _, panel = installed
    before = runtime.get_forecast(panel.series_ids[0], 3)
    reloaded = make_runtime(settings, tmp_path, panel.dates[SHIFT_DAY].date()).load()
    try:
        assert reloaded.model_version == summary["model_version"]
        assert reloaded.artifact is not None
        assert reloaded.artifact.version == summary["calibration_version"]
        after = reloaded.get_forecast(panel.series_ids[0], 3)
    finally:
        reloaded.close()
    assert before == after


def test_a_tampered_calibration_file_refuses_to_load(
    installed: tuple[ForecastRuntime, dict[str, JsonValue], pd.DataFrame, Panel],
    settings: Settings,
    tmp_path: Path,
) -> None:
    runtime, _, _, panel = installed
    scores_path = runtime.paths.calibration / SCORES_NPZ
    poisoned = {name: values * 5.0 for name, values in np.load(scores_path).items()}
    np.savez(scores_path, **poisoned)
    with pytest.raises(CalibrationIntegrityError, match=r"scores\.npz"):
        make_runtime(settings, tmp_path, panel.dates[SHIFT_DAY].date()).load()


def test_loading_without_an_install_is_an_explicit_error(
    settings: Settings, tmp_path: Path
) -> None:
    runtime = make_runtime(settings, tmp_path / "empty", date(2024, 1, 1))
    with pytest.raises(ForecastRuntimeError, match="nothing installed"):
        runtime.load()


def test_an_unloaded_runtime_refuses_to_forecast(settings: Settings, tmp_path: Path) -> None:
    runtime = make_runtime(settings, tmp_path / "empty", date(2024, 1, 1))
    with pytest.raises(ForecastRuntimeError, match="not loaded"):
        runtime.run_forecast()


# --- forecasts ------------------------------------------------------------------------------


def test_a_forecast_carries_intervals_versions_and_an_order(
    installed: tuple[ForecastRuntime, dict[str, JsonValue], pd.DataFrame, Panel],
) -> None:
    runtime, summary, _, panel = installed
    result = runtime.get_forecast(panel.series_ids[0], 1)
    assert result is not None
    assert result["calibration_version"] == summary["calibration_version"]
    assert result["model_version"] == summary["model_version"]
    for tag in ("80", "95"):
        band = result["intervals"][tag]
        assert band["upper"] >= band["lower"] >= 0.0
        assert 0.0 <= band["alpha_effective"] <= 1.0
    assert result["intervals"]["95"]["upper"] >= result["intervals"]["80"]["upper"]
    order = result["order"]
    assert order["units"] >= 0
    assert order["fractile"] == pytest.approx(4.0 / 5.0)


def test_order_quantities_react_to_the_cost_ratio(
    installed: tuple[ForecastRuntime, dict[str, JsonValue], pd.DataFrame, Panel],
) -> None:
    runtime, _, _, panel = installed
    cautious = runtime.get_forecast(panel.series_ids[0], 1, holding_cost=1.0, stockout_cost=50.0)
    thrifty = runtime.get_forecast(panel.series_ids[0], 1, holding_cost=50.0, stockout_cost=1.0)
    assert cautious is not None
    assert thrifty is not None
    assert cautious["order"]["quantity"] >= thrifty["order"]["quantity"]


def test_an_unknown_series_has_no_forecast(
    installed: tuple[ForecastRuntime, dict[str, JsonValue], pd.DataFrame, Panel],
) -> None:
    runtime, _, _, _ = installed
    assert runtime.get_forecast("S9999_ST99", 1) is None


# --- ingest ---------------------------------------------------------------------------------


def test_ingest_screens_a_spike_and_scores_the_rest(
    installed: tuple[ForecastRuntime, dict[str, JsonValue], pd.DataFrame, Panel],
) -> None:
    runtime, _, _, panel = installed
    summary = runtime.ingest(records_for(panel, SHIFT_DAY, spike_index=0), actor="key:ingestor")
    assert summary.received == 8
    assert summary.accepted + summary.quarantined == 8
    assert summary.unknown_series == 0
    quarantined = runtime.store.actuals(status="quarantined")
    # The injected 80x spike must be among the quarantined rows (a 3x regime shift on the same
    # day can legitimately trip the guard for a low-variance series too).
    assert panel.series_ids[0] in set(quarantined["series_id"])
    assert (
        quarantined.loc[quarantined["series_id"] == panel.series_ids[0], "z_score"].iloc[0] > 10.0
    )
    assert runtime.store.actuals(status="accepted").shape[0] == summary.accepted


def test_quarantined_values_never_enter_the_panel(
    installed: tuple[ForecastRuntime, dict[str, JsonValue], pd.DataFrame, Panel],
) -> None:
    runtime, _, _, panel = installed
    before = runtime.store.counts()["sales"]
    summary = runtime.ingest(records_for(panel, SHIFT_DAY, spike_index=0), actor="key:ingestor")
    after = runtime.store.load_panel()
    assert summary.quarantined >= 1
    assert runtime.store.counts()["sales"] == before + summary.accepted
    assert after["sales"].max() < 5_000  # the 80x spike never reached the panel


def test_unknown_series_are_counted_and_ignored(
    installed: tuple[ForecastRuntime, dict[str, JsonValue], pd.DataFrame, Panel],
) -> None:
    runtime, _, _, panel = installed
    records = [
        *records_for(panel, SHIFT_DAY)[:2],
        ActualRecord(series_id="S9999_ST99", date=panel.dates[SHIFT_DAY].date(), sales=5),
    ]
    summary = runtime.ingest(records, actor="key:ingestor")
    assert summary.unknown_series == 1
    assert summary.accepted == 2


def test_an_empty_ingest_is_a_contract_violation(
    installed: tuple[ForecastRuntime, dict[str, JsonValue], pd.DataFrame, Panel],
) -> None:
    runtime, _, _, _ = installed
    with pytest.raises(DataContractError):
        runtime.ingest([], actor="key:ingestor")


def test_duplicate_rows_in_one_batch_are_a_contract_violation(
    installed: tuple[ForecastRuntime, dict[str, JsonValue], pd.DataFrame, Panel],
) -> None:
    runtime, _, _, panel = installed
    record = records_for(panel, SHIFT_DAY)[0]
    with pytest.raises(DataContractError):
        runtime.ingest([record, record], actor="key:ingestor")


def test_ingest_moves_the_aci_alpha_after_a_shift(
    installed: tuple[ForecastRuntime, dict[str, JsonValue], pd.DataFrame, Panel],
) -> None:
    runtime, _, _, panel = installed
    assert runtime.artifact is not None
    before = dict(runtime.artifact.aci.effective_alphas())
    for offset in range(10):
        runtime.ingest(records_for(panel, SHIFT_DAY + offset), actor="key:ingestor")
        runtime.run_forecast()
    after = runtime.artifact.aci.effective_alphas()
    assert any(after[key] != before[key] for key in before)
    assert all(0.0 <= value <= 1.0 for value in after.values())


# --- monitoring and recalibration ------------------------------------------------------------


def test_a_regime_shift_eventually_opens_an_approval_gated_proposal(
    installed: tuple[ForecastRuntime, dict[str, JsonValue], pd.DataFrame, Panel],
) -> None:
    runtime, _, _, panel = installed
    alarms: list[str] = []
    proposals: list[str] = []
    for offset in range(POST_DAYS):
        summary = runtime.ingest(records_for(panel, SHIFT_DAY + offset), actor="key:ingestor")
        alarms.extend(summary.alarms)
        proposals.extend(summary.proposals_opened)
        runtime.run_forecast()
    assert alarms, "a 3x level shift must break the coverage SLO"
    assert proposals, "a breach must open a recalibration proposal"

    pending = runtime.proposals("pending")
    assert pending
    proposal = pending[0]
    # Nobody has approved it, so it cannot be applied.
    with pytest.raises(WorkflowError, match="only an approved proposal"):
        runtime.workflow.mark_applied(
            proposal.id, "key:attacker", from_version="a" * 64, to_version="b" * 64
        )

    before_version = runtime.artifact.version if runtime.artifact else ""
    applied = runtime.approve_and_apply(proposal.id, "key:operator", "regime shift")
    assert applied.status == "applied"
    assert runtime.artifact is not None
    assert runtime.artifact.version != before_version
    assert applied.applied_version == runtime.artifact.version
    # Other (segment, level) keys may still have their own pending proposals; this one is done.
    assert proposal.id not in {item.id for item in runtime.proposals("pending")}
    assert runtime.workflow.get(proposal.id).status == "applied"

    actions = [entry.action for entry in runtime.audit.entries()]
    assert "recalibration_proposed" in actions
    assert "recalibration_approved" in actions
    assert "recalibration_applied" in actions
    assert verify_chain(runtime.paths.audit).ok is True


def test_rejecting_a_proposal_keeps_the_live_calibration(
    installed: tuple[ForecastRuntime, dict[str, JsonValue], pd.DataFrame, Panel],
) -> None:
    runtime, summary, _, panel = installed
    opened: list[str] = []
    for offset in range(POST_DAYS):
        result = runtime.ingest(records_for(panel, SHIFT_DAY + offset), actor="key:ingestor")
        opened.extend(result.proposals_opened)
        runtime.run_forecast()
    assert opened
    rejected = runtime.reject(opened[0], "key:operator", "known promotion")
    assert rejected.status == "rejected"
    assert runtime.artifact is not None
    assert runtime.artifact.version == summary["calibration_version"]


def test_the_calibration_summary_reports_state_operators_need(
    installed: tuple[ForecastRuntime, dict[str, JsonValue], pd.DataFrame, Panel],
) -> None:
    runtime, summary, _, panel = installed
    runtime.ingest(records_for(panel, SHIFT_DAY), actor="key:ingestor")
    report = runtime.calibration_summary()
    assert report["calibration_version"] == summary["calibration_version"]
    assert report["model_version"] == summary["model_version"]
    assert report["alphas"] == [0.2, 0.05]
    assert report["group_counts"]
    assert report["alpha_effective"]
    assert isinstance(report["pending_proposals"], list)


def test_readiness_tracks_forecast_freshness(
    installed: tuple[ForecastRuntime, dict[str, JsonValue], pd.DataFrame, Panel],
    settings: Settings,
    tmp_path: Path,
) -> None:
    runtime, _, _, panel = installed
    assert runtime.forecasts_fresh() is True
    stale = make_runtime(settings, tmp_path, date(2030, 1, 1)).load()
    try:
        assert stale.forecasts_fresh() is False
    finally:
        stale.close()
    assert panel.n_days == SHIFT_DAY + POST_DAYS
