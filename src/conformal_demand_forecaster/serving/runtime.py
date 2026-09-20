"""The forecasting runtime: artifacts, batch forecasts, actuals ingest and recalibration.

Install (``train_and_install``): write the panel to DuckDB, train and calibrate at the last
observed day, register the LightGBM text models in the content-addressed registry, build the
calibration artifact, write the run manifest, log the run to MLflow and issue the first batch.

Serve (``load``): verify every artifact against its recorded digest, then

- ``run_forecast`` writes conformalized forecasts (ACI-adjusted alphas) for every series;
- ``ingest`` validates actuals against the contract, screens each value with the robust z-score
  guard, scores accepted values against the forecasts issued for them, updates the ACI alpha
  states and the coverage monitor, opens recalibration proposals on alarms and appends the
  accepted days to the panel;
- ``approve_and_apply`` promotes a new calibration version from the live scores, audited.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from conformal_demand_forecaster.conformal.artifacts import CalibrationArtifact
from conformal_demand_forecaster.conformal.cqr import ScoreBuffer, cqr_scores
from conformal_demand_forecaster.core.canonical import canonical_json
from conformal_demand_forecaster.core.config import Settings
from conformal_demand_forecaster.core.logging import get_logger
from conformal_demand_forecaster.data.panel import Panel
from conformal_demand_forecaster.decisions.newsvendor import order_quantity
from conformal_demand_forecaster.features.builder import FEATURE_NAMES, FeatureBuilder
from conformal_demand_forecaster.ml.contracts import (
    MAX_UNITS,
    SEGMENTS,
    SERIES_ID_PATTERN,
    DataContractError,
    validate_actuals,
)
from conformal_demand_forecaster.ml.drift import psi
from conformal_demand_forecaster.ml.manifest import RunManifest, create_manifest, sha256_file
from conformal_demand_forecaster.ml.registry import ModelRegistry
from conformal_demand_forecaster.ml.serialization import load_json, save_json
from conformal_demand_forecaster.models.gbm import GbmConfig, QuantileGbm
from conformal_demand_forecaster.monitoring.audit import AuditChain, verify_chain
from conformal_demand_forecaster.monitoring.coverage import CoverageMonitor, MonitorConfig
from conformal_demand_forecaster.monitoring.metrics import ForecastMetrics
from conformal_demand_forecaster.monitoring.recalibration import (
    Proposal,
    ProposalStatus,
    RecalibrationWorkflow,
)
from conformal_demand_forecaster.serving.guard import RobustZScoreGuard
from conformal_demand_forecaster.serving.store import ForecastStore
from conformal_demand_forecaster.tracking import log_training_run
from conformal_demand_forecaster.training import (
    TrainingConfig,
    forecast_frame,
    forecast_rows,
    train_at,
)

__all__ = [
    "ActualRecord",
    "ForecastBatch",
    "ForecastRuntime",
    "ForecastRuntimeError",
    "IngestSummary",
    "RuntimePaths",
]

MODEL_NAME = "demand-gbm"
CALIBRATION_NAME = "calibration"
PRODUCTION = "production"
TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "pandera",
    "pydantic",
    "lightgbm",
    "scipy",
    "statsmodels",
    "duckdb",
    "mlflow-skinny",
)
_log = get_logger("conformal_demand_forecaster.runtime")
FloatArray = npt.NDArray[np.float64]


class ForecastRuntimeError(RuntimeError):
    """The runtime is not in a state to serve the request (nothing installed, unknown series)."""


class ActualRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    series_id: str = Field(pattern=SERIES_ID_PATTERN, max_length=32)
    date: date
    sales: int = Field(ge=0, le=MAX_UNITS)
    stockout: int = Field(default=0, ge=0, le=1)


class IngestSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    received: int
    accepted: int
    quarantined: int
    duplicates: int
    unknown_series: int
    scored: int
    alarms: list[str]
    proposals_opened: list[str]
    calibration_version: str


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    root: Path

    @property
    def store(self) -> Path:
        return self.root / "store.duckdb"

    @property
    def registry(self) -> Path:
        return self.root / "registry"

    @property
    def calibration(self) -> Path:
        return self.root / "calibration"

    @property
    def runtime(self) -> Path:
        return self.root / "runtime"

    @property
    def audit(self) -> Path:
        return self.root / "audit.jsonl"

    @property
    def proposals(self) -> Path:
        return self.root / "proposals.json"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def install(self) -> Path:
        return self.root / "install.json"

    @property
    def panel_csv(self) -> Path:
        return self.root / "panel.csv"

    @property
    def tracking(self) -> Path:
        return self.root / "mlruns"


@dataclass(slots=True)
class ForecastBatch:
    origin: date
    frame: pd.DataFrame
    static_frame: pd.DataFrame
    rows: int


def _tag(alpha: float) -> str:
    return f"{round((1 - alpha) * 100):d}"


def _training_from_settings(settings: Settings, gbm: GbmConfig | None = None) -> TrainingConfig:
    model = gbm or GbmConfig(horizon_buckets=((1, 7), (8, 14), (15, settings.forecast_horizon)))
    return TrainingConfig(
        gbm=model,
        alphas=settings.alphas,
        calibration_days=settings.calibration_window_days,
        aci_gamma=settings.aci_gamma,
    )


class ForecastRuntime:
    def __init__(
        self,
        settings: Settings,
        paths: RuntimePaths,
        *,
        metrics: ForecastMetrics | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.paths = paths
        self.metrics = metrics
        self.clock = clock or (lambda: datetime.now(UTC))
        # The store, registry and audit log all open files under the root, so it must exist
        # before any of them is constructed (a fresh data directory is the normal first run).
        paths.root.mkdir(parents=True, exist_ok=True)
        self.store = ForecastStore(paths.store)
        self.registry = ModelRegistry(paths.registry)
        self.audit = AuditChain(paths.audit)
        self.workflow = RecalibrationWorkflow(paths.proposals, self.audit)
        self.guard = RobustZScoreGuard(threshold=settings.guard_z_threshold)
        self.monitor = CoverageMonitor(
            MonitorConfig(
                window_days=settings.coverage_window_days,
                tolerance=settings.coverage_tolerance,
                significance=settings.coverage_significance,
                min_observations=settings.coverage_min_observations,
            )
        )
        self.training: TrainingConfig | None = None
        self.model: QuantileGbm | None = None
        self.artifact: CalibrationArtifact | None = None
        self.model_version: str | None = None
        self.live_scores = ScoreBuffer(settings.calibration_window_days)
        self._panel: Panel | None = None

    # --- lifecycle -------------------------------------------------------------------------

    def bind_metrics(self, metrics: ForecastMetrics) -> None:
        self.metrics = metrics

    def close(self) -> None:
        self.store.close()

    @property
    def installed(self) -> bool:
        return self.paths.install.exists()

    @property
    def panel(self) -> Panel:
        if self._panel is None:
            # The stored rows were contract-validated on the way in (install and ingest), and a
            # quarantined day legitimately leaves one series without an observation, so the
            # panel is rebuilt tolerantly rather than re-validated for strict continuity.
            self._panel = Panel.from_frame(
                self.store.load_panel(),
                validate=False,
                fill_missing_days=True,
                max_days=self.settings.panel_max_days,
            )
        return self._panel

    def _require(self) -> tuple[QuantileGbm, CalibrationArtifact, TrainingConfig]:
        if self.model is None or self.artifact is None or self.training is None:
            raise ForecastRuntimeError("runtime is not loaded; run train first")
        return self.model, self.artifact, self.training

    def calibration_loaded(self) -> bool:
        return self.artifact is not None and self.model is not None

    def forecasts_fresh(self) -> bool:
        """Whether the newest origin is recent -- and not ahead of the service's own clock.

        A negative age means the store holds forecasts issued from a day that has not happened,
        which is a sign of a bad origin rather than of freshness, so readiness fails on it.
        """
        latest = self.store.latest_origin()
        if latest is None:
            return False
        age = (self.clock().date() - latest).days
        return 0 <= age <= self.settings.freshness_max_days

    # --- install -----------------------------------------------------------------------------

    def train_and_install(
        self,
        panel_frame: pd.DataFrame,
        *,
        seed: int,
        generator_params: dict[str, JsonValue],
        gbm: GbmConfig | None = None,
        run_name: str = "install",
    ) -> dict[str, JsonValue]:
        training = _training_from_settings(self.settings, gbm)
        self.paths.root.mkdir(parents=True, exist_ok=True)
        panel_frame.to_csv(self.paths.panel_csv, index=False, lineterminator="\n")
        data_sha256 = sha256_file(self.paths.panel_csv)
        # An install replaces the served panel; rows of an earlier one would otherwise decide the
        # origin and leave forecasts and actuals behind that belong to a panel nobody serves.
        self.store.replace_panel(panel_frame)
        self._panel = None
        panel = self.panel
        origin = panel.last_observed_day
        bundle = train_at(panel, origin, training)

        manifest = create_manifest(
            seed=seed,
            datasets=[self.paths.panel_csv],
            params={
                "generator": generator_params,
                "training": _training_params(training),
                "uv_lock_sha256": _lock_digest(),
                "origin": str(panel.dates[origin].date()),
            },
            metrics={
                "training_rows": float(bundle.training_rows),
                "calibration_rows": float(bundle.calibration_rows),
            },
            packages=TRACKED_PACKAGES,
        )
        files = {f"{key}.txt": text.encode("utf-8") for key, text in bundle.model.to_text().items()}
        files["config.json"] = canonical_json(_training_params(training)).encode("utf-8")
        version = self.registry.register(MODEL_NAME, files, manifest)
        self.registry.set_alias(MODEL_NAME, PRODUCTION, version.digest)

        artifact = CalibrationArtifact.build(
            scores=bundle.scores,
            segment_map=dict(zip(panel.series_ids, panel.segments, strict=True)),
            model_version=version.digest,
            data_sha256=data_sha256,
            alphas=training.alphas,
            horizon_buckets=training.gbm.horizon_buckets,
            quantile_pairs=bundle.calibrator.pairs,
            reference=bundle.reference,
            calibrated_through_day=str(panel.dates[origin].date()),
            gamma=training.aci_gamma,
        )
        self._install_calibration(artifact, manifest)
        manifest.write(self.paths.manifest)
        save_json(
            self.paths.install,
            {
                "model_version": version.digest,
                "calibration_version": artifact.version,
                "training": _training_params(training),
                "data_sha256": data_sha256,
                "seed": seed,
            },
        )
        run_id = log_training_run(
            self.paths.tracking,
            run_name=run_name,
            params={
                "seed": seed,
                "data_sha256": data_sha256,
                "model_version": version.digest,
                "calibration_version": artifact.version,
                "uv_lock_sha256": manifest.params["uv_lock_sha256"],
                **{f"gbm.{k}": v for k, v in _training_params(training)["gbm"].items()},
                "alphas": list(training.alphas),
                "calibration_days": training.calibration_days,
                "aci_gamma": training.aci_gamma,
            },
            metrics=manifest.metrics,
            artifacts=[self.paths.manifest],
        )
        self.audit.append(
            "trainer",
            "model_installed",
            {
                "model_version": version.digest,
                "calibration_version": artifact.version,
                "run_id": run_id,
            },
        )
        self.training = training
        self.model = bundle.model
        self.model_version = version.digest
        self.artifact = artifact
        self.live_scores = ScoreBuffer(training.calibration_days)
        batch = self.run_forecast()
        return {
            "model_version": version.digest,
            "calibration_version": artifact.version,
            "data_sha256": data_sha256,
            "mlflow_run_id": run_id,
            "training_rows": bundle.training_rows,
            "calibration_rows": bundle.calibration_rows,
            "forecast_rows": batch.rows,
            "origin": str(batch.origin),
        }

    def _install_calibration(self, artifact: CalibrationArtifact, manifest: RunManifest) -> None:
        artifact.save(self.paths.calibration)
        files = {
            name: (self.paths.calibration / name).read_bytes()
            for name in ("calibration.json", "scores.npz", "reference.npz")
            if (self.paths.calibration / name).exists()
        }
        version = self.registry.register(CALIBRATION_NAME, files, manifest)
        self.registry.set_alias(CALIBRATION_NAME, PRODUCTION, version.digest)

    # --- load --------------------------------------------------------------------------------

    def load(self) -> ForecastRuntime:
        """Load the production model and calibration, verifying every digest."""
        if not self.installed:
            raise ForecastRuntimeError("nothing installed under the data directory")
        install = load_json(self.paths.install)
        training = _training_from_params(install["training"])
        model_version = str(install["model_version"])
        files = self.registry.load_files(self.registry.get(MODEL_NAME, model_version))
        texts = {
            name[:-4]: data.decode("utf-8") for name, data in files.items() if name.endswith(".txt")
        }
        self.model = QuantileGbm.from_text(texts, training.gbm, FEATURE_NAMES)
        self.model_version = model_version
        self.artifact = CalibrationArtifact.load(self.paths.calibration, gamma=training.aci_gamma)
        if self.artifact.record.model_version != model_version:
            raise ForecastRuntimeError("calibration artifact belongs to a different model version")
        self.training = training
        self._load_runtime_state()
        return self

    def _load_runtime_state(self) -> None:
        monitor_path = self.paths.runtime / "monitor.json"
        if monitor_path.exists():
            self.monitor = CoverageMonitor.from_dict(load_json(monitor_path), self.monitor.config)
        live_path = self.paths.runtime / "live_scores.json"
        if live_path.exists():
            payload = load_json(live_path)
            self.live_scores = ScoreBuffer(int(payload["window_days"]))
            for entry in payload["entries"]:
                self.live_scores.append(
                    (str(entry["group"]), float(entry["alpha"])), entry["days"], entry["scores"]
                )

    def _save_runtime_state(self) -> None:
        _, artifact, _ = self._require()
        self.paths.runtime.mkdir(parents=True, exist_ok=True)
        artifact.save_alpha_state(self.paths.calibration)
        save_json(self.paths.runtime / "monitor.json", self.monitor.to_dict())
        entries = [
            {
                "group": group,
                "alpha": alpha,
                "days": self.live_scores.days((group, alpha)).tolist(),
                "scores": self.live_scores.scores((group, alpha)).tolist(),
            }
            for group, alpha in self.live_scores.keys
        ]
        save_json(
            self.paths.runtime / "live_scores.json",
            {"window_days": self.live_scores.window_days, "entries": entries},
        )

    # --- forecasts ---------------------------------------------------------------------------

    def run_forecast(self, origin: date | None = None) -> ForecastBatch:
        model, artifact, training = self._require()
        panel = self.panel
        origin_index = (
            panel.last_observed_day if origin is None else panel.day_index(pd.Timestamp(origin))
        )
        horizon = training.horizon
        extended = panel.extend(max(0, origin_index + horizon + 1 - panel.n_days))
        rows = FeatureBuilder(extended).rows([origin_index], np.arange(1, horizon + 1))
        calibrator = training.calibrator()
        prediction, intervals = forecast_rows(
            model,
            calibrator,
            artifact.scores,
            extended,
            rows,
            alpha_effective=artifact.aci.effective_alphas(),
        )
        _, static = forecast_rows(model, calibrator, artifact.scores, extended, rows)
        frame = forecast_frame(calibrator, extended, rows, prediction, intervals)
        static_frame = forecast_frame(calibrator, extended, rows, prediction, static)
        now = self.clock().replace(tzinfo=None)
        frame["calibration_version"] = artifact.version
        frame["model_version"] = self.model_version
        frame["created_at"] = now
        self.store.write_forecasts(frame)
        origin_date = panel.dates[origin_index].date()
        if self.metrics is not None:
            for alpha in training.alphas:
                tag = _tag(alpha)
                for segment in SEGMENTS:
                    mask = frame["segment"] == segment
                    if mask.any():
                        width = float(
                            (
                                frame.loc[mask, f"upper_{tag}"] - frame.loc[mask, f"lower_{tag}"]
                            ).mean()
                        )
                        self.metrics.interval_width.labels(tag, segment).set(width)
                saturated = int(frame[f"saturated_{tag}"].sum())
                if saturated:
                    self.metrics.calibration_saturated_total.labels(tag).inc(saturated)
            self._publish_alpha_metrics()
            self.metrics.forecast_freshness_seconds.set(
                max(0.0, (self.clock().date() - origin_date).days * 86_400.0)
            )
        _log.info("forecast_batch", origin=str(origin_date), rows=len(frame))
        return ForecastBatch(
            origin=origin_date, frame=frame, static_frame=static_frame, rows=len(frame)
        )

    def get_forecast(
        self,
        series_id: str,
        horizon: int,
        *,
        holding_cost: float | None = None,
        stockout_cost: float | None = None,
    ) -> dict[str, Any] | None:
        _, _, training = self._require()
        record = self.store.latest_forecast(series_id, horizon)
        if record is None:
            return None
        holding = holding_cost if holding_cost is not None else self.settings.holding_cost
        stockout = stockout_cost if stockout_cost is not None else self.settings.stockout_cost
        intervals = {
            alpha: (float(record[f"lower_{_tag(alpha)}"]), float(record[f"upper_{_tag(alpha)}"]))
            for alpha in training.alphas
        }
        decision = order_quantity(float(record["q0_500"]), intervals, holding, stockout)
        if decision.clipped and self.metrics is not None:
            self.metrics.newsvendor_fractile_clipped_total.inc()
        return {
            "series_id": series_id,
            "origin_date": str(record["origin_date"]),
            "target_date": str(record["target_date"]),
            "horizon": int(record["horizon"]),
            "segment": str(record["segment"]),
            "point": float(record["point"]),
            "median": float(record["q0_500"]),
            "intervals": {
                _tag(alpha): {
                    "lower": intervals[alpha][0],
                    "upper": intervals[alpha][1],
                    "alpha_effective": float(record[f"alpha_eff_{_tag(alpha)}"]),
                    "saturated": bool(record[f"saturated_{_tag(alpha)}"]),
                }
                for alpha in training.alphas
            },
            "calibration_version": str(record["calibration_version"]),
            "model_version": str(record["model_version"]),
            "order": {
                "holding_cost": holding,
                "stockout_cost": stockout,
                "fractile": decision.fractile,
                # The interpolation saturates at the outermost calibrated quantile, so a cost
                # ratio beyond it is served at a different fractile than the one requested.
                "fractile_served": decision.fractile_served,
                "fractile_clipped": decision.clipped,
                "quantity": decision.order,
                "units": decision.order_units,
            },
        }

    # --- ingest ------------------------------------------------------------------------------

    def ingest(self, records: Sequence[ActualRecord], actor: str) -> IngestSummary:
        _, artifact, training = self._require()
        if not records:
            raise DataContractError([{"column": "records", "check": "non_empty", "count": 1}])
        frame = pd.DataFrame(
            {
                "series_id": [r.series_id for r in records],
                "date": pd.to_datetime([r.date for r in records]),
                "sales": np.array([r.sales for r in records], dtype=np.int64),
            }
        )
        panel = self.panel
        first_day = panel.dates[0].date()
        last_day = panel.dates[panel.last_observed_day].date()
        validate_actuals(
            frame,
            min_date=first_day,
            max_date=last_day + timedelta(days=training.horizon),
        )
        segments = self.store.series_segments()
        known = [r for r in records if r.series_id in segments]
        unknown = len(records) - len(known)

        # A day already in the panel is history. It is never rewritten, never re-scored and never
        # allowed to move the guard's own reference window; the attempt is audited instead.
        recorded = self.store.existing_sales_days([(r.series_id, r.date) for r in known])
        duplicates = [r for r in known if (r.series_id, r.date) in recorded]
        fresh = [r for r in known if (r.series_id, r.date) not in recorded]
        if duplicates:
            self._audit_duplicates(duplicates, actor)

        accepted = self._screen(fresh, actor)
        scored, touched_days = self._score_accepted(accepted, artifact, training, segments)
        alarms, opened = self._raise_alarms(touched_days, training)

        self.store.append_sales(
            [
                {
                    "series_id": r.series_id,
                    "date": r.date,
                    "sales": r.sales,
                    "stockout": r.stockout,
                    "segment": segments[r.series_id],
                }
                for r in accepted
            ]
        )
        self._panel = None
        self._publish_alpha_metrics()
        self._publish_drift(artifact)
        self._save_runtime_state()
        return IngestSummary(
            received=len(records),
            accepted=len(accepted),
            quarantined=len(fresh) - len(accepted),
            duplicates=len(duplicates),
            unknown_series=unknown,
            scored=scored,
            alarms=alarms,
            proposals_opened=opened,
            calibration_version=artifact.version,
        )

    def _screen(self, records: Sequence[ActualRecord], actor: str) -> list[ActualRecord]:
        """Run every record past the outlier guard; store the verdicts and return the accepted."""
        accepted: list[ActualRecord] = []
        stored: list[dict[str, Any]] = []
        for record in records:
            history = self.store.recent_sales(
                record.series_id, self.settings.calibration_window_days
            )
            decision = self.guard.check(float(record.sales), history)
            stored.append(
                {
                    "series_id": record.series_id,
                    "date": record.date,
                    "sales": record.sales,
                    "status": "accepted" if decision.accepted else "quarantined",
                    "z_score": None if math.isnan(decision.z) else round(decision.z, 4),
                    "actor": actor,
                }
            )
            if decision.accepted:
                accepted.append(record)
            elif self.metrics is not None:
                self.metrics.actuals_quarantined_total.labels(decision.reason).inc()
        self.store.write_actuals(stored)
        if self.metrics is not None and accepted:
            self.metrics.actuals_accepted_total.inc(len(accepted))
        return accepted

    def _raise_alarms(
        self, touched_days: set[int], training: TrainingConfig
    ) -> tuple[list[str], list[str]]:
        """Evaluate the SLO on every day this ingest touched; open a proposal per fresh breach."""
        alarms: list[str] = []
        opened: list[str] = []
        keys = [(segment, alpha) for segment in SEGMENTS for alpha in training.alphas]
        for day in sorted(touched_days):
            for check in self.monitor.evaluate(day, keys):
                if check.alarm:
                    alarms.append(f"{check.segment}@{_tag(check.alpha)}")
                    if self.metrics is not None:
                        self.metrics.coverage_alarm_total.labels(
                            _tag(check.alpha), check.segment
                        ).inc()
                    proposal = self.workflow.open(check)
                    if proposal is not None:
                        opened.append(proposal.id)
                        if self.metrics is not None:
                            self.metrics.recalibration_proposals_total.labels("opened").inc()
                if self.metrics is not None and check.n:
                    self.metrics.rolling_coverage.labels(_tag(check.alpha), check.segment).set(
                        check.coverage
                    )
        return sorted(set(alarms)), opened

    def _audit_duplicates(self, duplicates: Sequence[ActualRecord], actor: str) -> None:
        """Record an attempt to re-post days the panel already holds (values are not logged)."""
        if self.metrics is not None:
            self.metrics.actuals_rejected_total.labels("duplicate_day").inc(len(duplicates))
        self.audit.append(
            actor,
            "actuals_duplicate_rejected",
            {
                "count": len(duplicates),
                "series": sorted({r.series_id for r in duplicates})[:10],
                "days": sorted({str(r.date) for r in duplicates})[:10],
            },
        )
        _log.warning("actuals_duplicate_rejected", actor=actor, count=len(duplicates))

    def _score_accepted(
        self,
        accepted: Sequence[ActualRecord],
        artifact: CalibrationArtifact,
        training: TrainingConfig,
        segments: dict[str, str],
    ) -> tuple[int, set[int]]:
        """Score accepted actuals against the forecasts issued for them; update ACI and monitor."""
        usable = [r for r in accepted if r.stockout == 0]
        if not usable:
            return 0, set()
        forecasts = self.store.forecasts_for_targets([(r.series_id, r.date) for r in usable])
        if forecasts.empty:
            return 0, set()
        sales = {(r.series_id, r.date): float(r.sales) for r in usable}
        forecasts["y"] = [
            sales[(str(s), pd.Timestamp(d).date())]
            for s, d in zip(forecasts["series_id"], forecasts["target_date"], strict=True)
        ]
        start = self.panel.dates[0]
        forecasts["day"] = [(pd.Timestamp(d) - start).days for d in forecasts["target_date"]]
        calibrator = training.calibrator()
        labels = np.array(
            calibrator.group_labels(
                [segments[str(s)] for s in forecasts["series_id"]], forecasts["horizon"].to_numpy()
            )
        )
        y = forecasts["y"].to_numpy(dtype=np.float64)
        days = forecasts["day"].to_numpy(dtype=np.int64)
        touched: set[int] = set()
        for alpha in training.alphas:
            tag = _tag(alpha)
            lo_level, hi_level = calibrator.pairs[alpha]
            raw_lo = forecasts[f"q{lo_level:.3f}".replace(".", "_")].to_numpy(dtype=np.float64)
            raw_hi = forecasts[f"q{hi_level:.3f}".replace(".", "_")].to_numpy(dtype=np.float64)
            scores = cqr_scores(y, raw_lo, raw_hi)
            misses = (y < forecasts[f"lower_{tag}"].to_numpy()) | (
                y > forecasts[f"upper_{tag}"].to_numpy()
            )
            for day in np.unique(days):
                today = days == day
                touched.add(int(day))
                for label in np.unique(labels[today]):
                    mask = today & (labels == label)
                    artifact.aci.update(str(label), alpha, int(misses[mask].sum()), int(mask.sum()))
                    self.live_scores.append((str(label), alpha), days[mask], scores[mask])
                for segment in SEGMENTS:
                    mask = today & np.array([lab.startswith(segment) for lab in labels])
                    if mask.any():
                        self.monitor.record(
                            segment, alpha, int(day), int(mask.sum()), int(misses[mask].sum())
                        )
        return len(forecasts), touched

    def _publish_alpha_metrics(self) -> None:
        if self.metrics is None or self.artifact is None:
            return
        for (label, alpha), state in self.artifact.aci.states.items():
            segment, _, bucket = label.partition("/b")
            self.metrics.alpha_effective.labels(_tag(alpha), segment, bucket).set(state.effective)

    def _publish_drift(self, artifact: CalibrationArtifact) -> None:
        if self.metrics is None or not artifact.reference:
            return
        recent = self.store.actuals(status="accepted")
        if recent.empty:
            return
        segments = self.store.series_segments()
        recent["segment"] = [segments.get(str(s), "") for s in recent["series_id"]]
        for name, reference in artifact.reference.items():
            segment = name.rpartition("__")[2]
            current = recent.loc[recent["segment"] == segment, "sales"].to_numpy(dtype=np.float64)
            if current.size >= 30:
                self.metrics.feature_psi.labels(name).set(psi(reference, current))

    # --- calibration and recalibration --------------------------------------------------------

    def calibration_summary(self) -> dict[str, Any]:
        _, artifact, training = self._require()
        return {
            "calibration_version": artifact.version,
            "model_version": self.model_version,
            "calibrated_through_day": artifact.record.calibrated_through_day,
            "alphas": list(training.alphas),
            "window_days": artifact.record.window_days,
            "group_counts": dict(artifact.record.group_counts),
            "alpha_effective": {
                f"{label}|{_tag(alpha)}": round(state.effective, 6)
                for (label, alpha), state in sorted(artifact.aci.states.items())
            },
            "long_run_error": {
                f"{label}|{_tag(alpha)}": round(state.long_run_error, 6)
                for (label, alpha), state in sorted(artifact.aci.states.items())
            },
            "live_scores": {
                f"{g}|{_tag(a)}": self.live_scores.count((g, a)) for g, a in self.live_scores.keys
            },
            "pending_proposals": [p.id for p in self.workflow.list("pending")],
        }

    def proposals(self, status: ProposalStatus | None = None) -> list[Proposal]:
        return self.workflow.list(status)

    def reject(self, proposal_id: str, operator: str, reason: str | None = None) -> Proposal:
        proposal = self.workflow.reject(proposal_id, operator, reason)
        if self.metrics is not None:
            self.metrics.recalibration_proposals_total.labels("rejected").inc()
        return proposal

    def approve_and_apply(
        self, proposal_id: str, operator: str, reason: str | None = None
    ) -> Proposal:
        """Approve a pending proposal and promote a recalibrated version in one audited step."""
        _, artifact, training = self._require()
        self.workflow.approve(proposal_id, operator, reason)
        merged = artifact.scores.copy()
        merged.merge(self.live_scores)
        latest = merged.latest_day()
        through = (
            str(self.panel.dates[min(latest, self.panel.n_days - 1)].date())
            if latest is not None
            else artifact.record.calibrated_through_day
        )
        replacement = CalibrationArtifact.build(
            scores=merged,
            segment_map=artifact.record.segment_map,
            model_version=artifact.record.model_version,
            data_sha256=artifact.record.data_sha256,
            alphas=training.alphas,
            horizon_buckets=training.gbm.horizon_buckets,
            quantile_pairs=training.calibrator().pairs,
            reference=artifact.reference,
            calibrated_through_day=through,
            gamma=training.aci_gamma,
        )
        manifest = RunManifest.load(self.paths.manifest)
        self._install_calibration(replacement, manifest)
        self.artifact = replacement
        self.live_scores = ScoreBuffer(training.calibration_days)
        self.monitor = CoverageMonitor(self.monitor.config)
        self._save_runtime_state()
        applied = self.workflow.mark_applied(
            proposal_id, operator, from_version=artifact.version, to_version=replacement.version
        )
        if self.metrics is not None:
            self.metrics.recalibration_proposals_total.labels("applied").inc()
        _log.info("recalibration_applied", proposal=proposal_id, version=replacement.version)
        return applied

    def audit_status(self) -> dict[str, Any]:
        result = verify_chain(self.paths.audit)
        return {
            "ok": result.ok,
            "entries": result.entries,
            "head": result.head,
            "problem": result.problem,
        }


def _training_params(training: TrainingConfig) -> dict[str, Any]:
    gbm = training.gbm
    return {
        "gbm": {
            "quantiles": list(gbm.quantiles),
            "horizon_buckets": [list(b) for b in gbm.horizon_buckets],
            "num_iterations": gbm.num_iterations,
            "learning_rate": gbm.learning_rate,
            "num_leaves": gbm.num_leaves,
            "min_data_in_leaf": gbm.min_data_in_leaf,
            "feature_fraction": gbm.feature_fraction,
            "bagging_fraction": gbm.bagging_fraction,
            "bagging_freq": gbm.bagging_freq,
            "lambda_l2": gbm.lambda_l2,
            "num_threads": gbm.num_threads,
            "seed": gbm.seed,
            "tweedie_variance_power": gbm.tweedie_variance_power,
        },
        "alphas": list(training.alphas),
        "train_days": training.train_days,
        "train_stride": training.train_stride,
        "calibration_days": training.calibration_days,
        "calibration_stride": training.calibration_stride,
        "min_group_size": training.min_group_size,
        "aci_gamma": training.aci_gamma,
    }


def _training_from_params(params: dict[str, Any]) -> TrainingConfig:
    gbm = params["gbm"]
    return TrainingConfig(
        gbm=GbmConfig(
            quantiles=tuple(float(q) for q in gbm["quantiles"]),
            horizon_buckets=tuple((int(a), int(b)) for a, b in gbm["horizon_buckets"]),
            num_iterations=int(gbm["num_iterations"]),
            learning_rate=float(gbm["learning_rate"]),
            num_leaves=int(gbm["num_leaves"]),
            min_data_in_leaf=int(gbm["min_data_in_leaf"]),
            feature_fraction=float(gbm["feature_fraction"]),
            bagging_fraction=float(gbm["bagging_fraction"]),
            bagging_freq=int(gbm["bagging_freq"]),
            lambda_l2=float(gbm["lambda_l2"]),
            num_threads=int(gbm["num_threads"]),
            seed=int(gbm["seed"]),
            tweedie_variance_power=float(gbm["tweedie_variance_power"]),
        ),
        alphas=tuple(float(a) for a in params["alphas"]),
        train_days=int(params["train_days"]),
        train_stride=int(params["train_stride"]),
        calibration_days=int(params["calibration_days"]),
        calibration_stride=int(params["calibration_stride"]),
        min_group_size=int(params["min_group_size"]),
        aci_gamma=float(params["aci_gamma"]),
    )


def _lock_digest() -> str:
    """SHA-256 of uv.lock when running from a checkout, else a marker."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "uv.lock"
        if candidate.exists():
            return sha256_file(candidate)
    return "not-available"


def load_install(paths: RuntimePaths) -> dict[str, Any]:
    data = json.loads(paths.install.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ForecastRuntimeError("install.json is malformed")
    return data
