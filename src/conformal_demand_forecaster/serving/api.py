"""Role-based HTTP API on top of the hardened app factory.

- ``GET /v1/forecast/{series_id}`` (reader): point, intervals, calibration version and order.
- ``POST /v1/actuals`` (ingestor): validated, rate-limited, guard-screened actuals ingest.
- ``GET /v1/calibration`` (reader): calibration version, alpha states, pending proposals.
- ``GET /v1/recalibration/proposals`` (operator), ``POST .../{id}/approve`` and ``/reject``.
- ``GET /v1/audit`` (operator): hash-chain verification of the decision log.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import FastAPI, Path, Query
from prometheus_client import CollectorRegistry
from pydantic import BaseModel, ConfigDict, Field

from conformal_demand_forecaster.core.app import create_app
from conformal_demand_forecaster.core.config import Settings
from conformal_demand_forecaster.core.errors import AppError
from conformal_demand_forecaster.core.logging import get_logger
from conformal_demand_forecaster.ml.contracts import SERIES_ID_PATTERN, DataContractError
from conformal_demand_forecaster.monitoring.metrics import create_forecast_metrics
from conformal_demand_forecaster.monitoring.recalibration import ProposalStatus, WorkflowError
from conformal_demand_forecaster.serving.roles import (
    IngestorPrincipal,
    OperatorPrincipal,
    ReaderPrincipal,
)
from conformal_demand_forecaster.serving.runtime import ActualRecord, ForecastRuntime, IngestSummary

__all__ = ["DecisionRequest", "IngestRequest", "create_service_app"]

_log = get_logger("conformal_demand_forecaster.api")
_MAX_RECORDS = 100_000
# The hard ceiling of Settings.forecast_horizon; the configured value is enforced per request.
MAX_FORECAST_HORIZON = 90


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[ActualRecord] = Field(min_length=1, max_length=_MAX_RECORDS)


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str | None = Field(default=None, max_length=200)


# These aliases must be resolvable at module scope: ``from __future__ import annotations`` turns
# every parameter annotation into a string, and FastAPI rebuilds them without the enclosing
# function's locals. A bound that depends on settings is therefore checked inside the handler.
SeriesId = Annotated[str, Path(pattern=SERIES_ID_PATTERN, max_length=32)]
ProposalId = Annotated[str, Path(pattern=r"^[0-9a-f]{32}$")]
Horizon = Annotated[int, Query(ge=1, le=MAX_FORECAST_HORIZON)]
Cost = Annotated[float | None, Query(gt=0, le=1e6)]
ProposalFilter = Annotated[ProposalStatus | None, Query()]


def create_service_app(
    settings: Settings,
    runtime: ForecastRuntime,
    *,
    registry: CollectorRegistry | None = None,
) -> FastAPI:
    app = create_app(
        settings,
        readiness_checks={
            "calibration_loaded": runtime.calibration_loaded,
            "forecast_freshness": runtime.forecasts_fresh,
        },
        registry=registry,
    )
    runtime.bind_metrics(create_forecast_metrics(app.state.metrics.registry))
    app.state.runtime = runtime

    @app.get("/v1/forecast/{series_id}")
    async def forecast(
        series_id: SeriesId,
        principal: ReaderPrincipal,
        horizon: Horizon = 1,
        holding_cost: Cost = None,
        stockout_cost: Cost = None,
    ) -> dict[str, Any]:
        if horizon > settings.forecast_horizon:
            raise AppError(422, "invalid_horizon", "Horizon is beyond the configured maximum.")
        result = runtime.get_forecast(
            series_id, horizon, holding_cost=holding_cost, stockout_cost=stockout_cost
        )
        if result is None:
            raise AppError(404, "not_found", "No forecast for this series and horizon.")
        _log.info("forecast_read", key_id=principal.key_id, horizon=horizon)
        return result

    @app.post("/v1/actuals", response_model=IngestSummary)
    async def ingest(body: IngestRequest, principal: IngestorPrincipal) -> IngestSummary:
        if len(body.records) > settings.ingest_max_records:
            raise AppError(413, "too_many_records", "Too many records in one request.")
        try:
            summary = runtime.ingest(body.records, actor=f"key:{principal.key_id}")
        except DataContractError as exc:
            raise AppError(422, "contract_violation", "Actuals violate the data contract.") from exc
        _log.info(
            "actuals_ingested",
            key_id=principal.key_id,
            accepted=summary.accepted,
            quarantined=summary.quarantined,
        )
        return summary

    @app.get("/v1/calibration")
    async def calibration(principal: ReaderPrincipal) -> dict[str, Any]:
        del principal
        return runtime.calibration_summary()

    @app.get("/v1/recalibration/proposals")
    async def proposals(
        principal: OperatorPrincipal, status: ProposalFilter = None
    ) -> list[dict[str, Any]]:
        del principal
        return [item.model_dump(mode="json") for item in runtime.proposals(status)]

    @app.post("/v1/recalibration/proposals/{proposal_id}/approve")
    async def approve(
        proposal_id: ProposalId, body: DecisionRequest, principal: OperatorPrincipal
    ) -> dict[str, Any]:
        try:
            applied = runtime.approve_and_apply(proposal_id, f"key:{principal.key_id}", body.reason)
        except WorkflowError as exc:
            raise AppError(409, "invalid_transition", "Proposal cannot be approved.") from exc
        return applied.model_dump(mode="json")

    @app.post("/v1/recalibration/proposals/{proposal_id}/reject")
    async def reject(
        proposal_id: ProposalId, body: DecisionRequest, principal: OperatorPrincipal
    ) -> dict[str, Any]:
        try:
            rejected = runtime.reject(proposal_id, f"key:{principal.key_id}", body.reason)
        except WorkflowError as exc:
            raise AppError(409, "invalid_transition", "Proposal cannot be rejected.") from exc
        return rejected.model_dump(mode="json")

    @app.get("/v1/audit")
    async def audit(principal: OperatorPrincipal) -> dict[str, Any]:
        del principal
        return runtime.audit_status()

    return app
