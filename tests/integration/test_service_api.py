"""The role-based HTTP API.

Three roles, three disjoint capabilities: a reader can read forecasts, an ingestor can post
actuals, and only an operator can decide a recalibration. The tests drive the real runtime
behind the real middleware stack, so authentication, role checks, request validation, rate
limiting and error shapes are exercised together rather than mocked.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from conformal_demand_forecaster.core.config import Settings
from conformal_demand_forecaster.core.security import hash_api_key
from conformal_demand_forecaster.data.generator import GeneratorConfig, generate_panel
from conformal_demand_forecaster.data.panel import Panel
from conformal_demand_forecaster.models.gbm import GbmConfig
from conformal_demand_forecaster.serving.api import create_service_app
from conformal_demand_forecaster.serving.runtime import ForecastRuntime, RuntimePaths

pytestmark = [pytest.mark.integration, pytest.mark.slow]

READER_KEY = "reader-key-" + "r" * 24
INGESTOR_KEY = "ingestor-key-" + "i" * 24
OPERATOR_KEY = "operator-key-" + "o" * 24
READER = {"X-API-Key": READER_KEY}
INGESTOR = {"X-API-Key": INGESTOR_KEY}
OPERATOR = {"X-API-Key": OPERATOR_KEY}

N_DAYS = 220
GBM = GbmConfig(
    horizon_buckets=((1, 7),), num_iterations=20, num_leaves=7, min_data_in_leaf=10, num_threads=2
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return generate_panel(GeneratorConfig(n_series=6, n_days=N_DAYS, seed=555, shifts=()))


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "environment": "test",
        "data_dir": tmp_path,
        "reader_key_sha256": [hash_api_key(READER_KEY), hash_api_key(OPERATOR_KEY)],
        "ingestor_key_sha256": [hash_api_key(INGESTOR_KEY)],
        "operator_key_sha256": [hash_api_key(OPERATOR_KEY)],
        "forecast_horizon": 7,
        "calibration_window_days": 30,
        "coverage_window_days": 10,
        "coverage_min_observations": 20,
        "rate_limit_per_second": 1000.0,
        "rate_limit_burst": 1000,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def service(
    frame: pd.DataFrame, tmp_path: Path
) -> Iterator[tuple[TestClient, ForecastRuntime, Panel]]:
    panel = Panel.from_frame(frame, validate=False)
    settings = make_settings(tmp_path)
    today = panel.dates[-1].date()
    runtime = ForecastRuntime(
        settings,
        RuntimePaths(tmp_path / "runtime"),
        clock=lambda: datetime.combine(today, datetime.min.time(), tzinfo=UTC),
    )
    runtime.train_and_install(frame, seed=555, generator_params={}, gbm=GBM, run_name="api-test")
    app = create_service_app(settings, runtime, registry=CollectorRegistry())
    with TestClient(app, raise_server_exceptions=False) as client:
        try:
            yield client, runtime, panel
        finally:
            runtime.close()


def actual_payload(panel: Panel, day_offset: int = 1) -> dict[str, Any]:
    target = (panel.dates[-1] + pd.Timedelta(days=day_offset)).date()
    return {
        "records": [
            {"series_id": series_id, "date": str(target), "sales": 5, "stockout": 0}
            for series_id in panel.series_ids
        ]
    }


# --- probes ---------------------------------------------------------------------------------


def test_health_is_public_and_readiness_reports_the_forecast_checks(
    service: tuple[TestClient, ForecastRuntime, Panel],
) -> None:
    client, _, _ = service
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["checks"] == {"calibration_loaded": "ok", "forecast_freshness": "ok"}


def test_readiness_fails_when_forecasts_go_stale(frame: pd.DataFrame, tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    panel = Panel.from_frame(frame, validate=False)
    clock = {"now": datetime.combine(panel.dates[-1].date(), datetime.min.time(), tzinfo=UTC)}
    runtime = ForecastRuntime(
        settings, RuntimePaths(tmp_path / "runtime"), clock=lambda: clock["now"]
    )
    runtime.train_and_install(frame, seed=555, generator_params={}, gbm=GBM, run_name="stale")
    app = create_service_app(settings, runtime, registry=CollectorRegistry())
    with TestClient(app, raise_server_exceptions=False) as client:
        try:
            clock["now"] = clock["now"] + timedelta(days=30)
            response = client.get("/readyz")
            assert response.status_code == 503
            assert response.json()["checks"]["forecast_freshness"] == "fail"
        finally:
            runtime.close()


# --- authentication -------------------------------------------------------------------------


def test_every_forecast_route_needs_a_key(
    service: tuple[TestClient, ForecastRuntime, Panel],
) -> None:
    client, _, panel = service
    series = panel.series_ids[0]
    assert client.get(f"/v1/forecast/{series}").status_code == 401
    assert client.get("/v1/calibration").status_code == 401
    assert client.post("/v1/actuals", json=actual_payload(panel)).status_code == 401
    assert client.get("/v1/recalibration/proposals").status_code == 401
    assert client.get("/v1/audit").status_code == 401


def test_an_unknown_key_is_rejected_without_revealing_anything(
    service: tuple[TestClient, ForecastRuntime, Panel],
) -> None:
    client, _, panel = service
    response = client.get(f"/v1/forecast/{panel.series_ids[0]}", headers={"X-API-Key": "nope"})
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "unauthorized"
    assert "nope" not in response.text


def test_a_bearer_token_is_accepted_too(
    service: tuple[TestClient, ForecastRuntime, Panel],
) -> None:
    client, _, panel = service
    response = client.get(
        f"/v1/forecast/{panel.series_ids[0]}",
        headers={"Authorization": f"Bearer {READER_KEY}"},
    )
    assert response.status_code == 200


# --- roles ----------------------------------------------------------------------------------


def test_a_reader_key_cannot_ingest_or_decide(
    service: tuple[TestClient, ForecastRuntime, Panel],
) -> None:
    client, _, panel = service
    ingest = client.post("/v1/actuals", json=actual_payload(panel), headers=READER)
    assert ingest.status_code == 403
    assert ingest.json()["error"]["code"] == "forbidden"
    assert client.get("/v1/recalibration/proposals", headers=READER).status_code == 403
    assert client.get("/v1/audit", headers=READER).status_code == 403


def test_an_ingestor_key_cannot_read_forecasts_or_decide(
    service: tuple[TestClient, ForecastRuntime, Panel],
) -> None:
    client, _, panel = service
    assert client.get(f"/v1/forecast/{panel.series_ids[0]}", headers=INGESTOR).status_code == 403
    assert client.get("/v1/calibration", headers=INGESTOR).status_code == 403
    assert client.get("/v1/recalibration/proposals", headers=INGESTOR).status_code == 403


def test_an_operator_key_can_reach_the_decision_routes(
    service: tuple[TestClient, ForecastRuntime, Panel],
) -> None:
    client, _, _ = service
    proposals = client.get("/v1/recalibration/proposals", headers=OPERATOR)
    assert proposals.status_code == 200
    assert proposals.json() == []
    audit = client.get("/v1/audit", headers=OPERATOR)
    assert audit.status_code == 200
    assert audit.json()["ok"] is True
    assert audit.json()["entries"] >= 1


def test_an_operator_cannot_ingest_actuals(
    service: tuple[TestClient, ForecastRuntime, Panel],
) -> None:
    """Roles are checked per endpoint, so holding two roles does not imply the third."""
    client, _, panel = service
    assert (
        client.post("/v1/actuals", json=actual_payload(panel), headers=OPERATOR).status_code == 403
    )


# --- forecasts ------------------------------------------------------------------------------


def test_a_forecast_response_carries_intervals_versions_and_an_order(
    service: tuple[TestClient, ForecastRuntime, Panel],
) -> None:
    client, _, panel = service
    response = client.get(f"/v1/forecast/{panel.series_ids[0]}?horizon=3", headers=READER)
    assert response.status_code == 200
    body = response.json()
    assert body["series_id"] == panel.series_ids[0]
    assert body["horizon"] == 3
    assert set(body["intervals"]) == {"80", "95"}
    assert body["intervals"]["95"]["upper"] >= body["intervals"]["80"]["upper"]
    assert len(body["calibration_version"]) == 64
    assert body["order"]["units"] >= 0


def test_cost_overrides_are_bounded_and_applied(
    service: tuple[TestClient, ForecastRuntime, Panel],
) -> None:
    client, _, panel = service
    series = panel.series_ids[0]
    cautious = client.get(
        f"/v1/forecast/{series}?holding_cost=1&stockout_cost=100", headers=READER
    ).json()
    thrifty = client.get(
        f"/v1/forecast/{series}?holding_cost=100&stockout_cost=1", headers=READER
    ).json()
    assert cautious["order"]["quantity"] >= thrifty["order"]["quantity"]
    assert client.get(f"/v1/forecast/{series}?holding_cost=0", headers=READER).status_code == 422


@pytest.mark.parametrize("horizon", ["0", "99", "abc"])
def test_out_of_range_horizons_are_rejected_by_validation(
    service: tuple[TestClient, ForecastRuntime, Panel], horizon: str
) -> None:
    client, _, panel = service
    response = client.get(f"/v1/forecast/{panel.series_ids[0]}?horizon={horizon}", headers=READER)
    assert response.status_code == 422


@pytest.mark.parametrize("series_id", ["../../etc/passwd", "a b", "x" * 40, "drop;table"])
def test_malformed_series_ids_never_reach_the_runtime(
    service: tuple[TestClient, ForecastRuntime, Panel], series_id: str
) -> None:
    response = service[0].get(f"/v1/forecast/{series_id}", headers=READER)
    assert response.status_code in {404, 422}


def test_an_unknown_but_well_formed_series_is_a_clean_404(
    service: tuple[TestClient, ForecastRuntime, Panel],
) -> None:
    response = service[0].get("/v1/forecast/S9999_ST99", headers=READER)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


# --- ingest ---------------------------------------------------------------------------------


def test_an_ingestor_can_post_actuals_and_gets_a_summary(
    service: tuple[TestClient, ForecastRuntime, Panel],
) -> None:
    client, _, panel = service
    response = client.post("/v1/actuals", json=actual_payload(panel), headers=INGESTOR)
    assert response.status_code == 200
    body = response.json()
    assert body["received"] == len(panel.series_ids)
    assert body["accepted"] + body["quarantined"] == len(panel.series_ids)
    assert body["duplicates"] == 0
    assert body["over_coverage_alarms"] == []
    assert len(body["calibration_version"]) == 64


def test_reposting_the_same_day_is_reported_and_changes_nothing(
    service: tuple[TestClient, ForecastRuntime, Panel],
) -> None:
    client, runtime, panel = service
    payload = actual_payload(panel)
    first = client.post("/v1/actuals", json=payload, headers=INGESTOR).json()
    assert first["accepted"] >= 1
    stored = runtime.store.load_panel()["sales"].tolist()

    replay = client.post("/v1/actuals", json=payload, headers=INGESTOR)

    assert replay.status_code == 200
    body = replay.json()
    assert body["duplicates"] == first["accepted"]
    assert body["accepted"] == 0
    assert runtime.store.load_panel()["sales"].tolist() == stored


def test_a_date_outside_the_serving_window_is_refused(
    service: tuple[TestClient, ForecastRuntime, Panel],
) -> None:
    client, runtime, panel = service
    payload = actual_payload(panel)
    payload["records"] = payload["records"][:1]
    payload["records"][0]["date"] = "2045-01-01"

    response = client.post("/v1/actuals", json=payload, headers=INGESTOR)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "contract_violation"
    assert runtime.panel.n_days == panel.n_days


def test_a_poisoned_spike_is_quarantined_by_the_api_path(
    service: tuple[TestClient, ForecastRuntime, Panel],
) -> None:
    client, runtime, panel = service
    payload = actual_payload(panel)
    payload["records"][0]["sales"] = 90_000
    response = client.post("/v1/actuals", json=payload, headers=INGESTOR)
    assert response.status_code == 200
    assert response.json()["quarantined"] >= 1
    quarantined = runtime.store.actuals(status="quarantined")
    assert panel.series_ids[0] in set(quarantined["series_id"])


@pytest.mark.parametrize(
    "record",
    [
        {"series_id": "S0001_ST00", "date": "2024-01-01", "sales": -5},
        {"series_id": "bad id", "date": "2024-01-01", "sales": 5},
        {"series_id": "S0001_ST00", "date": "not-a-date", "sales": 5},
        {"series_id": "S0001_ST00", "date": "2024-01-01", "sales": 5, "extra": 1},
        {"series_id": "S0001_ST00", "date": "2024-01-01", "sales": 999_999_999},
    ],
)
def test_malformed_records_are_rejected_before_the_runtime(
    service: tuple[TestClient, ForecastRuntime, Panel], record: dict[str, Any]
) -> None:
    response = service[0].post("/v1/actuals", json={"records": [record]}, headers=INGESTOR)
    assert response.status_code == 422


def test_an_empty_or_oversized_batch_is_rejected(
    service: tuple[TestClient, ForecastRuntime, Panel],
) -> None:
    client, _, panel = service
    assert client.post("/v1/actuals", json={"records": []}, headers=INGESTOR).status_code == 422
    target = str((panel.dates[-1] + pd.Timedelta(days=2)).date())
    many = {
        "records": [
            {"series_id": f"S{i:04d}_ST00", "date": target, "sales": 1} for i in range(1_001)
        ]
    }
    assert client.post("/v1/actuals", json=many, headers=INGESTOR).status_code == 413


def test_duplicate_records_are_a_contract_violation(
    service: tuple[TestClient, ForecastRuntime, Panel],
) -> None:
    client, _, panel = service
    payload = actual_payload(panel)
    payload["records"].append(payload["records"][0])
    response = client.post("/v1/actuals", json=payload, headers=INGESTOR)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "contract_violation"


# --- recalibration --------------------------------------------------------------------------


def test_deciding_an_unknown_proposal_is_a_clean_conflict(
    service: tuple[TestClient, ForecastRuntime, Panel],
) -> None:
    client, _, _ = service
    for action in ("approve", "reject"):
        response = client.post(
            f"/v1/recalibration/proposals/{'f' * 32}/{action}", json={}, headers=OPERATOR
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "invalid_transition"


def test_a_malformed_proposal_id_is_rejected_by_validation(
    service: tuple[TestClient, ForecastRuntime, Panel],
) -> None:
    response = service[0].post(
        "/v1/recalibration/proposals/not-an-id/approve", json={}, headers=OPERATOR
    )
    assert response.status_code == 422


def test_an_over_long_decision_reason_is_rejected(
    service: tuple[TestClient, ForecastRuntime, Panel],
) -> None:
    response = service[0].post(
        f"/v1/recalibration/proposals/{'f' * 32}/reject",
        json={"reason": "x" * 500},
        headers=OPERATOR,
    )
    assert response.status_code == 422


def test_the_calibration_route_reports_versions_and_alpha_state(
    service: tuple[TestClient, ForecastRuntime, Panel],
) -> None:
    body = service[0].get("/v1/calibration", headers=READER).json()
    assert len(body["calibration_version"]) == 64
    assert body["alphas"] == [0.2, 0.05]
    assert body["alpha_effective"]
    assert body["pending_proposals"] == []


# --- hardening ------------------------------------------------------------------------------


def test_metrics_require_a_key_and_expose_the_forecast_gauges(
    service: tuple[TestClient, ForecastRuntime, Panel],
) -> None:
    client, _, _ = service
    assert client.get("/metrics").status_code == 401
    body = client.get("/metrics", headers=READER).text
    assert "app_interval_width" in body
    assert "app_alpha_effective" in body


def test_docs_are_disabled_by_default(
    service: tuple[TestClient, ForecastRuntime, Panel],
) -> None:
    client, _, _ = service
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_rate_limiting_applies_to_the_forecast_route(frame: pd.DataFrame, tmp_path: Path) -> None:
    settings = make_settings(tmp_path, rate_limit_per_second=1.0, rate_limit_burst=3)
    panel = Panel.from_frame(frame, validate=False)
    today = panel.dates[-1].date()
    runtime = ForecastRuntime(
        settings,
        RuntimePaths(tmp_path / "runtime"),
        clock=lambda: datetime.combine(today, datetime.min.time(), tzinfo=UTC),
    )
    runtime.train_and_install(frame, seed=555, generator_params={}, gbm=GBM, run_name="rl")
    app = create_service_app(settings, runtime, registry=CollectorRegistry())
    url = f"/v1/forecast/{panel.series_ids[0]}"
    with TestClient(app, raise_server_exceptions=False) as client:
        try:
            statuses = [client.get(url, headers=READER).status_code for _ in range(10)]
        finally:
            runtime.close()
    assert 200 in statuses
    assert 429 in statuses
