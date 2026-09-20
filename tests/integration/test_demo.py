"""The offline demo, which is the repository's narrative in one command.

It must tell a true story: split conformal degrades after a regime shift, ACI recovers, the
binomial monitor fires, an adversarial spike is quarantined, and a recalibration proposal waits
for an operator. The assertions below check that story rather than a fixed set of numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from conformal_demand_forecaster import cli
from conformal_demand_forecaster.demo import DEMO_SEED, run_demo

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def summary(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return dict(run_demo(tmp_path_factory.mktemp("demo")))


def test_the_demo_reports_the_scenario_it_ran(summary: dict[str, Any]) -> None:
    assert summary["series"] == 24
    assert summary["post_shift_days"] == 70
    assert summary["shift_day"]
    assert len(str(summary["model_version"])) == 64
    assert len(str(summary["calibration_version"])) == 64
    assert summary["mlflow_run_id"]


def test_the_pre_shift_backtest_is_close_to_nominal_coverage(summary: dict[str, Any]) -> None:
    assert summary["backtest_coverage_80_mondrian_pre_shift"] == pytest.approx(0.8, abs=0.15)
    assert summary["backtest_coverage_95_mondrian_pre_shift"] == pytest.approx(0.95, abs=0.1)
    assert summary["backtest_mase_gbm"] > 0.0
    assert summary["backtest_mase_snaive"] > 0.0


def test_aci_holds_coverage_better_than_frozen_split_conformal(summary: dict[str, Any]) -> None:
    """The headline claim: after the shift, the adaptive band tracks the target more closely."""
    static = summary["static_coverage_80_post_shift"]
    adaptive = summary["aci_coverage_80_post_shift"]
    assert abs(adaptive - 0.8) <= abs(static - 0.8)


def test_the_coverage_monitor_fires_and_opens_a_proposal(summary: dict[str, Any]) -> None:
    assert summary["first_alarm_day_after_shift"] is not None
    assert 0 <= int(summary["first_alarm_day_after_shift"]) < 70
    assert summary["proposals_opened"]
    # Nothing is promoted without an operator, so the proposal is still waiting at the end.
    assert summary["pending_proposals"]


def test_the_adversarial_spike_is_quarantined(summary: dict[str, Any]) -> None:
    assert int(summary["quarantined_actuals"]) >= 1


def test_the_audit_chain_is_intact(summary: dict[str, Any]) -> None:
    assert summary["audit_chain_ok"] is True
    assert int(summary["audit_entries"]) >= 1


def test_the_example_forecast_is_complete(summary: dict[str, Any]) -> None:
    example = summary["example_forecast"]
    assert example is not None
    assert example["horizon"] == 7
    assert set(example["intervals"]) == {"80", "95"}
    assert example["intervals"]["95"]["upper"] >= example["intervals"]["80"]["upper"]
    assert example["order"]["units"] >= 0


def test_alpha_state_moved_away_from_the_nominal_level(summary: dict[str, Any]) -> None:
    alphas = summary["aci_alpha_effective"]
    assert alphas
    assert any(abs(value - 0.2) > 1e-9 for key, value in alphas.items() if key.endswith("|80"))


def test_the_daily_tail_shows_both_coverage_tracks(summary: dict[str, Any]) -> None:
    tail = summary["daily_tail"]
    assert len(tail) == 5
    for row in tail:
        assert 0.0 <= row["static_cov_80_cum"] <= 1.0
        assert 0.0 <= row["aci_cov_80_cum"] <= 1.0
        assert row["day"]


@pytest.mark.usefixtures("restore_logging")
def test_the_cli_demo_prints_one_json_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["demo", "--workdir", str(tmp_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["demo"] == "conformal-demand-forecaster"
    assert payload["audit_chain_ok"] is True


def test_the_demo_seed_is_pinned() -> None:
    assert DEMO_SEED == 20260914
