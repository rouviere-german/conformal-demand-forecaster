"""Bench reporting: threshold enforcement, result files and the markdown tables.

These helpers decide whether CI fails, so a silently-skipped threshold would quietly retire a
README claim. The tests pin down that a missing metric is a failure, not a pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conformal_demand_forecaster.evaluation.report import (
    check_thresholds,
    load_thresholds,
    machine_info,
    markdown_table,
    write_results,
)

pytestmark = pytest.mark.unit


def write_thresholds(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_metric_inside_its_limits_passes() -> None:
    problems = check_thresholds({"latency": 5.0}, {"latency": {"max": 10.0, "min": 1.0}})
    assert problems == []


def test_a_metric_above_its_maximum_is_a_regression() -> None:
    problems = check_thresholds({"latency": 20.0}, {"latency": {"max": 10.0}})
    assert len(problems) == 1
    assert "exceeds max" in problems[0]
    assert "latency" in problems[0]


def test_a_metric_below_its_minimum_is_a_regression() -> None:
    problems = check_thresholds({"coverage": 0.5}, {"coverage": {"min": 0.8}})
    assert "is below min" in problems[0]


def test_a_missing_metric_is_a_regression_rather_than_a_silent_pass() -> None:
    """Renaming a metric must fail CI, not quietly stop enforcing the claim behind it."""
    problems = check_thresholds({"other": 1.0}, {"coverage": {"min": 0.8}})
    assert problems == ["coverage: metric missing from results"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_metric_is_a_regression(value: float) -> None:
    problems = check_thresholds({"latency": value}, {"latency": {"max": 10.0}})
    assert "is not finite" in problems[0]


def test_unlisted_metrics_are_ignored() -> None:
    assert check_thresholds({"extra": 999.0}, {}) == []


def test_thresholds_are_loaded_per_bench(tmp_path: Path) -> None:
    path = write_thresholds(
        tmp_path / "thresholds.json",
        {"baseline": {"latency.p95_us": {"max": 10}}, "backtest": {"mase": {"max": 1}}},
    )
    assert load_thresholds(path, "baseline") == {"latency.p95_us": {"max": 10.0}}
    assert load_thresholds(path, "backtest") == {"mase": {"max": 1.0}}
    assert load_thresholds(path, "absent") == {}


def test_a_malformed_threshold_section_is_refused(tmp_path: Path) -> None:
    path = write_thresholds(tmp_path / "thresholds.json", {"baseline": [1, 2, 3]})
    with pytest.raises(TypeError, match="must be an object"):
        load_thresholds(path, "baseline")


def test_the_repository_thresholds_load_for_every_bench(repo_root: Path) -> None:
    path = repo_root / "bench" / "thresholds.json"
    for bench in ("baseline", "conformal", "monitor", "shift", "backtest"):
        limits = load_thresholds(path, bench)
        assert limits, f"{bench} has no thresholds"
        for metric, bounds in limits.items():
            assert set(bounds) <= {"min", "max"}, metric


def test_results_are_written_as_sorted_json(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "results.json"
    write_results(path, {"b": 2, "a": {"seed": 1}})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"a": {"seed": 1}, "b": 2}
    assert path.read_text(encoding="utf-8").index('"a"') < path.read_text(encoding="utf-8").index(
        '"b"'
    )


def test_results_fall_back_to_str_for_unserializable_values(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    write_results(path, {"where": Path("var/panel.csv")})
    assert "panel.csv" in json.loads(path.read_text(encoding="utf-8"))["where"]


def test_machine_info_describes_the_host() -> None:
    info = machine_info()
    assert set(info) == {"platform", "python", "cpu_count", "processor", "timestamp"}
    assert isinstance(info["python"], str)
    assert str(info["timestamp"]).endswith("+00:00")


def test_markdown_table_renders_a_header_and_rows() -> None:
    table = markdown_table(["name", "value"], [["a", 1.5], ["b", 2]])
    lines = table.splitlines()
    assert lines[0] == "| name | value |"
    assert lines[1] == "|---|---|"
    assert lines[2] == "| a | 1.5000 |"
    assert lines[3] == "| b | 2 |"


def test_markdown_table_formats_by_magnitude() -> None:
    table = markdown_table(["v"], [[0.12345], [123.456], [123456.789], [float("nan")]])
    values = [line.strip("| ") for line in table.splitlines()[2:]]
    assert values[0] == "0.1235"
    assert values[1] == "123.46"
    assert values[2] == "123456.8"
    assert values[3] == "nan"


def test_markdown_table_handles_no_rows() -> None:
    assert markdown_table(["a"], []).splitlines() == ["| a |", "|---|"]
