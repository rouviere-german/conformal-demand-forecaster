"""Bench reporting: machine description, result files and regression thresholds.

``bench/thresholds.json`` maps ``<bench>.<metric>`` to ``{"max": x}`` and/or ``{"min": y}``.
A metric outside its bounds is a regression; a metric missing from the results is one too.
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["check_thresholds", "load_thresholds", "machine_info", "markdown_table", "write_results"]


def machine_info() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or "unknown",
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def load_thresholds(path: Path, bench: str) -> dict[str, dict[str, float]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    section = raw.get(bench, {})
    if not isinstance(section, dict):
        raise TypeError(f"thresholds for {bench} must be an object")
    return {
        str(metric): {str(k): float(v) for k, v in limits.items()}
        for metric, limits in section.items()
    }


def check_thresholds(
    results: Mapping[str, float], thresholds: Mapping[str, Mapping[str, float]]
) -> list[str]:
    problems: list[str] = []
    for metric, limits in thresholds.items():
        if metric not in results:
            problems.append(f"{metric}: metric missing from results")
            continue
        value = results[metric]
        if not math.isfinite(value):
            problems.append(f"{metric} = {value} is not finite")
            continue
        if "max" in limits and value > limits["max"]:
            problems.append(f"{metric} = {value:.6g} exceeds max {limits['max']}")
        if "min" in limits and value < limits["min"]:
            problems.append(f"{metric} = {value:.6g} is below min {limits['min']}")
    return problems


def write_results(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    def cell(value: object) -> str:
        if isinstance(value, float):
            if not math.isfinite(value):
                return str(value)
            if value.is_integer() and abs(value) < 1e15:
                return str(int(value))
            if abs(value) < 10:
                return f"{value:.4f}"
            return f"{value:.2f}" if abs(value) < 1000 else f"{value:.1f}"
        return str(value)

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return "\n".join(lines)
