"""Command line entry point.

Invoked as ``conformal-demand-forecaster`` or ``python -m conformal_demand_forecaster``.

Commands: ``generate`` (synthetic panel to CSV), ``train`` (install a runtime under the data
directory), ``forecast`` (batch forecasts into DuckDB), ``backtest`` (rolling-origin evaluation),
``serve`` (HTTP API), ``generate-api-key``, ``audit verify`` and ``demo``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path

__all__ = ["main"]


def _configure_cli_logging(level: str = "WARNING") -> None:
    """Send logs to stderr so every subcommand's stdout stays parseable JSON."""
    from conformal_demand_forecaster.core.logging import configure_logging  # noqa: PLC0415

    configure_logging(level, json_output=True, stream=sys.stderr)


def _serve(host: str | None, port: int | None) -> int:
    import uvicorn  # noqa: PLC0415 - only needed when serving

    from conformal_demand_forecaster.core.config import get_settings  # noqa: PLC0415
    from conformal_demand_forecaster.core.logging import configure_logging  # noqa: PLC0415
    from conformal_demand_forecaster.core.tracing import (  # noqa: PLC0415
        install_global_tracer_provider,
    )
    from conformal_demand_forecaster.serving.api import create_service_app  # noqa: PLC0415
    from conformal_demand_forecaster.serving.runtime import (  # noqa: PLC0415
        ForecastRuntime,
        RuntimePaths,
    )

    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json, stream=sys.stderr)
    runtime = ForecastRuntime(settings, RuntimePaths(settings.data_dir)).load()
    app = create_service_app(settings, runtime)
    install_global_tracer_provider(app.state.tracer_provider)
    uvicorn.run(
        app,
        host=host or settings.host,
        port=port or settings.port,
        log_config=None,
        access_log=False,
        server_header=False,
    )
    return 0


def _generate(out: Path, n_series: int, n_days: int, seed: int) -> int:
    from conformal_demand_forecaster.data.generator import (  # noqa: PLC0415
        GeneratorConfig,
        generate_panel,
    )

    frame = generate_panel(GeneratorConfig(n_series=n_series, n_days=n_days, seed=seed))
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False, lineterminator="\n")
    print(json.dumps({"rows": len(frame), "series": n_series, "days": n_days, "path": str(out)}))
    return 0


def _train(panel_path: Path, data_dir: Path, seed: int) -> int:
    import pandas as pd  # noqa: PLC0415

    from conformal_demand_forecaster.core.config import get_settings  # noqa: PLC0415
    from conformal_demand_forecaster.serving.runtime import (  # noqa: PLC0415
        ForecastRuntime,
        RuntimePaths,
    )

    settings = get_settings().model_copy(update={"data_dir": data_dir})
    frame = pd.read_csv(panel_path, parse_dates=["date"])
    runtime = ForecastRuntime(settings, RuntimePaths(data_dir))
    try:
        summary = runtime.train_and_install(
            frame, seed=seed, generator_params={"source": str(panel_path)}
        )
    finally:
        runtime.close()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _forecast(data_dir: Path, origin: str | None) -> int:
    from conformal_demand_forecaster.core.config import get_settings  # noqa: PLC0415
    from conformal_demand_forecaster.serving.runtime import (  # noqa: PLC0415
        ForecastRuntime,
        RuntimePaths,
    )

    settings = get_settings().model_copy(update={"data_dir": data_dir})
    runtime = ForecastRuntime(settings, RuntimePaths(data_dir)).load()
    try:
        batch = runtime.run_forecast(date.fromisoformat(origin) if origin else None)
    finally:
        runtime.close()
    print(json.dumps({"origin": str(batch.origin), "rows": batch.rows}))
    return 0


def _backtest(panel_path: Path, n_origins: int, out: Path | None, ets_limit: int | None) -> int:
    import pandas as pd  # noqa: PLC0415

    from conformal_demand_forecaster.data.panel import Panel  # noqa: PLC0415
    from conformal_demand_forecaster.evaluation.backtest import (  # noqa: PLC0415
        BacktestConfig,
        run_backtest,
    )

    panel = Panel.from_frame(pd.read_csv(panel_path, parse_dates=["date"]))
    result = run_backtest(panel, BacktestConfig(n_origins=n_origins, ets_series_limit=ets_limit))
    payload = {
        "metrics": result.metrics,
        "per_origin": result.per_origin,
        "digest": result.digest(),
    }
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))
    return 0


def _demo(workdir: Path) -> int:
    from conformal_demand_forecaster.demo import run_demo  # noqa: PLC0415

    summary = run_demo(workdir)
    print(json.dumps({"demo": "conformal-demand-forecaster", **summary}, indent=2, sort_keys=True))
    return 0


def _generate_api_key() -> int:
    from conformal_demand_forecaster.core.security import generate_api_key  # noqa: PLC0415

    raw, digest = generate_api_key()
    print(json.dumps({"api_key": raw, "api_key_sha256": digest}, indent=2))
    print("Store only api_key_sha256 in APP_<ROLE>_KEY_SHA256.", file=sys.stderr)
    return 0


def _audit_verify(data_dir: Path, expect_head: str | None) -> int:
    from conformal_demand_forecaster.monitoring.audit import verify_chain  # noqa: PLC0415

    result = verify_chain(data_dir / "audit.jsonl")
    ok = result.ok and (expect_head is None or result.head == expect_head)
    print(
        json.dumps(
            {"ok": ok, "entries": result.entries, "head": result.head, "problem": result.problem}
        )
    )
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="conformal-demand-forecaster")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP service")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    generate = sub.add_parser("generate", help="write a synthetic sales panel to CSV")
    generate.add_argument("--out", type=Path, default=Path("var/panel.csv"))
    generate.add_argument("--n-series", type=int, default=200)
    generate.add_argument("--n-days", type=int, default=1096)
    generate.add_argument("--seed", type=int, default=20260914)

    train = sub.add_parser("train", help="train, calibrate and install a runtime")
    train.add_argument("--panel", type=Path, default=Path("var/panel.csv"))
    train.add_argument("--data-dir", type=Path, default=Path("var"))
    train.add_argument("--seed", type=int, default=20260914)

    forecast = sub.add_parser("forecast", help="write a batch of forecasts to DuckDB")
    forecast.add_argument("--data-dir", type=Path, default=Path("var"))
    forecast.add_argument("--origin", default=None, help="YYYY-MM-DD (default: last observed day)")

    backtest = sub.add_parser("backtest", help="rolling-origin backtest of a panel")
    backtest.add_argument("--panel", type=Path, default=Path("var/panel.csv"))
    backtest.add_argument("--n-origins", type=int, default=12)
    backtest.add_argument("--out", type=Path, default=None)
    backtest.add_argument("--ets-series-limit", type=int, default=None)

    sub.add_parser("generate-api-key", help="print a new API key and its SHA-256 digest")

    audit = sub.add_parser("audit", help="audit log tools")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True)
    verify = audit_sub.add_parser("verify", help="verify the hash-chained audit log")
    verify.add_argument("--data-dir", type=Path, default=Path("var"))
    verify.add_argument("--expect-head", default=None)

    demo = sub.add_parser("demo", help="run the offline demo")
    demo.add_argument("--workdir", type=Path, default=Path("var/demo"))

    args = parser.parse_args(argv)
    if args.command != "serve":
        _configure_cli_logging()
    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "serve": lambda a: _serve(a.host, a.port),
        "generate": lambda a: _generate(a.out, a.n_series, a.n_days, a.seed),
        "train": lambda a: _train(a.panel, a.data_dir, a.seed),
        "forecast": lambda a: _forecast(a.data_dir, a.origin),
        "backtest": lambda a: _backtest(a.panel, a.n_origins, a.out, a.ets_series_limit),
        "generate-api-key": lambda a: _generate_api_key(),
        "audit": lambda a: _audit_verify(a.data_dir, a.expect_head),
        "demo": lambda a: _demo(a.workdir),
    }
    return handlers[str(args.command)](args)


if __name__ == "__main__":
    raise SystemExit(main())
