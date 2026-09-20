"""structlog JSON logging with PII redaction for both structlog and stdlib (uvicorn) records."""

from __future__ import annotations

import logging
import sys
from typing import Any, TextIO, cast

import structlog
from structlog.typing import FilteringBoundLogger, Processor

from conformal_demand_forecaster.core.redaction import redact_processor

__all__ = ["configure_logging", "get_logger"]


def _shared_processors() -> list[Processor]:
    # Redaction runs after exception formatting so tracebacks are scrubbed too.
    return [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        redact_processor,
    ]


def configure_logging(
    level: str = "INFO", *, json_output: bool = True, stream: TextIO | None = None
) -> None:
    """Configure structlog and route stdlib logging through the same redacting pipeline."""
    level_no = logging.getLevelNamesMapping()[level.upper()]
    output: TextIO = stream if stream is not None else sys.stdout
    renderer: Processor = (
        structlog.processors.JSONRenderer(sort_keys=True)
        if json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    shared = _shared_processors()
    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level_no),
        logger_factory=structlog.PrintLoggerFactory(file=output),
        cache_logger_on_first_use=False,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )
    handler = logging.StreamHandler(output)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level_no)
    for noisy in ("uvicorn", "uvicorn.error"):
        logging.getLogger(noisy).handlers = []
        logging.getLogger(noisy).propagate = True
    # Access logs would print raw query strings; RequestContextMiddleware logs paths instead.
    logging.getLogger("uvicorn.access").disabled = True


def get_logger(name: str | None = None, **initial_values: Any) -> FilteringBoundLogger:
    """Return a structlog logger.

    With no initial values the lazy proxy is returned unbound, because binding materializes it
    against whatever configuration is active at that moment. Modules create their logger at
    import time, which usually happens before :func:`configure_logging` runs; an eagerly bound
    logger would keep writing to structlog's default sink (stdout, console renderer) for the
    rest of the process, which in a CLI means log lines land in the middle of JSON output.
    """
    logger = structlog.get_logger(name) if name else structlog.get_logger()
    if initial_values:
        return cast(FilteringBoundLogger, logger.bind(**initial_values))
    return cast(FilteringBoundLogger, logger)
