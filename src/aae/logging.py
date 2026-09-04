"""Structured logging.

Emits JSON outside development so records are machine-readable, and a coloured
console renderer locally. A ``correlation_id`` is bound once per decision and
propagates through every stage of the pipeline, which is what makes a single
decision traceable end to end.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from aae.config import Settings


def configure_logging(settings: Settings) -> None:
    """Configure structlog and the stdlib logging bridge.

    Safe to call more than once; the last call wins.

    Args:
        settings: Provides the log level and environment.
    """
    from aae.config import Environment

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    shared: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: structlog.typing.Processor = (
        structlog.dev.ConsoleRenderer()
        if settings.env is Environment.DEVELOPMENT
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger.

    Args:
        name: Usually ``__name__``.

    Returns:
        A structlog logger.
    """
    return structlog.stdlib.get_logger(name)


def bind_correlation_id(correlation_id: str) -> None:
    """Bind a correlation id to the current logging context.

    Args:
        correlation_id: Typically the decision id.
    """
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
