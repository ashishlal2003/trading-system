"""
Centralised logging configuration for the trading system.

Usage
-----
In any module::

    from src.utils.logger import get_logger

    log = get_logger(__name__)
    log.info("order placed", symbol="RELIANCE", qty=10, price=2850.50)

Call ``configure_logging()`` once at application startup (e.g. in main.py)
before importing any other module that calls ``get_logger``.
"""

from __future__ import annotations

import logging
import sys
from typing import Any


import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """
    Set up structlog with human-readable output when attached to a TTY,
    and JSON output when writing to a file / pipe (e.g. in production).

    Parameters
    ----------
    log_level:
        One of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``.
        Defaults to ``INFO``.
    """
    numeric_level: int = getattr(logging, log_level.upper(), logging.INFO)

    # Also configure stdlib logging so third-party libraries respect the level.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    # Choose renderer based on whether stderr is a real terminal.
    renderer: Any = (
        structlog.dev.ConsoleRenderer(colors=True)
        if sys.stderr.isatty()
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[
            # Merge any context variables bound with structlog.contextvars.bind_contextvars().
            structlog.contextvars.merge_contextvars,
            # Add log level string ("info", "warning", …).
            structlog.processors.add_log_level,
            # ISO-8601 timestamp.
            structlog.processors.TimeStamper(fmt="iso"),
            # Render the stack trace when exc_info is present.
            structlog.processors.StackInfoRenderer(),
            # Format exceptions neatly.
            structlog.processors.ExceptionRenderer(),
            # Final renderer: pretty in dev, JSON in prod.
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Return a structlog bound logger for the given *name*.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module, e.g.
        ``"src.broker.client"``.

    Returns
    -------
    structlog.BoundLogger
        A pre-bound logger instance that carries the *name* as a context key.
    """
    return structlog.get_logger(name)
