# ============================================================
# src/observability/logger.py - Structured Logging
#
# Learning Note:
#   In production, plain print() statements are not enough.
#   We use 'structlog' to emit JSON-formatted logs. This makes
#   it easy for tools like Google Cloud Logging to parse, filter,
#   and alert on log entries. Every log entry carries context
#   fields (request_id, user_id, node_name) for traceability.
# ============================================================

import structlog
import logging
import sys
from src.config import settings


def configure_logging() -> None:
    """
    Configure structlog for either human-readable (dev) or
    JSON (production) output. Call this once at app startup.
    """
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Processors transform log records before output.
    # They run in order - each one adds or transforms data.
    shared_processors = [
        structlog.contextvars.merge_contextvars,          # adds bound context vars
        structlog.stdlib.add_log_level,                   # adds "level" field
        structlog.stdlib.add_logger_name,                 # adds "logger" field
        structlog.processors.TimeStamper(fmt="iso"),      # adds ISO timestamp
        structlog.processors.StackInfoRenderer(),         # renders stack traces
    ]

    if settings.environment == "development":
        # Pretty colored output for local development
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        # JSON output for production - parseable by Cloud Logging
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a named logger. Usage:
        logger = get_logger(__name__)
        logger.info("retrieval_complete", chunks_found=5, query="...")
    """
    return structlog.get_logger(name)
