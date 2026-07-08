"""Logging utilities for the Music Gen Framework.

Provides a single :func:`setup_logging` entry point that configures the
root framework logger with:

- Rich, colorized console output when ``rich`` is installed and the
  environment supports it (falls back gracefully to plain ``logging``
  formatting otherwise, e.g. in some CI environments).
- Optional rotating file logging.
- Optional JSON-lines file format for downstream log aggregation.

All other modules should obtain loggers via :func:`get_logger` rather than
calling ``logging.getLogger`` directly, so that configuration stays
centralized.
"""

from __future__ import annotations

import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

_FRAMEWORK_LOGGER_NAME = "music_gen_framework"
_configured = False


class JsonLinesFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _build_console_handler(use_rich: bool, level: int) -> logging.Handler:
    """Build the console log handler, preferring ``rich`` when available.

    Args:
        use_rich: Whether to attempt using ``rich.logging.RichHandler``.
        level: Minimum log level for this handler.

    Returns:
        A configured ``logging.Handler`` instance.
    """
    if use_rich:
        try:
            from rich.logging import RichHandler

            handler: logging.Handler = RichHandler(
                level=level,
                show_time=True,
                show_path=False,
                rich_tracebacks=True,
                markup=False,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            return handler
        except ImportError:
            logging.getLogger(_FRAMEWORK_LOGGER_NAME).debug(
                "rich is not installed; falling back to standard console formatting."
            )

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    return handler


def _build_file_handler(
    log_dir: Path,
    filename: str,
    level: int,
    json_format: bool,
    max_bytes: int,
    backup_count: int,
) -> logging.Handler:
    """Build a rotating file handler for persistent log storage.

    Args:
        log_dir: Directory in which to create the log file.
        filename: Name of the log file.
        level: Minimum log level for this handler.
        json_format: Whether to use JSON-lines formatting.
        max_bytes: Maximum size in bytes before the log file is rotated.
        backup_count: Number of rotated backups to retain.

    Returns:
        A configured ``RotatingFileHandler`` instance.

    Raises:
        OSError: If ``log_dir`` cannot be created.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / filename
    handler = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)
    if json_format:
        handler.setFormatter(JsonLinesFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
    return handler


def setup_logging(
    level: str = "INFO",
    *,
    log_dir: Optional[Path] = None,
    log_filename: str = "music_gen_framework.log",
    log_to_file: bool = True,
    use_rich: bool = True,
    json_format: bool = False,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """Configure the framework's root logger.

    This function is idempotent: calling it multiple times reconfigures
    handlers on the existing framework logger rather than stacking
    duplicate handlers.

    Args:
        level: Minimum log level name (e.g. ``"DEBUG"``, ``"INFO"``).
        log_dir: Directory for file logging. Required if ``log_to_file``
            is True.
        log_filename: Name of the log file, used only if ``log_to_file``.
        log_to_file: Whether to also write logs to a rotating file.
        use_rich: Whether to prefer ``rich`` console formatting.
        json_format: Whether file logs should be JSON-lines formatted.
        max_bytes: Max log file size before rotation.
        backup_count: Number of rotated log backups to keep.

    Returns:
        The configured framework root logger.

    Raises:
        ValueError: If ``level`` is not a recognized logging level, or if
            ``log_to_file`` is True but ``log_dir`` is not provided.
    """
    global _configured

    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level!r}")

    if log_to_file and log_dir is None:
        raise ValueError("log_dir must be provided when log_to_file=True")

    logger = logging.getLogger(_FRAMEWORK_LOGGER_NAME)
    logger.setLevel(numeric_level)
    logger.propagate = False

    # Clear any previously attached handlers to keep this call idempotent.
    for existing_handler in list(logger.handlers):
        logger.removeHandler(existing_handler)
        existing_handler.close()

    logger.addHandler(_build_console_handler(use_rich=use_rich, level=numeric_level))

    if log_to_file:
        assert log_dir is not None  # narrowed above
        try:
            logger.addHandler(
                _build_file_handler(
                    log_dir=log_dir,
                    filename=log_filename,
                    level=numeric_level,
                    json_format=json_format,
                    max_bytes=max_bytes,
                    backup_count=backup_count,
                )
            )
        except OSError as exc:
            logger.warning("Could not set up file logging at %s: %s", log_dir, exc)

    _configured = True
    logger.debug("Logging configured (level=%s, log_to_file=%s).", level, log_to_file)
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger namespaced under the framework's root logger.

    Args:
        name: Optional submodule name (typically ``__name__``). If omitted,
            returns the framework root logger itself.

    Returns:
        A ``logging.Logger`` instance. If :func:`setup_logging` has not yet
        been called, a sensible default configuration (INFO, console-only)
        is applied automatically so that log output is never silently lost.
    """
    if not _configured:
        # Lazily apply a safe default so libraries importing this module
        # never emit "No handlers could be found" warnings.
        logging.getLogger(_FRAMEWORK_LOGGER_NAME).addHandler(logging.NullHandler())
        default_logger = logging.getLogger(_FRAMEWORK_LOGGER_NAME)
        if not any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)
            for h in default_logger.handlers
        ):
            default_logger.addHandler(_build_console_handler(use_rich=True, level=logging.INFO))
            default_logger.setLevel(logging.INFO)

    if name is None or name == _FRAMEWORK_LOGGER_NAME:
        return logging.getLogger(_FRAMEWORK_LOGGER_NAME)
    return logging.getLogger(f"{_FRAMEWORK_LOGGER_NAME}.{name}")
