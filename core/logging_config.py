"""Structured logging configuration for Bimify."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger


class JSONFormatter:
    """JSON formatter for structured logging."""
    
    def __call__(self, record: dict[str, Any]) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "message": record["message"],
            "module": record.get("module", ""),
            "function": record.get("function", ""),
            "line": record.get("line", 0),
        }
        
        # Add exception info if present
        if "exception" in record:
            log_data["exception"] = {
                "type": record["exception"].type.__name__ if record["exception"].type else None,
                "value": str(record["exception"].value) if record["exception"].value else None,
            }
        
        # Add extra fields
        if "extra" in record:
            log_data.update(record["extra"])
        
        return json.dumps(log_data, ensure_ascii=False)


def setup_logging(
    *,
    level: str = "INFO",
    json_format: bool = False,
    log_file: Path | None = None,
) -> None:
    """Configure logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_format: Whether to use JSON formatting (useful for production).
        log_file: Optional path to log file. If None, logs only to stderr.
    """
    # Remove default handler
    logger.remove()
    
    # Determine format
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=formatter if not json_format else formatter,
        level=level,
        colorize=not json_format,
    )
    
    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format=formatter if not json_format else formatter,
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
        )


def get_logger(name: str | None = None) -> Any:
    """Get a logger instance.
    
    Args:
        name: Optional logger name. If None, returns the default logger.
    
    Returns:
        Logger instance.
    """
    if name:
        return logger.bind(name=name)
    return logger

