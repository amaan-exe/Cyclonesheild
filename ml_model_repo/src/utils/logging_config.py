"""
Cyclone Horizon — Structured Logging
JSON-structured logs with run ID, stage, storm ID for full traceability.
"""

import logging
import json
import sys
import uuid
from datetime import datetime, timezone
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def __init__(self, run_id: str):
        super().__init__()
        self.run_id = run_id

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "run_id": self.run_id,
            "stage": getattr(record, "stage", None),
            "storm_id": getattr(record, "storm_id", None),
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        # Remove None values for cleaner output
        log_entry = {k: v for k, v in log_entry.items() if v is not None}
        
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, default=str)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    run_id: Optional[str] = None,
) -> str:
    """
    Configure structured JSON logging for the pipeline.
    
    Parameters
    ----------
    level : logging level
    log_file : optional path to write logs
    run_id : unique identifier for this pipeline run (auto-generated if None)
    
    Returns
    -------
    run_id : the run ID used
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    
    root_logger = logging.getLogger("cyclone_horizon")
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    
    formatter = JSONFormatter(run_id)
    
    # Console handler (human-readable for development)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (structured JSON for pipeline traceability)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return run_id


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the cyclone_horizon namespace."""
    return logging.getLogger(f"cyclone_horizon.{name}")
