"""Logging setup: console + timestamped file under results/logs/."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def setup_logging(results_dir: Path, level: str = "INFO") -> Path:
    logs_dir = Path(results_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper()))
    root.handlers.clear()

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(FORMAT, datefmt="%H:%M:%S"))
    root.addHandler(console)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(FORMAT))
    root.addHandler(file_handler)

    logging.getLogger(__name__).info("Logging to %s", log_file)
    return log_file
