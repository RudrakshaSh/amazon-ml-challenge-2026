"""Logging setup plus timing and memory helpers.

All pipeline loggers live under the ``ber`` namespace. Every stage also writes
its log to ``artifacts/<run>/<stage>/log.txt`` (see :mod:`ber.artifacts`).
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)-5s %(name)s | %(message)s"
DATEFMT = "%H:%M:%S"


def setup_logging(level: str = "INFO") -> None:
    """Configure the ``ber`` logger once (idempotent)."""
    root = logging.getLogger("ber")
    root.setLevel(level.upper())
    if not any(getattr(h, "_ber_console", False) for h in root.handlers):
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter(FORMAT, DATEFMT))
        h._ber_console = True  # type: ignore[attr-defined]
        root.addHandler(h)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("ber") else f"ber.{name}")


def add_file_handler(path: Path) -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(path, mode="a", encoding="utf-8")
    h.setFormatter(logging.Formatter(FORMAT, "%Y-%m-%d %H:%M:%S"))
    logging.getLogger("ber").addHandler(h)
    return h


def remove_handler(h: logging.Handler) -> None:
    logging.getLogger("ber").removeHandler(h)
    h.close()


def peak_rss_gb() -> float:
    """Peak resident memory of this process so far, in GB."""
    try:
        import resource  # Unix only
    except ImportError:  # Windows: psutil exposes the peak working set
        import psutil
        info = psutil.Process().memory_info()
        return getattr(info, "peak_wset", info.rss) / 1e9
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss * 1024 / 1e9  # bytes vs KiB


@contextmanager
def timed(logger: logging.Logger, what: str):
    """Log the duration and peak memory of a block."""
    t = time.perf_counter()
    logger.info("%s ...", what)
    yield
    logger.info("%s done in %.1fs (peak RSS %.2f GB)", what, time.perf_counter() - t,
                peak_rss_gb())
