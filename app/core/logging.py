"""Structured logging setup for the FastAPI Braille Engine."""

import logging
import sys


def setup_logger(name: str = "braille_engine") -> logging.Logger:
    _logger = logging.getLogger(name)
    if not _logger.handlers:
        _logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        _logger.addHandler(handler)
        _logger.propagate = False
    return _logger


logger = setup_logger()
