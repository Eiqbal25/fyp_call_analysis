"""
utils/logger.py
===============
Centralised logging configuration for the FYP1 Call Analysis System.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
"""

import os
import logging
import sys
from config import OUTPUTS_DIR


def get_logger(name: str,
               log_file: str = None,
               level: int = logging.INFO) -> logging.Logger:
    """
    Create or retrieve a named logger with console + file handlers.

    Parameters
    ----------
    name     : logger name (typically __name__)
    log_file : path to log file (default: outputs/pipeline.log)
    level    : logging level

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger   # Already configured

    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file is None:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        log_file = os.path.join(OUTPUTS_DIR, "pipeline.log")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
