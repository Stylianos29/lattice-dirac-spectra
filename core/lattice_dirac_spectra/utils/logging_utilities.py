"""
Logging utilities
=================

A small, configurable logging helper for both CLI scripts and Jupyter
notebooks, with separate control of console and file output. Adapted from the
logging system of ``qpb-data-tools``.
"""

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Optional, Union


class LogLevel(Enum):
    """Standard log levels with descriptive names."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


def setup_logger(
    name: str = "lattice_dirac_spectra",
    *,
    console: bool = True,
    console_level: LogLevel = LogLevel.INFO,
    log_file: Optional[Union[str, Path]] = None,
    file_level: LogLevel = LogLevel.DEBUG,
    console_format: str = "%(levelname)s: %(message)s",
    file_format: str = "%(asctime)s - %(levelname)s - %(message)s",
    clear_existing_handlers: bool = True,
) -> logging.Logger:
    """Create and configure a logger.

    Parameters
    ----------
    name : str
        Logger name.
    console : bool
        If True, attach a console (stderr) handler.
    console_level : LogLevel
        Level for the console handler.
    log_file : str or Path, optional
        If given, attach a file handler writing to this path (parent dirs are
        created as needed).
    file_level : LogLevel
        Level for the file handler.
    console_format, file_format : str
        Format strings for the respective handlers.
    clear_existing_handlers : bool
        If True, remove any pre-existing handlers on this logger first (useful
        when re-running notebook cells).

    Returns
    -------
    logging.Logger
        The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if clear_existing_handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    if console:
        ch = logging.StreamHandler(stream=sys.stderr)
        ch.setLevel(console_level.value)
        ch.setFormatter(logging.Formatter(console_format))
        logger.addHandler(ch)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setLevel(file_level.value)
        fh.setFormatter(logging.Formatter(file_format))
        logger.addHandler(fh)

    return logger
