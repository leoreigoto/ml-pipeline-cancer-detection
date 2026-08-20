"""
This module provides a utility function for setting up a logger with file handling.
It creates a log file in a structured directory based on a unique identifier and
the current date, facilitating organized logging for applications.
"""

from datetime import datetime
import logging
from pathlib import Path
import sys


def get_logger(logger_save_id: str, logger_name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Creates or retrieves a configured logger with file and console handlers.

    The logger writes to a structured directory: 'logs/{save_id}/{YYYY-MM}/{save_id}_{date}.log'.
    It includes a check to prevent adding duplicate handlers if the logger is requested multiple times.

    Args:
        logger_save_id (str): Unique identifier for directory/file organization.
        logger_name (str): The name of the logger instance.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: A configured Logger object.
    """

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # If this logger already has handlers, it means it was initialized before.
    # Return it instead of creating a new one to prevent duplicate lines in logs.
    if logger.hasHandlers():
        return logger

    # Log directory path: logs / save_id / YYYY-MM
    timestamp = datetime.now()
    log_directory = Path('logs') / logger_save_id / timestamp.strftime("%Y-%m")
    log_directory.mkdir(parents=True, exist_ok=True)

    log_filename = f"{logger_save_id}_{timestamp.strftime('%Y-%m-%d')}.log"
    log_file_path = log_directory / log_filename

    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(level)

    # This allows to see logs in the terminal/Airflow UI.
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Prevent propagation to root/parent logger to avoid double logging.
    logger.propagate = False

    return logger




