# src/logging_utils.py

import time
import functools
import psutil
from typing import Callable
import logging
from contextlib import contextmanager


def log_phase(logger: logging.Logger) -> Callable:
    """
    Decorator to log the start, end, and duration of each processing phase.
    Compatible with the existing sepsis prediction logger.

    Usage:
    @log_phase(logger)
    def your_function():
        ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            phase_name = func.__name__.replace("_", " ").title()

            # Start phase logging
            logger.info("=" * 80)
            logger.info(f"Starting Phase: {phase_name}")
            logger.info("-" * 80)

            # Log initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            logger.info(f"Initial Memory Usage: {initial_memory:.2f} MB")

            # Execute phase
            start_time = time.time()
            try:
                result = func(*args, **kwargs)

                # Log successful completion
                duration = time.time() - start_time
                final_memory = process.memory_info().rss / 1024 / 1024
                memory_diff = final_memory - initial_memory

                logger.info("-" * 80)
                logger.info(f"Phase Complete: {phase_name}")
                logger.info(f"Duration: {duration:.2f} seconds")
                logger.info(f"Final Memory Usage: {final_memory:.2f} MB")
                logger.info(f"Memory Change: {memory_diff:+.2f} MB")
                logger.info("=" * 80)

                return result

            except Exception as e:
                # Log failure
                logger.error(f"Phase Failed: {phase_name}")
                logger.error(f"Error: {str(e)}", exc_info=True)
                logger.error("=" * 80)
                raise

        return wrapper

    return decorator


@contextmanager
def log_step(logger: logging.Logger, step_name: str):
    """
    Context manager for logging individual steps within a phase.

    Usage:
    with log_step(logger, "Data Preprocessing"):
        preprocess_data()
    """
    try:
        logger.info(f"Starting step: {step_name}")
        start_time = time.time()
        yield
        duration = time.time() - start_time
        logger.info(f"Completed step: {step_name} (Duration: {duration:.2f}s)")
    except Exception as e:
        logger.error(f"Error in step {step_name}: {str(e)}")
        raise


def log_memory(logger: logging.Logger, label: str = "Current"):
    """Log current memory usage."""
    process = psutil.Process()
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
    logger.info(f"{label} Memory Usage: {memory_usage:.2f} MB")


def log_dataframe_info(logger: logging.Logger, df, name: str):
    """
    Log comprehensive information about a DataFrame.

    Parameters:
    -----------
    logger : logging.Logger
        The logger instance to use
    df : pandas.DataFrame
        The DataFrame to analyze
    name : str
        A descriptive name for the DataFrame
    """
    logger.info(f"\n{name} DataFrame Information:")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Memory usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB")
    if "SepsisLabel" in df.columns:
        class_dist = df["SepsisLabel"].value_counts()
        logger.info(f"Class distribution: {dict(class_dist)}")

    # Log missing value information
    missing_info = df.isnull().sum()
    if missing_info.any():
        missing_cols = missing_info[missing_info > 0]
        logger.info("\nColumns with missing values:")
        for col, count in missing_cols.items():
            percentage = (count / len(df)) * 100
            logger.info(f"{col}: {count} missing values ({percentage:.2f}%)")
