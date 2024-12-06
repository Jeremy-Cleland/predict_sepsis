# src/utils.py

import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Removed: from .feature_engineering import preprocess_data  # No longer needed


def setup_logger(log_file="logs/sepsis_prediction.log"):
    """Set up the logger."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger("SepsisPredictionLogger")
    logger.setLevel(logging.DEBUG)

    # Avoid adding multiple handlers if the logger already has handlers
    if not logger.handlers:
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(log_file)
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.DEBUG)

        # Create formatters and add to handlers
        c_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        f_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger


def corr_matrix(df, figsize=(40, 40), vmax=0.3, logger=None):
    """Draw a correlation heatmap."""
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        mask=mask,
        cmap="Paired",
        vmax=vmax,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    plt.title("Correlation Matrix")
    if logger:
        logger.info("Saved correlation matrix plot.")
    plt.show()


def diagnostic_plots(df, variable, logger=None):
    """Draw histogram and QQ plot for a variable."""
    fig = plt.figure(figsize=(15, 4))
    ax = fig.add_subplot(121)
    df[variable].hist(bins=30, ax=ax)
    ax.set_title(f"Histogram of {variable}")

    ax = fig.add_subplot(122)
    stats.probplot(df[variable], dist="norm", plot=plt)
    ax.set_title(f"QQ Plot of {variable}")

    plt.tight_layout()
    plt.show()
    if logger:
        logger.info(f"Diagnostic plots for {variable} displayed.")


def try_gaussian(df, col, logger=None):
    """Apply various transformations to make the distribution more Gaussian."""
    if logger:
        logger.info(f"Processing column: {col}")
    diagnostic_plots(df, col, logger)

    # Yeo-Johnson Transformation
    df["col_yj"], _ = stats.yeojohnson(df[col])
    if logger:
        logger.info("After Yeo-Johnson Transformation")
    diagnostic_plots(df, "col_yj", logger)

    # Exponential Transformation
    df["col_1.5"] = df[col] ** (1 / 1.5)
    if logger:
        logger.info("After Exponentiation (1/1.5)")
    diagnostic_plots(df, "col_1.5", logger)

    # Square Root Transformation
    df["col_.5"] = df[col] ** 0.5
    if logger:
        logger.info("After Square Root Transformation")
    diagnostic_plots(df, "col_.5", logger)

    # Reciprocal Transformation
    df["col_rec"] = 1 / (df[col] + 1e-5)
    if logger:
        logger.info("After Reciprocal Transformation")
    diagnostic_plots(df, "col_rec", logger)

    # Log Transformation
    df["col_log"] = np.log(df[col] + 1)
    if logger:
        logger.info("After Log Transformation")
    diagnostic_plots(df, "col_log", logger)

    # Drop redundant columns
    df.drop(columns=["col_yj", "col_1.5", "col_.5", "col_rec", "col_log"], inplace=True)
    if logger:
        logger.info(f"Redundant transformation columns dropped for {col}.")
