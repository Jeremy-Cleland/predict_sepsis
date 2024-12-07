# # src/utils.py

# import json
# import logging
# import os

# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# from scipy import stats
# from sklearn.preprocessing import StandardScaler

# # Removed: from .feature_engineering import preprocess_data  # No longer needed


# def setup_logger(log_file="logs/sepsis_prediction.log"):
#     """Set up the logger with proper configuration to avoid double logging."""
#     os.makedirs(os.path.dirname(log_file), exist_ok=True)

#     # Get the logger
#     logger = logging.getLogger("SepsisPredictionLogger")

#     # Clear any existing handlers
#     if logger.handlers:
#         logger.handlers.clear()

#     # Prevent the logger from propagating messages to the root logger
#     logger.propagate = False

#     # Set the logging level
#     logger.setLevel(logging.INFO)

#     # Create handlers
#     c_handler = logging.StreamHandler()
#     f_handler = logging.FileHandler(log_file)

#     # Set levels
#     c_handler.setLevel(logging.INFO)
#     f_handler.setLevel(logging.DEBUG)

#     # Create formatters
#     formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
#     c_handler.setFormatter(formatter)
#     f_handler.setFormatter(formatter)

#     # Add handlers to the logger
#     logger.addHandler(c_handler)
#     logger.addHandler(f_handler)

#     return logger


# def corr_matrix(df, figsize=(40, 40), vmax=0.3, logger=None):
#     """Draw a correlation heatmap."""
#     corr = df.corr()
#     mask = np.triu(np.ones_like(corr, dtype=bool))
#     plt.figure(figsize=figsize)
#     sns.heatmap(
#         corr,
#         mask=mask,
#         cmap="Paired",
#         vmax=vmax,
#         center=0,
#         square=True,
#         linewidths=0.5,
#         cbar_kws={"shrink": 0.5},
#     )
#     plt.title("Correlation Matrix")
#     if logger:
#         logger.info("Saved correlation matrix plot.")
#     plt.show()


# def diagnostic_plots(df, variable, logger=None):
#     """Draw histogram and QQ plot for a variable."""
#     fig = plt.figure(figsize=(15, 4))
#     ax = fig.add_subplot(121)
#     df[variable].hist(bins=30, ax=ax)
#     ax.set_title(f"Histogram of {variable}")

#     ax = fig.add_subplot(122)
#     stats.probplot(df[variable], dist="norm", plot=plt)
#     ax.set_title(f"QQ Plot of {variable}")

#     plt.tight_layout()
#     plt.show()
#     if logger:
#         logger.info(f"Diagnostic plots for {variable} displayed.")


# def try_gaussian(df, col, logger=None):
#     """Apply various transformations to make the distribution more Gaussian."""
#     if logger:
#         logger.info(f"Processing column: {col}")
#     diagnostic_plots(df, col, logger)

#     # Yeo-Johnson Transformation
#     df["col_yj"], _ = stats.yeojohnson(df[col])
#     if logger:
#         logger.info("After Yeo-Johnson Transformation")
#     diagnostic_plots(df, "col_yj", logger)

#     # Exponential Transformation
#     df["col_1.5"] = df[col] ** (1 / 1.5)
#     if logger:
#         logger.info("After Exponentiation (1/1.5)")
#     diagnostic_plots(df, "col_1.5", logger)

#     # Square Root Transformation
#     df["col_.5"] = df[col] ** 0.5
#     if logger:
#         logger.info("After Square Root Transformation")
#     diagnostic_plots(df, "col_.5", logger)

#     # Reciprocal Transformation
#     df["col_rec"] = 1 / (df[col] + 1e-5)
#     if logger:
#         logger.info("After Reciprocal Transformation")
#     diagnostic_plots(df, "col_rec", logger)

#     # Log Transformation
#     df["col_log"] = np.log(df[col] + 1)
#     if logger:
#         logger.info("After Log Transformation")
#     diagnostic_plots(df, "col_log", logger)

#     # Drop redundant columns
#     df.drop(columns=["col_yj", "col_1.5", "col_.5", "col_rec", "col_log"], inplace=True)
#     if logger:
#         logger.info(f"Redundant transformation columns dropped for {col}.")


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
    """Set up the logger with proper configuration to avoid double logging."""
    os.makedirs(
        os.path.dirname(log_file), exist_ok=True
    )  # Ensure the log directory exists

    # Get the logger
    logger = logging.getLogger("SepsisPredictionLogger")

    # Clear any existing handlers to avoid duplicate logs
    if logger.handlers:
        logger.handlers.clear()

    # Prevent the logger from propagating messages to the root logger
    logger.propagate = False

    # Set the logging level for the logger
    logger.setLevel(logging.INFO)

    # Create console and file handlers
    c_handler = logging.StreamHandler()  # Logs to the console
    f_handler = logging.FileHandler(log_file)  # Logs to a file

    # Set logging levels for handlers
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.DEBUG)

    # Define log message format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(formatter)  # Format for console handler
    f_handler.setFormatter(formatter)  # Format for file handler

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger  # Return the configured logger


def corr_matrix(df, figsize=(40, 40), vmax=0.3, logger=None):
    """Draw a correlation heatmap.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        figsize (tuple): Size of the plot.
        vmax (float): Maximum value for the color scale.
        logger (logging.Logger, optional): Logger for logging information.
    """
    corr = df.corr()  # Compute the correlation matrix
    mask = np.triu(
        np.ones_like(corr, dtype=bool)
    )  # Create a mask for the upper triangle
    plt.figure(figsize=figsize)  # Set figure size

    # Plot the heatmap
    sns.heatmap(
        corr,
        mask=mask,
        cmap="Paired",  # Color map for heatmap
        vmax=vmax,  # Max value for color bar
        center=0,  # Center the colormap at 0
        square=True,  # Make cells square
        linewidths=0.5,  # Add lines between cells
        cbar_kws={"shrink": 0.5},  # Shrink the color bar
    )
    plt.title("Correlation Matrix")  # Add title to the plot

    if logger:
        logger.info("Saved correlation matrix plot.")  # Log plot creation
    plt.show()  # Display the plot


def diagnostic_plots(df, variable, logger=None):
    """Draw histogram and QQ plot for a variable.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        variable (str): Column name of the variable to analyze.
        logger (logging.Logger, optional): Logger for logging information.
    """
    fig = plt.figure(figsize=(15, 4))  # Set figure size

    # Create histogram plot
    ax = fig.add_subplot(121)  # First subplot
    df[variable].hist(bins=30, ax=ax)  # Plot histogram
    ax.set_title(f"Histogram of {variable}")  # Add title

    # Create QQ plot
    ax = fig.add_subplot(122)  # Second subplot
    stats.probplot(df[variable], dist="norm", plot=plt)  # QQ plot for normality
    ax.set_title(f"QQ Plot of {variable}")  # Add title

    plt.tight_layout()  # Adjust layout
    plt.show()  # Display the plots

    if logger:
        logger.info(f"Diagnostic plots for {variable} displayed.")  # Log plot creation


def try_gaussian(df, col, logger=None):
    """Apply various transformations to make the distribution more Gaussian.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        col (str): Column name to transform.
        logger (logging.Logger, optional): Logger for logging information.
    """
    if logger:
        logger.info(f"Processing column: {col}")  # Log start of processing

    # Initial diagnostic plots
    diagnostic_plots(df, col, logger)

    # Apply Yeo-Johnson transformation
    df["col_yj"], _ = stats.yeojohnson(df[col])
    if logger:
        logger.info("After Yeo-Johnson Transformation")
    diagnostic_plots(df, "col_yj", logger)

    # Apply exponential transformation
    df["col_1.5"] = df[col] ** (1 / 1.5)
    if logger:
        logger.info("After Exponentiation (1/1.5)")
    diagnostic_plots(df, "col_1.5", logger)

    # Apply square root transformation
    df["col_.5"] = df[col] ** 0.5
    if logger:
        logger.info("After Square Root Transformation")
    diagnostic_plots(df, "col_.5", logger)

    # Apply reciprocal transformation
    df["col_rec"] = 1 / (df[col] + 1e-5)  # Add small value to avoid division by zero
    if logger:
        logger.info("After Reciprocal Transformation")
    diagnostic_plots(df, "col_rec", logger)

    # Apply log transformation
    df["col_log"] = np.log(df[col] + 1)  # Add 1 to avoid log(0)
    if logger:
        logger.info("After Log Transformation")
    diagnostic_plots(df, "col_log", logger)

    # Drop redundant columns after visualization
    df.drop(columns=["col_yj", "col_1.5", "col_.5", "col_rec", "col_log"], inplace=True)
    if logger:
        logger.info(f"Redundant transformation columns dropped for {col}.")
