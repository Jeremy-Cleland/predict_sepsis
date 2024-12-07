# src/utils.py

import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


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
