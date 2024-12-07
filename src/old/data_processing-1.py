# src/data_processing.py

import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file with error handling and validation.

    Parameters:
    -----------
    filepath : str
        Path to the CSV file

    Returns:
    --------
    pd.DataFrame
        Loaded and validated DataFrame

    Raises:
    -------
    FileNotFoundError
        If the specified file doesn't exist
    ValueError
        If the loaded data doesn't meet expected format
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at: {filepath}")

    # Load the data
    df = pd.read_csv(filepath)

    # Validate required columns
    required_columns = ["Patient_ID", "SepsisLabel"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Convert Patient_ID to string to ensure consistent handling
    df["Patient_ID"] = df["Patient_ID"].astype(str)

    # Basic data validation
    if df["SepsisLabel"].nunique() > 2:
        raise ValueError("SepsisLabel contains more than two unique values")

    logging.info(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    return df


def split_data(
    combined_df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    random_state: int = 42,
    stratify_by_sepsis: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the combined data into training, validation, and testing datasets while keeping Patient_IDs together
    and optionally stratifying by sepsis prevalence.

    Parameters:
    -----------
    combined_df : pd.DataFrame
        The input DataFrame containing patient data
    train_size : float, default=0.7
        The proportion of data to include in the training set
    val_size : float, default=0.15
        The proportion of data to include in the validation set
    random_state : int, default=42
        Random state for reproducibility
    stratify_by_sepsis : bool, default=True
        Whether to maintain similar sepsis prevalence across splits

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Training, validation, and testing datasets

    Raises:
    -------
    ValueError
        If input parameters are invalid or if data doesn't meet requirements
    """
    # Input validation
    if not 0 < train_size + val_size < 1:
        raise ValueError("train_size + val_size must be between 0 and 1")

    if not isinstance(combined_df, pd.DataFrame):
        raise ValueError("combined_df must be a pandas DataFrame")

    # Get unique Patient_IDs and their characteristics
    patient_stats = (
        combined_df.groupby("Patient_ID")
        .agg(
            {
                "SepsisLabel": "max",  # 1 if patient ever had sepsis
                "Hour": "count",  # number of measurements per patient
            }
        )
        .reset_index()
    )

    # Set random seed for reproducibility
    np.random.seed(random_state)

    if stratify_by_sepsis:
        # Split separately for sepsis and non-sepsis patients to maintain distribution
        sepsis_patients = patient_stats[patient_stats["SepsisLabel"] == 1]["Patient_ID"]
        non_sepsis_patients = patient_stats[patient_stats["SepsisLabel"] == 0][
            "Patient_ID"
        ]

        # Function to split patient IDs while maintaining proportions
        def split_patient_ids(patient_ids):
            n_train = int(len(patient_ids) * train_size)
            n_val = int(len(patient_ids) * val_size)
            shuffled = np.random.permutation(patient_ids)
            return (
                shuffled[:n_train],
                shuffled[n_train : n_train + n_val],
                shuffled[n_train + n_val :],
            )

        # Split both groups
        train_sepsis, val_sepsis, test_sepsis = split_patient_ids(sepsis_patients)
        train_non_sepsis, val_non_sepsis, test_non_sepsis = split_patient_ids(
            non_sepsis_patients
        )

        # Combine splits
        train_patients = np.concatenate([train_sepsis, train_non_sepsis])
        val_patients = np.concatenate([val_sepsis, val_non_sepsis])
        test_patients = np.concatenate([test_sepsis, test_non_sepsis])
    else:
        # Simple random split without stratification
        shuffled_patients = np.random.permutation(patient_stats["Patient_ID"])
        n_train = int(len(shuffled_patients) * train_size)
        n_val = int(len(shuffled_patients) * val_size)

        train_patients = shuffled_patients[:n_train]
        val_patients = shuffled_patients[n_train : n_train + n_val]
        test_patients = shuffled_patients[n_train + n_val :]

    # Create the splits
    df_train = combined_df[combined_df["Patient_ID"].isin(train_patients)].copy()
    df_val = combined_df[combined_df["Patient_ID"].isin(val_patients)].copy()
    df_test = combined_df[combined_df["Patient_ID"].isin(test_patients)].copy()

    # Verify no patient overlap
    assert (
        len(set(train_patients) & set(val_patients)) == 0
    ), "Patient overlap between train and val"
    assert (
        len(set(train_patients) & set(test_patients)) == 0
    ), "Patient overlap between train and test"
    assert (
        len(set(val_patients) & set(test_patients)) == 0
    ), "Patient overlap between val and test"

    # Log split information
    logging.info("\nData Split Summary:")
    logging.info("-" * 50)
    logging.info(
        f"Training set:   {len(df_train)} rows, {len(train_patients)} unique patients"
    )
    logging.info(
        f"Validation set: {len(df_val)} rows, {len(val_patients)} unique patients"
    )
    logging.info(
        f"Testing set:    {len(df_test)} rows, {len(test_patients)} unique patients"
    )

    # Log sepsis distribution
    for name, df in [
        ("Training", df_train),
        ("Validation", df_val),
        ("Testing", df_test),
    ]:
        sepsis_rate = df.groupby("Patient_ID")["SepsisLabel"].max().mean()
        logging.info(f"{name} set sepsis rate: {sepsis_rate:.1%}")

    # Save the split datasets
    os.makedirs("data/processed", exist_ok=True)
    df_train.to_csv("data/processed/train_data.csv", index=False)
    df_val.to_csv("data/processed/val_data.csv", index=False)
    df_test.to_csv("data/processed/test_data.csv", index=False)

    return df_train, df_val, df_test


def load_processed_data(
    train_path: str, val_path: str, test_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load processed training, validation, and testing data with validation.

    Parameters:
    -----------
    train_path : str
        Path to training data CSV
    val_path : str
        Path to validation data CSV
    test_path : str
        Path to test data CSV

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Training, validation, and testing DataFrames

    Raises:
    -------
    FileNotFoundError
        If any of the specified files don't exist
    ValueError
        If the loaded data doesn't meet expected format
    """
    # Check file existence
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found at: {path}")

    # Load datasets
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    # Validate columns match across datasets
    if not (set(df_train.columns) == set(df_val.columns) == set(df_test.columns)):
        raise ValueError("Column mismatch between datasets")

    # Convert Patient_ID to string in all datasets
    for df in [df_train, df_val, df_test]:
        df["Patient_ID"] = df["Patient_ID"].astype(str)

    # Verify no patient overlap
    train_patients = set(df_train["Patient_ID"])
    val_patients = set(df_val["Patient_ID"])
    test_patients = set(df_test["Patient_ID"])

    if train_patients & val_patients:
        raise ValueError("Patient overlap between train and validation sets")
    if train_patients & test_patients:
        raise ValueError("Patient overlap between train and test sets")
    if val_patients & test_patients:
        raise ValueError("Patient overlap between validation and test sets")

    return df_train, df_val, df_test


def get_data_ready(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the dataframe into a format compatible with the model.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame to process

    Returns:
    --------
    pd.DataFrame
        Processed DataFrame ready for model input

    Notes:
    ------
    This is a wrapper function that calls the appropriate preprocessing
    functions from feature_engineering module.
    """
    from .feature_engineering import preprocess_data

    # Make a copy to avoid modifying the original
    df = df.copy()

    # Apply preprocessing
    df = preprocess_data(df)

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate the dataset format and content.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate

    Raises:
    -------
    ValueError
        If the dataset doesn't meet the expected format
    """
    # Check required columns
    required_columns = ["Patient_ID", "Hour", "SepsisLabel"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Validate data types
    if not pd.api.types.is_numeric_dtype(df["Hour"]):
        raise ValueError("Hour column must be numeric")
    if not pd.api.types.is_numeric_dtype(df["SepsisLabel"]):
        raise ValueError("SepsisLabel column must be numeric")

    # Validate value ranges
    if df["Hour"].min() < 0:
        raise ValueError("Hour column contains negative values")
    if not set(df["SepsisLabel"].unique()).issubset({0, 1}):
        raise ValueError("SepsisLabel must contain only 0 and 1")

    # Check for duplicates
    duplicates = df.groupby(["Patient_ID", "Hour"]).size()
    if (duplicates > 1).any():
        raise ValueError("Found duplicate time points for some patients")
