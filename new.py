import json
import logging  # Required to use logging.DEBUG
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler

from src.logger_config import setup_logger

# Initialize logger with custom log level ("DEBUG") and JSON formatting disabled
logger = setup_logger(
    name="sepsis_prediction.preprocessing",
    log_file="logs/preprocessing.log",
    level=logging.DEBUG,  # Custom log level
    use_json=False,  # Disable JSON formatting
)


def drop_redundant_columns(df):
    """Drop specified redundant columns from the dataset."""
    columns_drop = {
        "Unnamed: 0",  # Index column
        # We'll keep SBP and DBP as they're used for derived features
        "EtCO2",  # High missing rate
        # Blood gas and chemistry redundancies
        "BaseExcess",
        "HCO3",
        "pH",
        "PaCO2",
        # High missing rate lab values
        "AST",
        "Alkalinephos",
        "Bilirubin_direct",
        "Bilirubin_total",
        "Lactate",
        "TroponinI",
        "SaO2",
        "FiO2",
    }

    # Only drop columns that exist
    columns_to_drop = [col for col in columns_drop if col in df.columns]
    df = df.drop(columns=columns_to_drop, errors="ignore")
    return df


def engineer_vital_features(df):
    """Engineer features from vital signs."""
    df = df.copy()

    # Calculate shock index if components exist
    if "HR" in df.columns and "SBP" in df.columns:
        df["shock_index"] = df["HR"] / df["SBP"].clip(lower=1)

    # Calculate pulse pressure if components exist
    if "SBP" in df.columns and "DBP" in df.columns:
        df["pulse_pressure"] = df["SBP"] - df["DBP"]

    # Group by patient for temporal features
    patient_groups = df.groupby("Patient_ID")

    # Core vital signs that are typically well-measured
    vital_signs = ["HR", "O2Sat", "Temp", "MAP", "Resp"]
    vital_signs = [vs for vs in vital_signs if vs in df.columns]

    # Create temporal features for vitals
    for vital in vital_signs:
        # Rate of change
        df[f"{vital}_rate"] = patient_groups[vital].transform(
            lambda x: x.diff() / df["Hour"].diff()
        )

        # Rolling mean and std (6-hour window)
        df[f"{vital}_rolling_mean_6h"] = patient_groups[vital].transform(
            lambda x: x.rolling(6, min_periods=1).mean()
        )
        df[f"{vital}_rolling_std_6h"] = patient_groups[vital].transform(
            lambda x: x.rolling(6, min_periods=1).std()
        )

    return df


def handle_missing_values(df):
    """Handle missing values with appropriate strategies."""
    df = df.copy()

    # Forward fill vital signs within patient groups (up to 4 hours)
    vital_signs = ["HR", "O2Sat", "Temp", "SBP", "DBP", "MAP", "Resp"]
    existing_vitals = [col for col in vital_signs if col in df.columns]

    if existing_vitals:
        df[existing_vitals] = df.groupby("Patient_ID")[existing_vitals].transform(
            lambda x: x.ffill(limit=4)
        )

    # Create missing indicators for important lab values
    lab_values = ["BUN", "Creatinine", "Glucose", "WBC", "Platelets"]

    for col in lab_values:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isnull().astype(int)

    # Use iterative imputation for remaining numeric values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    imputer = IterativeImputer(
        random_state=42, max_iter=10, initial_strategy="median", skip_complete=True
    )

    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    return df


def calculate_severity_scores(df):
    """Calculate clinical severity scores."""
    df = df.copy()

    # SIRS criteria (Systemic Inflammatory Response Syndrome)
    if "Temp" in df.columns:
        df["sirs_temp"] = ((df["Temp"] < 36) | (df["Temp"] > 38)).astype(int)
    if "HR" in df.columns:
        df["sirs_hr"] = (df["HR"] > 90).astype(int)
    if "Resp" in df.columns:
        df["sirs_resp"] = (df["Resp"] > 20).astype(int)
    if "WBC" in df.columns:
        df["sirs_wbc"] = ((df["WBC"] < 4) | (df["WBC"] > 12)).astype(int)

    # Calculate total SIRS score if all components exist
    sirs_components = ["sirs_temp", "sirs_hr", "sirs_resp", "sirs_wbc"]
    if all(comp in df.columns for comp in sirs_components):
        df["sirs_score"] = df[sirs_components].sum(axis=1)

    return df


def fit_preprocessor(df_train):
    """
    Fit preprocessor on training data and return fitted parameters.

    Args:
        df_train (pd.DataFrame): Training dataset

    Returns:
        dict: Dictionary containing fitted preprocessing parameters
    """
    fitted_params = {
        "scaler": RobustScaler().fit(df_train.select_dtypes(include=[np.number])),
        "transformations": {},
    }

    # Fit transformations on training data
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols.drop(
        [
            "Patient_ID",
            "Hour",
            "SepsisLabel",
            "sirs_temp",
            "sirs_hr",
            "sirs_resp",
            "sirs_wbc",
            "sirs_score",
        ]
        + [col for col in df_train.columns if col.endswith("_missing")],
        errors="ignore",
    )

    for col in numeric_cols:
        if df_train[col].nunique() > 2:
            # Calculate best transformation using training data
            temp_df = pd.DataFrame()
            temp_df[col] = df_train[col]

            # Store transformation parameters
            fitted_params["transformations"][col] = {
                "yj_lambda": stats.yeojohnson_normmax(df_train[col])
            }

    return fitted_params


def preprocess_data(df, fitted_params=None, is_training=False):
    """
    Preprocess data using either fitted parameters or by fitting new ones.

    Args:
        df (pd.DataFrame): Dataset to preprocess
        fitted_params (dict): Dictionary of fitted preprocessing parameters
        is_training (bool): Whether this is the training dataset

    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    logger.info("Starting preprocessing pipeline")
    df = df.copy()

    # Step 1-4: These steps don't require fitting
    df = drop_redundant_columns(df)
    df = handle_missing_values(df)
    df = engineer_vital_features(df)
    df = calculate_severity_scores(df)

    # Step 5: Apply transformations
    if is_training:
        fitted_params = fit_preprocessor(df)

    if fitted_params is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols.drop(
            [
                "Patient_ID",
                "Hour",
                "SepsisLabel",
                "sirs_temp",
                "sirs_hr",
                "sirs_resp",
                "sirs_wbc",
                "sirs_score",
            ]
            + [col for col in df.columns if col.endswith("_missing")],
            errors="ignore",
        )

        for col, params in fitted_params["transformations"].items():
            if col in df.columns:
                # Apply the same transformation using stored parameters
                df[col], _ = stats.yeojohnson(df[col], lmbda=params["yj_lambda"])

        # Apply scaling using fitted scaler
        if "scaler" in fitted_params:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = numeric_cols.drop(
                ["Patient_ID", "Hour", "SepsisLabel"], errors="ignore"
            )
            if len(numeric_cols) > 0:
                df[numeric_cols] = fitted_params["scaler"].transform(df[numeric_cols])

    # Step 6: One-hot encoding
    categorical_cols = ["Gender", "Unit1", "Unit2"]
    existing_cats = [col for col in categorical_cols if col in df.columns]

    for col in existing_cats:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        df.drop(columns=[col], inplace=True)

    return df, fitted_params if is_training else df
