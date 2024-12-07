# src/feature_engineering.py

import logging

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler, StandardScaler

# from .utils import (
#     corr_matrix,
#     diagnostic_plots,
#     try_gaussian,
# )  # Ensure utils are imported correctly


def drop_columns(df):
    """Drop specified redundant columns except Unit1 and Unit2."""
    columns_drop = {
        "Unnamed: 0",
        "SBP",
        "DBP",
        "EtCO2",
        "BaseExcess",
        "HCO3",
        "pH",
        "PaCO2",
        "Alkalinephos",
        "Calcium",
        "Magnesium",
        "Phosphate",
        "Potassium",
        "PTT",
        "Fibrinogen",
    }
    df = df.drop(columns=columns_drop)
    return df


# def fill_missing_values(df):
#     """Impute missing values using backfill and forward fill."""
#     grouped_by_patient = df.groupby(
#         "Patient_ID", group_keys=False
#     )  # Added group_keys=False
#     df = grouped_by_patient.apply(lambda x: x.bfill().ffill())
#     return df


def fill_missing_values(df):
    """Impute missing values using IterativeImputer."""
    # Create a copy of the dataframe
    df_copy = df.copy()

    # Separate Patient_ID and categorical columns
    id_column = df_copy["Patient_ID"]
    categorical_columns = ["Gender", "Unit1", "Unit2"]
    categorical_data = df_copy[categorical_columns]

    # Get numerical columns for imputation
    numerical_columns = df_copy.select_dtypes(include=[np.number]).columns
    numerical_data = df_copy[numerical_columns]

    # Initialize and fit the IterativeImputer
    imputer = IterativeImputer(
        random_state=42, max_iter=15, initial_strategy="mean", skip_complete=True
    )

    # Perform imputation on numerical columns
    imputed_numerical = pd.DataFrame(
        imputer.fit_transform(numerical_data),
        columns=numerical_columns,
        index=df_copy.index,
    )

    # Combine the imputed numerical data with categorical data
    df = pd.concat([id_column, categorical_data, imputed_numerical], axis=1)

    return df


def drop_null_columns(df):
    """Drop specified columns if they exist."""
    null_col = [
        "TroponinI",
        "Bilirubin_direct",
        "AST",
        "Bilirubin_total",
        "Lactate",
        "SaO2",
        "FiO2",
        "Patient_ID",
    ]

    # Identify columns that exist in the DataFrame
    existing_cols = [col for col in null_col if col in df.columns]

    if existing_cols:
        df = df.drop(columns=existing_cols)
        logging.info(f"Dropped columns: {existing_cols}")
    else:
        logging.warning(f"No columns to drop from drop_null_columns: {null_col}")

    return df


def one_hot_encode_gender(df):
    """One-hot encode the Gender column, ensuring no column name conflicts."""
    one_hot = pd.get_dummies(df["Gender"], prefix="Gender")

    # Ensure there are no overlapping columns
    overlapping_columns = set(one_hot.columns) & set(df.columns)
    if overlapping_columns:
        df = df.drop(columns=overlapping_columns)

    df = df.join(one_hot)
    df = df.drop("Gender", axis=1)
    return df


# ! Old
def log_transform(df, columns):
    """Apply log transformation to specified columns."""
    for col in columns:
        df[col] = np.log(df[col] + 1)
    return df


# ! New function
# def log_transform(df, columns):
#     """Apply log transformation to specified columns."""
#     for col in columns:
#         # Add small constant and handle negative values
#         df[col] = np.log(df[col].clip(lower=1e-10) + 1)
#     return df


def standard_scale(df, columns):
    """Standardize specified columns."""
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def robust_scale(df, columns):
    """Apply Robust Scaling to specified columns."""
    scaler = RobustScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def preprocess_data(df):
    """Complete preprocessing pipeline."""
    logging.info("Starting preprocessing")

    df = drop_columns(df)
    logging.info(f"After dropping columns: {df.columns.tolist()}")

    df = fill_missing_values(df)
    logging.info(f"After imputing missing values: {df.columns.tolist()}")

    df = drop_null_columns(df)
    logging.info(f"After dropping null columns: {df.columns.tolist()}")

    df = one_hot_encode_gender(df)
    logging.info(f"After one-hot encoding gender: {df.columns.tolist()}")

    # Log transformations
    columns_log = ["MAP", "BUN", "Creatinine", "Glucose", "WBC", "Platelets"]
    df = log_transform(df, columns_log)
    logging.info(f"After log transformation: {df.columns.tolist()}")

    # Standard scaling
    columns_scale = [
        "HR",
        "O2Sat",
        "Temp",
        "MAP",
        "Resp",
        "BUN",
        "Chloride",
        "Creatinine",
        "Glucose",
        "Hct",
        "Hgb",
        "WBC",
        "Platelets",
    ]
    df = robust_scale(df, columns_scale)
    logging.info(f"After scaling: {df.columns.tolist()}")

    # Drop any remaining NaNs
    df = df.dropna()
    logging.info(f"After dropping remaining NaNs: {df.columns.tolist()}")

    # Convert all column names to strings
    df.columns = df.columns.astype(str)
    logging.info(f"Final preprocessed columns: {df.columns.tolist()}")

    return df
