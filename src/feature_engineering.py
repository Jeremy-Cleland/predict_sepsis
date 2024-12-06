# src/feature_engineering.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .utils import (
    corr_matrix,
    diagnostic_plots,
    try_gaussian,
)  # Ensure utils are imported correctly


def drop_columns(df):
    """Drop specified redundant columns and merge Unit1 and Unit2."""
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
        "Unit1",
        "Unit2",
    }
    df = df.assign(Unit=df["Unit1"] + df["Unit2"])
    df = df.drop(columns=columns_drop)
    return df


def fill_missing_values(df):
    """Impute missing values using backfill and forward fill."""
    grouped_by_patient = df.groupby(
        "Patient_ID", group_keys=False
    )  # Added group_keys=False
    df = grouped_by_patient.apply(lambda x: x.bfill().ffill())
    return df


def drop_null_columns(df):
    """Drop columns with more than 25% null values and Patient_ID."""
    null_col = [
        "TroponinI",
        "Bilirubin_direct",
        "AST",
        "Bilirubin_total",
        "Lactate",
        "SaO2",
        "FiO2",
        "Unit",
        "Patient_ID",
    ]
    df = df.drop(columns=null_col)
    return df


def one_hot_encode_gender(df):
    """One-hot encode the Gender column."""
    one_hot = pd.get_dummies(df["Gender"], prefix="Gender")
    df = df.join(one_hot)
    df = df.drop("Gender", axis=1)
    return df


def log_transform(df, columns):
    """Apply log transformation to specified columns."""
    for col in columns:
        df[col] = np.log(df[col] + 1)
    return df


def standard_scale(df, columns):
    """Standardize specified columns."""
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def preprocess_data(df):
    """Complete preprocessing pipeline."""
    df = drop_columns(df)
    df = fill_missing_values(df)
    df = drop_null_columns(df)
    df = one_hot_encode_gender(df)
    # Log transformations
    columns_log = ["MAP", "BUN", "Creatinine", "Glucose", "WBC", "Platelets"]
    df = log_transform(df, columns_log)
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
    df = standard_scale(df, columns_scale)
    # Drop any remaining NaNs
    df = df.dropna()
    # Convert all column names to strings
    df.columns = df.columns.astype(str)
    return df
