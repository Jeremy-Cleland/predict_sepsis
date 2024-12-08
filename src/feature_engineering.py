# feature_engineering.py

import gc
import os
import tempfile
import time
from functools import partial
from multiprocessing import cpu_count
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import psutil
from joblib import Parallel, delayed
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import RobustScaler

from src.logger_config import get_logger
from src.logging_utils import (
    log_phase,
    log_step,
    log_memory,
    log_dataframe_info,
)

logger = get_logger("sepsis_prediction.preprocessing")


def analyze_and_drop_columns(df, missing_threshold=0.95, correlation_threshold=0.95):
    """
    Analyze and drop columns based on missing values, correlations, and clinical relevance.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame
    missing_threshold : float, default=0.95
        Threshold for dropping columns with missing values (0-1)
    correlation_threshold : float, default=0.95
        Threshold for identifying highly correlated features (0-1)

    Returns:
    --------
    pandas.DataFrame
        DataFrame with redundant columns removed
    dict
        Analysis results including dropped columns and their reasons
    """
    df = df.copy()
    dropped_columns = {}

    # 1. Remove index column if present
    if "Unnamed: 0" in df.columns:
        dropped_columns["index"] = ["Unnamed: 0"]
        df = df.drop("Unnamed: 0", axis=1)
        logger.info("Dropped 'Unnamed: 0' column as it is an index column.")

    # 2. Analyze missing values
    missing_rates = df.isnull().mean()
    high_missing_cols = missing_rates[missing_rates > missing_threshold].index.tolist()
    if high_missing_cols:
        dropped_columns["high_missing"] = {
            col: f"{missing_rates[col]:.2%} missing" for col in high_missing_cols
        }
        df = df.drop(columns=high_missing_cols)
        logger.info(
            f"Dropped columns due to high missing rates (> {missing_threshold*100}%) : {high_missing_cols}"
        )

    # 3. Analyze correlations between numerical columns
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    if len(numeric_cols) > 1:  # Need at least 2 columns for correlation
        corr_matrix = df[numeric_cols].corr().abs()

        # Create a mask to get only the upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find highly correlated pairs
        high_corr_pairs = []
        for col in upper.columns:
            correlated = upper[upper[col] > correlation_threshold][col]
            for idx, corr_value in correlated.items():
                high_corr_pairs.append(
                    {"col1": col, "col2": idx, "correlation": corr_value}
                )

        # Analyze correlation groups and select representatives
        if high_corr_pairs:
            dropped_columns["high_correlation"] = {}
            seen_cols = set()
            for pair in high_corr_pairs:
                if pair["col1"] not in seen_cols and pair["col2"] not in seen_cols:
                    # Keep the one with less missing values
                    col1_missing = missing_rates[pair["col1"]]
                    col2_missing = missing_rates[pair["col2"]]
                    keep_col = (
                        pair["col1"] if col1_missing < col2_missing else pair["col2"]
                    )
                    drop_col = (
                        pair["col2"] if col1_missing < col2_missing else pair["col1"]
                    )

                    dropped_columns["high_correlation"][drop_col] = {
                        "correlated_with": keep_col,
                        "correlation": f"{pair['correlation']:.2f}",
                        "missing_rate": f"{missing_rates[drop_col]:.2%}",
                    }
                    seen_cols.add(drop_col)

            # Drop the correlated columns
            correlated_drops = list(dropped_columns["high_correlation"].keys())
            df = df.drop(columns=correlated_drops)
            logger.info(
                f"Dropped columns due to high correlation (> {correlation_threshold*100}%) : {correlated_drops}"
            )

    # 4. Define clinically redundant pairs (based on medical knowledge)
    clinical_pairs = {
        "pH": ["HCO3", "BaseExcess"],  # pH is related to these measures
        "SaO2": ["FiO2"],  # Oxygen measures
        "Bilirubin_total": ["Bilirubin_direct"],  # Bilirubin measures
    }

    dropped_columns["clinical_redundancy"] = {}
    for primary, secondaries in clinical_pairs.items():
        if primary in df.columns:
            for secondary in secondaries:
                if secondary in df.columns:
                    # Keep the one with less missing values
                    if missing_rates.get(primary, 1) > missing_rates.get(secondary, 1):
                        keep_col, drop_col = secondary, primary
                    else:
                        keep_col, drop_col = primary, secondary

                    dropped_columns["clinical_redundancy"][drop_col] = {
                        "kept_instead": keep_col,
                        "missing_rate": f"{missing_rates[drop_col]:.2%}",
                    }
                    df = df.drop(columns=[drop_col])
                    logger.info(
                        f"Dropped '{drop_col}' due to clinical redundancy with '{keep_col}'."
                    )

    return df, dropped_columns


def calculate_severity_scores(df):
    """Calculate clinical severity scores."""
    df = df.copy()

    if "Temp" in df.columns:
        df["sirs_temp"] = ((df["Temp"] < 36) | (df["Temp"] > 38)).astype(int)
    if "HR" in df.columns:
        df["sirs_hr"] = (df["HR"] > 90).astype(int)
    if "Resp" in df.columns:
        df["sirs_resp"] = (df["Resp"] > 20).astype(int)
    if "WBC" in df.columns:
        df["sirs_wbc"] = ((df["WBC"] < 4) | (df["WBC"] > 12)).astype(int)

    sirs_components = ["sirs_temp", "sirs_hr", "sirs_resp", "sirs_wbc"]
    if all(comp in df.columns for comp in sirs_components):
        df["sirs_score"] = df[sirs_components].sum(axis=1)

    logger.info("Calculated SIRS severity scores.")
    log_memory(logger, "After Calculating Severity Scores")
    log_dataframe_info(logger, df, "After Calculating Severity Scores")

    return df


def engineer_basic_vital_features(df, n_jobs=-1):
    """Parallel implementation of basic vital feature engineering."""
    df = df.copy()

    # Simple calculations that don't need grouping
    if all(col in df.columns for col in ["HR", "SBP"]):
        df["shock_index"] = df["HR"] / df["SBP"].clip(lower=1)

    if all(col in df.columns for col in ["SBP", "DBP"]):
        df["pulse_pressure"] = df["SBP"] - df["DBP"]

    # Process groups in single thread when called from parallel pipeline
    vital_signs = ["HR", "O2Sat", "Temp", "MAP", "Resp"]
    vital_signs = [vs for vs in vital_signs if vs in df.columns]

    # Process each group sequentially
    for _, group in df.groupby("Patient_ID"):
        result = pd.DataFrame(index=group.index)
        for vital in vital_signs:
            result[f"{vital}_rate"] = group[vital].diff() / group["Hour"].diff()

        # Update the main dataframe
        for col in result.columns:
            df.loc[group.index, col] = result[col]

    logger.info("Engineered basic vital features.")
    log_memory(logger, "After Engineering Basic Vital Features")
    log_dataframe_info(logger, df, "After Engineering Basic Vital Features")

    return df


def engineer_advanced_vital_features(df, n_jobs=-1):
    """Parallel implementation of advanced vital feature engineering."""
    df = df.copy()

    vital_signs = ["HR", "O2Sat", "Temp", "MAP", "Resp"]
    vital_signs = [vs for vs in vital_signs if vs in df.columns]

    def process_patient_group(group_data):
        """Process a single patient group with rolling statistics."""
        result = pd.DataFrame(index=group_data.index)

        for vital in vital_signs:
            # Rolling statistics
            result[f"{vital}_rolling_mean_6h"] = (
                group_data[vital].rolling(6, min_periods=1).mean()
            )
            result[f"{vital}_rolling_std_6h"] = (
                group_data[vital].rolling(6, min_periods=1).std()
            )

            # Baseline deviation
            baseline = group_data[vital].iloc[0]
            result[f"{vital}_baseline_dev"] = group_data[vital] - baseline

        return result

    # Split data by patient
    patient_groups = [group for _, group in df.groupby("Patient_ID")]

    # Process groups in parallel
    with Parallel(n_jobs=n_jobs, verbose=0, backend="threading") as parallel:
        results = parallel(
            delayed(process_patient_group)(group) for group in patient_groups
        )

    # Combine results
    all_results = pd.concat(results)

    # Sort index to match original dataframe
    all_results = all_results.reindex(df.index)

    # Add new columns to original dataframe
    for col in all_results.columns:
        df[col] = all_results[col]

    logger.info("Engineered advanced vital features.")
    log_memory(logger, "After Engineering Advanced Vital Features")
    log_dataframe_info(logger, df, "After Engineering Advanced Vital Features")

    return df


def preprocess_chunk(chunk_data, transformers=None):
    """Process a chunk of data with given transformers."""
    chunk = chunk_data.copy()

    # Apply existing transformers if provided
    if transformers:
        if "imputer" in transformers:
            # Apply imputation
            numeric_cols = chunk.select_dtypes(include=[np.number]).columns
            chunk[numeric_cols] = transformers["imputer"].transform(chunk[numeric_cols])

        if "scaler" in transformers:
            # Apply scaling
            numeric_cols = chunk.select_dtypes(include=[np.number]).columns
            chunk[numeric_cols] = transformers["scaler"].transform(chunk[numeric_cols])

    return chunk


def process_dataframe_in_parallel(df, chunk_size=10000, n_jobs=-1):
    """Process large dataframes in parallel chunks."""
    n_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size != 0 else 0)
    chunks = np.array_split(df, n_chunks)

    with Parallel(n_jobs=n_jobs, verbose=0, backend="threading") as parallel:
        processed_chunks = parallel(
            delayed(preprocess_chunk)(chunk) for chunk in chunks
        )

    return pd.concat(processed_chunks)


def impute_chunk(chunk, chunk_id, global_min=None, global_max=None):
    """
    Impute a single chunk of data with improved stability.
    """
    try:
        # Ensure chunk is numpy array
        chunk = chunk if isinstance(chunk, np.ndarray) else chunk.values

        # Calculate bounds with safety margins
        chunk_min = global_min if global_min is not None else np.nanmin(chunk, axis=0)
        chunk_max = global_max if global_max is not None else np.nanmax(chunk, axis=0)

        # Ensure dimensions match
        if chunk_min.shape[0] != chunk.shape[1]:
            logger.warning(
                f"Dimension mismatch in chunk {chunk_id}. Adjusting bounds..."
            )
            chunk_min = chunk_min[: chunk.shape[1]]
            chunk_max = chunk_max[: chunk.shape[1]]

        # Add safety margin to bounds
        margin = (chunk_max - chunk_min) * 0.1
        chunk_min -= margin
        chunk_max += margin

        # Handle invalid bounds
        valid_bounds = chunk_min < chunk_max
        if not np.all(valid_bounds):
            invalid_count = np.sum(~valid_bounds)
            logger.warning(
                f"Chunk {chunk_id}: Found {invalid_count} columns with invalid bounds"
            )
            chunk_min[~valid_bounds] = -1e6
            chunk_max[~valid_bounds] = 1e6

        # Configure imputer with more robust settings
        chunk_imputer = IterativeImputer(
            max_iter=200,
            tol=1e-3,
            random_state=42 + chunk_id,
            initial_strategy="mean",
            min_value=chunk_min,
            max_value=chunk_max,
            verbose=0,
            imputation_order="ascending",
            n_nearest_features=min(5, chunk.shape[1] - 1),
            sample_posterior=True,
        )

        # Perform imputation
        start_time = time.time()
        try:
            imputed_chunk = chunk_imputer.fit_transform(chunk)
        except Exception as e:
            logger.error(f"Imputation failed for chunk {chunk_id}: {str(e)}")
            # Fallback to simple mean imputation
            imputed_chunk = SimpleImputer(strategy="mean").fit_transform(chunk)

        # Collect metrics
        metrics = {
            "chunk_id": chunk_id,
            "n_iterations": getattr(chunk_imputer, "n_iter_", 0),
            "time_taken": time.time() - start_time,
            "shape": chunk.shape,
            "missing_pct": np.isnan(chunk).mean() * 100,
        }

        return imputed_chunk, metrics

    except Exception as e:
        logger.error(f"Error in impute_chunk {chunk_id}: {str(e)}")
        # Return original chunk and error metrics if imputation fails
        return chunk, {
            "chunk_id": chunk_id,
            "error": str(e),
            "shape": chunk.shape,
            "time_taken": 0,
            "n_iterations": 0,
        }


def parallel_impute(data: pd.DataFrame, n_jobs: int = -1) -> np.ndarray:
    """
    Perform imputation on the entire dataset using IterativeImputer,
    ensuring consistency across all chunks.

    Parameters:
    -----------
    data : pd.DataFrame
        The numeric data to impute.
    n_jobs : int
        Number of parallel jobs.

    Returns:
    --------
    np.ndarray
        The imputed data as a NumPy array.
    """
    from sklearn.impute import IterativeImputer

    try:
        logger.info("Starting Iterative Imputer on the entire dataset")
        log_memory(logger, "Before Iterative Imputer")

        imputer = IterativeImputer(
            max_iter=100,
            random_state=42,
            n_nearest_features=5,
            initial_strategy="mean",
        )
        imputed_data = imputer.fit_transform(data)

        logger.info("Imputation completed successfully")
        log_memory(logger, "After Iterative Imputer")
        log_dataframe_info(
            logger, pd.DataFrame(imputed_data, columns=data.columns), "After Imputation"
        )

        return imputed_data

    except Exception as e:
        logger.error(f"Error during imputation: {e}", exc_info=True)
        log_memory(logger, "Error during Iterative Imputer")
        raise


def log_memory_usage():
    """Log current memory usage of the process."""
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")


def save_checkpoint(df, stage, output_dir="checkpoints"):
    """Save a checkpoint of the current dataframe."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = f"{output_dir}/checkpoint_{stage}.parquet"
    df.to_parquet(checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(stage, output_dir="checkpoints"):
    """Load a checkpoint if it exists."""
    checkpoint_path = f"{output_dir}/checkpoint_{stage}.parquet"
    if os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        return pd.read_parquet(checkpoint_path)
    return None


def process_chunk_with_memory_tracking(chunk_data, transformers=None):
    """Process a chunk of data with memory tracking."""
    try:
        initial_memory = psutil.Process().memory_info().rss
        chunk = chunk_data.copy()

        if transformers:
            if "imputer" in transformers:
                numeric_cols = chunk.select_dtypes(include=[np.number]).columns
                chunk[numeric_cols] = transformers["imputer"].transform(
                    chunk[numeric_cols]
                )

            if "scaler" in transformers:
                numeric_cols = chunk.select_dtypes(include=[np.number]).columns
                chunk[numeric_cols] = transformers["scaler"].transform(
                    chunk[numeric_cols]
                )

        final_memory = psutil.Process().memory_info().rss
        memory_diff = (final_memory - initial_memory) / 1024 / 1024  # MB

        if memory_diff > 100:  # Log if memory increase is more than 100MB
            logger.warning(
                f"Large memory increase in chunk processing: {memory_diff:.2f} MB"
            )

        return chunk

    except Exception as e:
        logger.error(f"Error in process_chunk: {str(e)}")
        raise
    finally:
        gc.collect()


def encode_categorical_variables_optimized(
    df, categorical_cols, is_training=True, training_categories=None
):
    """Memory-efficient categorical variable encoding with validation for binary units."""
    try:
        log_memory(logger, "Before Categorical Encoding")
        logger.info("Starting categorical encoding")

        def process_chunk(chunk, col, categories):
            try:
                result = {}
                chunk_col = (
                    chunk[col].fillna("Unknown").astype(str)
                )  # Handle missing values and ensure string type

                # For Unit columns, we know they're binary
                if col in ["Unit1", "Unit2"]:
                    # Ensure categories include 'Unknown'
                    categories = ["0", "1", "Unknown"]
                    result[f"{col}_0"] = (chunk_col == "0").astype(np.int8)
                    result[f"{col}_1"] = (chunk_col == "1").astype(np.int8)
                    result[f"{col}_Unknown"] = (chunk_col == "Unknown").astype(np.int8)
                else:
                    # For other categorical columns like Gender
                    for cat in categories:
                        result[f"{col}_{cat}"] = (chunk_col == cat).astype(np.int8)
                    # Add an 'Unknown' category if unseen categories are present
                    result[f"{col}_Unknown"] = (~chunk_col.isin(categories)).astype(
                        np.int8
                    )

                return pd.DataFrame(result, index=chunk.index)
            finally:
                del chunk_col
                gc.collect()

        # Calculate optimal chunk size
        available_memory = psutil.virtual_memory().available
        rows_per_chunk = min(
            50000,  # Maximum chunk size
            max(
                1000,  # Minimum chunk size
                int(available_memory / (len(df.columns) * 8 * 20)),  # Safety factor
            ),
        )
        logger.info(f"Using chunk size of {rows_per_chunk} for categorical encoding")

        categories = {} if is_training else training_categories
        result_df = df.copy()

        for col in categorical_cols:
            if col not in df.columns:
                continue

            logger.info(f"Processing column: {col}")
            log_memory(logger, f"Before encoding column {col}")

            if is_training:
                if col in ["Unit1", "Unit2"]:
                    # For units, we know the categories and add 'Unknown'
                    categories[col] = ["0", "1"]
                else:
                    # For other columns, get categories from data
                    unique_cats = set()
                    for start_idx in range(0, len(df), rows_per_chunk):
                        chunk = df.iloc[
                            start_idx : min(start_idx + rows_per_chunk, len(df))
                        ]
                        unique_cats.update(
                            chunk[col].fillna("Unknown").astype(str).unique()
                        )
                    categories[col] = sorted(
                        unique_cats - {"Unknown"}
                    )  # Exclude 'Unknown' if present

                logger.info(f"Found {len(categories[col])} categories for {col}")

            # Process in chunks
            for start_idx in range(0, len(df), rows_per_chunk):
                end_idx = min(start_idx + rows_per_chunk, len(df))
                chunk = df.iloc[start_idx:end_idx]

                # Process chunk
                encoded_chunk = process_chunk(chunk, col, categories[col])

                # Update result_df in place
                for encoded_col in encoded_chunk.columns:
                    result_df.loc[chunk.index, encoded_col] = encoded_chunk[encoded_col]

                del chunk, encoded_chunk
                gc.collect()

            # Remove original column
            result_df = result_df.drop(columns=[col])
            gc.collect()

            # Save intermediate checkpoint
            save_checkpoint(result_df, f"encoded_{col}")
            log_memory(logger, f"After encoding column {col}")
            log_dataframe_info(logger, result_df, f"After encoding column {col}")

        logger.info(
            f"Encoded categorical columns. Current features: {list(result_df.columns)}"
        )
        log_memory(logger, "After Categorical Encoding")
        log_dataframe_info(logger, result_df, "After Categorical Encoding")

        return (result_df, categories) if is_training else (result_df, None)

    except Exception as e:
        logger.error(f"Error in categorical encoding: {str(e)}")
        logger.error("Memory state when error occurred:")
        log_memory(logger, "Error during Categorical Encoding")
        raise
    finally:
        gc.collect()


def handle_missing_values_chunk(df, fit=False):
    """Single-threaded version of handle_missing_values for use in parallel processing."""
    df = df.copy()

    # Forward fill vital signs within patient groups
    vital_signs = ["HR", "O2Sat", "Temp", "SBP", "DBP", "MAP", "Resp"]
    existing_vitals = [col for col in vital_signs if col in df.columns]

    if existing_vitals:
        df[existing_vitals] = df.groupby("Patient_ID")[existing_vitals].transform(
            lambda x: x.ffill(limit=4)
        )
        logger.info("Applied forward fill to vital signs.")

    # Create missing indicators
    lab_values = ["BUN", "Creatinine", "Glucose", "WBC", "Platelets"]
    for col in lab_values:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isnull().astype(int)
            logger.info(f"Created missing indicator for {col}.")

    # Prepare numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols.drop(
        ["Patient_ID", "Hour", "SepsisLabel"]
        + [f"{col}_missing" for col in lab_values],
        errors="ignore",
    )

    # Handle columns with too many missing values
    missing_pct = df[numeric_cols].isnull().mean()
    too_many_missing = missing_pct > 0.99
    if any(too_many_missing):
        cols_to_drop = missing_pct[too_many_missing].index.tolist()
        df = df.drop(columns=cols_to_drop)
        logger.info(
            f"Dropped columns due to excessive missing values (>99%): {cols_to_drop}"
        )

    # Impute remaining numeric columns
    numeric_data = df[numeric_cols]
    if not numeric_data.empty:
        imputed_data = parallel_impute(numeric_data, n_jobs=-1)  # Single thread
        df[numeric_cols] = pd.DataFrame(
            imputed_data, index=df.index, columns=numeric_cols
        )
        logger.info("Imputed missing values in numeric columns.")
        log_memory(logger, "After Imputation")
        log_dataframe_info(logger, df, "After Imputation")

    return df


def preprocess_pipeline(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    n_jobs: int = -1,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Complete preprocessing pipeline with fixed missing value handling.
    """
    try:
        n_jobs = -1
        logger.info("Starting preprocessing pipeline")
        log_memory(logger, "Start of Preprocessing Pipeline")
        log_dataframe_info(logger, train_df, "Initial Training Data")

        # Configure parallel processing
        n_jobs = min(cpu_count(), max(1, n_jobs if n_jobs > 0 else cpu_count()))
        logger.info(f"Using {n_jobs} parallel jobs")

        # Store transformers for consistent preprocessing
        transformers = {}

        # 1. Initial cleaning and column analysis
        with log_step(logger, "Analyzing and Dropping Columns"):
            logger.info("Analyzing and dropping columns")
            train_df, drop_analysis = analyze_and_drop_columns(train_df)
            save_checkpoint(train_df, "after_column_analysis")
            log_memory(logger, "After Analyzing and Dropping Columns")
            log_dataframe_info(logger, train_df, "After Analyzing and Dropping Columns")

            # Drop the same columns from validation and test sets
            dropped_cols = set()
            if drop_analysis.get("high_missing"):
                dropped_cols.update(drop_analysis["high_missing"].keys())
            if drop_analysis.get("high_correlation"):
                dropped_cols.update(drop_analysis["high_correlation"].keys())
            if drop_analysis.get("clinical_redundancy"):
                dropped_cols.update(drop_analysis["clinical_redundancy"].keys())
            if "index" in drop_analysis:
                dropped_cols.update(drop_analysis["index"])

            if val_df is not None:
                val_df = val_df.drop(columns=list(dropped_cols), errors="ignore")
                logger.info(f"Dropped columns from validation set: {dropped_cols}")
                log_dataframe_info(
                    logger, val_df, "After Dropping Columns from Validation Set"
                )
            if test_df is not None:
                test_df = test_df.drop(columns=list(dropped_cols), errors="ignore")
                logger.info(f"Dropped columns from test set: {dropped_cols}")
                log_dataframe_info(
                    logger, test_df, "After Dropping Columns from Test Set"
                )

        # 2. Create missing indicators first
        with log_step(logger, "Creating Missing Indicators"):
            lab_values = ["BUN", "Creatinine", "Glucose", "WBC", "Platelets"]
            for df, name in zip([train_df, val_df, test_df], ["train", "val", "test"]):
                if df is not None:
                    for col in lab_values:
                        if col in df.columns:
                            df[f"{col}_missing"] = df[col].isnull().astype(int)
                            logger.info(
                                f"Created missing indicator for {col} in {name} set."
                            )
                            log_dataframe_info(
                                logger,
                                df,
                                f"After Creating Missing Indicator for {col} in {name} set",
                            )

            log_memory(logger, "After Creating Missing Indicators")

        # 3. Basic feature engineering
        with log_step(logger, "Basic Feature Engineering"):
            logger.info("Creating basic features")
            dfs = [train_df]
            if val_df is not None:
                dfs.append(val_df)
            if test_df is not None:
                dfs.append(test_df)

            processed_dfs = []
            for idx, df in enumerate(dfs):
                try:
                    processed_df = engineer_basic_vital_features(df, n_jobs=1)
                    processed_dfs.append(processed_df)
                    save_checkpoint(processed_df, f"basic_features_{idx}")
                    gc.collect()
                    logger.info(f"Processed basic features for dataset {idx}")
                    log_memory(
                        logger, f"After Basic Feature Engineering for dataset {idx}"
                    )
                    log_dataframe_info(
                        logger,
                        processed_df,
                        f"After Basic Feature Engineering for dataset {idx}",
                    )
                except Exception as e:
                    logger.error(
                        f"Error in basic feature engineering for dataset {idx}: {str(e)}"
                    )
                    raise

            train_df = processed_dfs[0]
            if val_df is not None:
                val_df = processed_dfs[1]
            if test_df is not None:
                test_df = processed_dfs[2]

            del processed_dfs
            gc.collect()
            log_memory(logger, "After Basic Feature Engineering")
            log_dataframe_info(logger, train_df, "Processed Training Set")
            if val_df is not None:
                log_dataframe_info(logger, val_df, "Processed Validation Set")
            if test_df is not None:
                log_dataframe_info(logger, test_df, "Processed Test Set")

        # 4. Handle missing values
        with log_step(logger, "Handling Missing Values"):
            logger.info("Handling missing values in datasets")
            train_df = handle_missing_values_chunk(train_df)
            if val_df is not None:
                val_df = handle_missing_values_chunk(val_df)
            if test_df is not None:
                test_df = handle_missing_values_chunk(test_df)

            save_checkpoint(train_df, "after_missing_values")
            if val_df is not None:
                save_checkpoint(val_df, "after_missing_values_val")
            if test_df is not None:
                save_checkpoint(test_df, "after_missing_values_test")
            log_memory(logger, "After Handling Missing Values")
            log_dataframe_info(
                logger, train_df, "After Handling Missing Values - Training Set"
            )
            if val_df is not None:
                log_dataframe_info(
                    logger, val_df, "After Handling Missing Values - Validation Set"
                )
            if test_df is not None:
                log_dataframe_info(
                    logger, test_df, "After Handling Missing Values - Test Set"
                )

        # 5. Calculate severity scores
        with log_step(logger, "Calculating Severity Scores"):
            logger.info("Calculating severity scores")
            train_df = calculate_severity_scores(train_df)
            if val_df is not None:
                val_df = calculate_severity_scores(val_df)
            if test_df is not None:
                test_df = calculate_severity_scores(test_df)

            save_checkpoint(train_df, "after_severity_scores")
            if val_df is not None:
                save_checkpoint(val_df, "after_severity_scores_val")
            if test_df is not None:
                save_checkpoint(test_df, "after_severity_scores_test")
            log_memory(logger, "After Calculating Severity Scores")
            log_dataframe_info(
                logger, train_df, "After Calculating Severity Scores - Training Set"
            )
            if val_df is not None:
                log_dataframe_info(
                    logger, val_df, "After Calculating Severity Scores - Validation Set"
                )
            if test_df is not None:
                log_dataframe_info(
                    logger, test_df, "After Calculating Severity Scores - Test Set"
                )

        # 6. Handle categorical variables
        with log_step(logger, "Categorical Encoding"):
            logger.info("Starting categorical encoding process")
            categorical_cols = ["Gender", "Unit1", "Unit2"]
            existing_cats = [col for col in categorical_cols if col in train_df.columns]

            if existing_cats:
                try:
                    logger.info("Encoding training data...")
                    train_df, categories = encode_categorical_variables_optimized(
                        train_df, existing_cats, is_training=True
                    )
                    save_checkpoint(train_df, "encoded_train")
                    transformers["categories"] = categories
                    gc.collect()
                    logger.info("Encoded training data.")
                    log_memory(logger, "After Encoding Training Data")
                    log_dataframe_info(logger, train_df, "After Encoding Training Data")

                    if val_df is not None:
                        logger.info("Encoding validation data...")
                        val_df, _ = encode_categorical_variables_optimized(
                            val_df,
                            existing_cats,
                            is_training=False,
                            training_categories=categories,
                        )
                        save_checkpoint(val_df, "encoded_val")
                        gc.collect()
                        logger.info("Encoded validation data.")
                        log_memory(logger, "After Encoding Validation Data")
                        log_dataframe_info(
                            logger, val_df, "After Encoding Validation Data"
                        )

                    if test_df is not None:
                        logger.info("Encoding test data...")
                        test_df, _ = encode_categorical_variables_optimized(
                            test_df,
                            existing_cats,
                            is_training=False,
                            training_categories=categories,
                        )
                        save_checkpoint(test_df, "encoded_test")
                        gc.collect()
                        logger.info("Encoded test data.")
                        log_memory(logger, "After Encoding Test Data")
                        log_dataframe_info(logger, test_df, "After Encoding Test Data")

                except Exception as e:
                    logger.error(f"Error in categorical encoding: {str(e)}")
                    raise

            log_memory(logger, "After Categorical Encoding")
            if existing_cats:
                log_dataframe_info(
                    logger, train_df, "After Categorical Encoding - Training Set"
                )
                if val_df is not None:
                    log_dataframe_info(
                        logger, val_df, "After Categorical Encoding - Validation Set"
                    )
                if test_df is not None:
                    log_dataframe_info(
                        logger, test_df, "After Categorical Encoding - Test Set"
                    )

        # 7. Scale numeric features
        with log_step(logger, "Scaling Numeric Features"):
            logger.info("Scaling numeric features with RobustScaler")
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns
            # Exclude ID columns, target variable, and missing indicators from scaling
            exclude_cols = ["Patient_ID", "Hour", "SepsisLabel"] + [
                f"{col}_missing"
                for col in ["BUN", "Creatinine", "Glucose", "WBC", "Platelets"]
            ]
            numeric_cols = numeric_cols.drop(exclude_cols, errors="ignore")

            if len(numeric_cols) > 0:
                try:
                    scaler = RobustScaler()
                    train_df[numeric_cols] = scaler.fit_transform(
                        train_df[numeric_cols]
                    )
                    transformers["scaler"] = scaler
                    joblib.dump(scaler, "models/transformers/scaler.pkl")
                    logger.info("Fitted and saved RobustScaler.")

                    if val_df is not None:
                        val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
                        logger.info("Transformed validation data with RobustScaler.")
                        log_dataframe_info(
                            logger, val_df, "After Scaling - Validation Set"
                        )
                    if test_df is not None:
                        test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])
                        logger.info("Transformed test data with RobustScaler.")
                        log_dataframe_info(logger, test_df, "After Scaling - Test Set")

                    save_checkpoint(train_df, "final_train")
                    if val_df is not None:
                        save_checkpoint(val_df, "final_val")
                    if test_df is not None:
                        save_checkpoint(test_df, "final_test")
                    log_memory(logger, "After Scaling Numeric Features")
                    log_dataframe_info(logger, train_df, "After Scaling - Training Set")

                except Exception as e:
                    logger.error(f"Error in feature scaling: {str(e)}")
                    raise

        # 8. Drop `Patient_ID` from all datasets before modeling
        with log_step(logger, "Dropping 'Patient_ID' Columns"):
            logger.info("Dropping 'Patient_ID' from all datasets before modeling")
            for df, name in zip([train_df, val_df, test_df], ["train", "val", "test"]):
                if df is not None and "Patient_ID" in df.columns:
                    df.drop(columns=["Patient_ID"], inplace=True)
                    logger.info(f"Dropped 'Patient_ID' from {name} set.")
                    log_dataframe_info(
                        logger,
                        df,
                        f"After Dropping 'Patient_ID' - {name.capitalize()} Set",
                    )

        # Save transformers
        with log_step(logger, "Saving Transformers"):
            try:
                os.makedirs("models/transformers", exist_ok=True)
                for name, transformer in transformers.items():
                    joblib.dump(transformer, f"models/transformers/{name}.pkl")
                    logger.info(
                        f"Saved transformer '{name}' to models/transformers/{name}.pkl"
                    )
            except Exception as e:
                logger.error(f"Error saving transformers: {str(e)}")
                raise

        logger.info(
            f"Preprocessing pipeline completed successfully. Final features: {list(train_df.columns)}"
        )
        log_memory(logger, "End of Preprocessing Pipeline")
        log_dataframe_info(logger, train_df, "Final Training Set")
        if val_df is not None:
            log_dataframe_info(logger, val_df, "Final Validation Set")
        if test_df is not None:
            log_dataframe_info(logger, test_df, "Final Test Set")

        return train_df, val_df, test_df

    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")
        log_memory(logger, "Error in Preprocessing Pipeline")
        raise
