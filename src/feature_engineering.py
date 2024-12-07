# """Feature engineering module for preprocessing medical data.

# This module provides functions for preprocessing medical dataset features including:
# - Dropping redundant and null columns
# - Imputing missing values
# - Encoding categorical variables
# - Applying transformations (log, scaling)

# The module is designed to work with patient medical data containing vital signs
# and lab test results.
# """

# import logging

# import numpy as np
# import pandas as pd
# from sklearn.experimental import enable_iterative_imputer  # noqa
# from sklearn.impute import IterativeImputer
# from sklearn.preprocessing import RobustScaler, StandardScaler


# def drop_columns(df):
#     """Drop specified redundant columns from the dataset.

#     Args:
#         df (pd.DataFrame): Input dataframe containing medical data

#     Returns:
#         pd.DataFrame: Dataframe with redundant columns removed

#     Note:
#         Preserves Unit1 and Unit2 columns while removing other specified columns
#     """
#     columns_drop = {
#         "Unnamed: 0",  # Index column
#         # Vital signs that are derived or redundant
#         "SBP",
#         "DBP",
#         "EtCO2",
#         # Blood gas and chemistry redundancies
#         "BaseExcess",
#         "HCO3",
#         "pH",
#         "PaCO2",
#         # Duplicated or highly correlated lab values
#         "Alkalinephos",
#         "Calcium",
#         "Magnesium",
#         "Phosphate",
#         "Potassium",
#         "PTT",
#         "Fibrinogen",
#     }
#     df = df.drop(columns=columns_drop)
#     return df


# def fill_missing_values(df):
#     """Impute missing values using iterative imputation (MICE algorithm).

#     Args:
#         df (pd.DataFrame): Input dataframe with missing values

#     Returns:
#         pd.DataFrame: Dataframe with imputed values

#     Note:
#         - Preserves categorical columns during imputation
#         - Uses mean initialization and 20 maximum iterations
#         - Maintains reproducibility with fixed random state
#     """
#     # Create a copy of the dataframe
#     df_copy = df.copy()

#     # Separate Patient_ID and categorical columns
#     id_column = df_copy["Patient_ID"]
#     categorical_columns = ["Gender", "Unit1", "Unit2"]
#     categorical_data = df_copy[categorical_columns]

#     # Get numerical columns for imputation
#     numerical_columns = df_copy.select_dtypes(include=[np.number]).columns
#     numerical_data = df_copy[numerical_columns]

#     # Initialize and fit the IterativeImputer
#     imputer = IterativeImputer(
#         random_state=42,  # Ensures reproducibility.
#         max_iter=30,  # Maximum number of iterations.
#         initial_strategy="mean",  # Initial imputation strategy.
#         skip_complete=True,  # Skips columns without missing values to save computation.
#     )

#     # Perform imputation on numerical columns
#     imputed_numerical = pd.DataFrame(
#         imputer.fit_transform(numerical_data),  # Perform imputation.
#         columns=numerical_columns,  # Preserve original column names.
#         index=df_copy.index,  # Maintain original index.
#     )

#     # Combine the imputed numerical data with categorical data
#     df = pd.concat([id_column, categorical_data, imputed_numerical], axis=1)

#     return df


# def drop_null_columns(df):
#     """Drop specified columns if they exist."""
#     null_col = [
#         "TroponinI",
#         "Bilirubin_direct",
#         "AST",
#         "Bilirubin_total",
#         "Lactate",
#         "SaO2",
#         "FiO2",
#         "Patient_ID",
#     ]

#     # Identify columns that exist in the DataFrame
#     existing_cols = [col for col in null_col if col in df.columns]

#     if existing_cols:
#         df = df.drop(columns=existing_cols)
#         logger.info(f"Dropped columns: {existing_cols}")
#     else:
#         logging.warning(f"No columns to drop from drop_null_columns: {null_col}")

#     return df


# def one_hot_encode_gender(df):
#     """One-hot encode the Gender column, ensuring no column name conflicts."""
#     one_hot = pd.get_dummies(df["Gender"], prefix="Gender")

#     # Ensure there are no overlapping columns
#     overlapping_columns = set(one_hot.columns) & set(df.columns)
#     if overlapping_columns:
#         df = df.drop(columns=overlapping_columns)

#     df = df.join(one_hot)
#     df = df.drop("Gender", axis=1)
#     return df


# def log_transform(df, columns):
#     """Apply log transformation to handle skewed numeric features.

#     Args:
#         df (pd.DataFrame): Input dataframe
#         columns (list): List of column names to transform

#     Returns:
#         pd.DataFrame: Dataframe with log-transformed columns

#     Note:
#         Uses log(x + 1) transformation with minimum clipping at 1e-5
#         to handle zeros and small values
#     """
#     for col in columns:
#         # Clip values to prevent log(0) or log(negative)
#         df[col] = np.log(df[col].clip(lower=1e-5) + 1)
#     return df


# def standard_scale(df, columns):
#     """Standardize specified columns."""
#     scaler = StandardScaler()
#     df[columns] = scaler.fit_transform(df[columns])
#     return df


# def robust_scale(df, columns):
#     """Apply Robust Scaling to specified columns."""
#     scaler = RobustScaler()
#     df[columns] = scaler.fit_transform(df[columns])
#     return df


# def preprocess_data(df):
#     """Execute complete preprocessing pipeline for medical data.

#     Pipeline steps:
#     1. Drop redundant columns
#     2. Impute missing values using MICE
#     3. Drop specified null columns
#     4. One-hot encode gender
#     5. Apply log transformation to skewed features
#     6. Apply robust scaling to numeric features
#     7. Handle remaining NaN values
#     8. Standardize column names

#     Args:
#         df (pd.DataFrame): Raw input dataframe

#     Returns:
#         pd.DataFrame: Fully preprocessed dataframe ready for modeling

#     Note:
#         Logs progress at each major preprocessing step
#     """
#     logger.info("Starting preprocessing")

#     df = drop_columns(df)
#     logger.info(f"After dropping columns: {df.columns.tolist()}")

#     df = fill_missing_values(df)
#     logger.info(f"After imputing missing values: {df.columns.tolist()}")

#     df = drop_null_columns(df)
#     logger.info(f"After dropping null columns: {df.columns.tolist()}")

#     df = one_hot_encode_gender(df)
#     logger.info(f"After one-hot encoding gender: {df.columns.tolist()}")

#     # Log transformations
#     columns_log = ["MAP", "BUN", "Creatinine", "Glucose", "WBC", "Platelets"]
#     df = log_transform(df, columns_log)
#     logger.info(f"After log transformation: {df.columns.tolist()}")

#     # Standard scaling
#     columns_scale = [
#         "HR",
#         "O2Sat",
#         "Temp",
#         "MAP",
#         "Resp",
#         "BUN",
#         "Chloride",
#         "Creatinine",
#         "Glucose",
#         "Hct",
#         "Hgb",
#         "WBC",
#         "Platelets",
#     ]
#     df = robust_scale(df, columns_scale)
#     logger.info(f"After scaling: {df.columns.tolist()}")

#     # Drop any remaining NaNs
#     df = df.dropna()
#     logger.info(f"After dropping remaining NaNs: {df.columns.tolist()}")

#     # Convert all column names to strings
#     df.columns = df.columns.astype(str)
#     logger.info(f"Final preprocessed columns: {df.columns.tolist()}")

#     return df


# import logging  # Required to use logging.DEBUG

# import numpy as np
# import pandas as pd
# from sklearn.experimental import enable_iterative_imputer  # noqa
# from sklearn.impute import IterativeImputer
# from sklearn.preprocessing import RobustScaler

# from src.logger_config import setup_logger

# # Initialize logger with custom log level ("DEBUG") and JSON formatting disabled
# logger = setup_logger(
#     name="sepsis_prediction.preprocessing",
#     log_file="logs/preprocessing.log",
#     level=logging.DEBUG,  # Custom log level
#     use_json=False,  # Disable JSON formatting
# )


# def drop_redundant_columns(df):
#     """Drop specified redundant columns from the dataset."""
#     columns_drop = {
#         "Unnamed: 0",  # Index column
#         # We'll keep SBP and DBP as they're used for derived features
#         "EtCO2",  # High missing rate
#         # Blood gas and chemistry redundancies
#         "BaseExcess",
#         "HCO3",
#         "pH",
#         "PaCO2",
#         # High missing rate lab values
#         "AST",
#         "Alkalinephos",
#         "Bilirubin_direct",
#         "Bilirubin_total",
#         "Lactate",
#         "TroponinI",
#         "SaO2",
#         "FiO2",
#     }

#     # Only drop columns that exist
#     columns_to_drop = [col for col in columns_drop if col in df.columns]
#     df = df.drop(columns=columns_to_drop, errors="ignore")
#     return df


# def engineer_vital_features(df):
#     """Engineer features from vital signs."""
#     df = df.copy()

#     # Calculate shock index if components exist
#     if "HR" in df.columns and "SBP" in df.columns:
#         df["shock_index"] = df["HR"] / df["SBP"].clip(lower=1)

#     # Calculate pulse pressure if components exist
#     if "SBP" in df.columns and "DBP" in df.columns:
#         df["pulse_pressure"] = df["SBP"] - df["DBP"]

#     # Group by patient for temporal features
#     patient_groups = df.groupby("Patient_ID")

#     # Core vital signs that are typically well-measured
#     vital_signs = ["HR", "O2Sat", "Temp", "MAP", "Resp"]
#     vital_signs = [vs for vs in vital_signs if vs in df.columns]

#     # Create temporal features for vitals
#     for vital in vital_signs:
#         # Rate of change
#         df[f"{vital}_rate"] = patient_groups[vital].transform(
#             lambda x: x.diff() / df["Hour"].diff()
#         )

#         # Rolling mean and std (6-hour window)
#         df[f"{vital}_rolling_mean_6h"] = patient_groups[vital].transform(
#             lambda x: x.rolling(6, min_periods=1).mean()
#         )
#         df[f"{vital}_rolling_std_6h"] = patient_groups[vital].transform(
#             lambda x: x.rolling(6, min_periods=1).std()
#         )

#     return df


# def handle_missing_values(df):
#     """Handle missing values with appropriate strategies."""
#     df = df.copy()

#     # Forward fill vital signs within patient groups (up to 4 hours)
#     vital_signs = ["HR", "O2Sat", "Temp", "SBP", "DBP", "MAP", "Resp"]
#     existing_vitals = [col for col in vital_signs if col in df.columns]

#     if existing_vitals:
#         df[existing_vitals] = df.groupby("Patient_ID")[existing_vitals].transform(
#             lambda x: x.ffill(limit=4)
#         )

#     # Create missing indicators for important lab values
#     lab_values = ["BUN", "Creatinine", "Glucose", "WBC", "Platelets"]

#     for col in lab_values:
#         if col in df.columns:
#             df[f"{col}_missing"] = df[col].isnull().astype(int)

#     # Use iterative imputation for remaining numeric values
#     numeric_cols = df.select_dtypes(include=[np.number]).columns
#     imputer = IterativeImputer(
#         random_state=42, max_iter=10, initial_strategy="median", skip_complete=True
#     )

#     df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

#     return df


# def calculate_severity_scores(df):
#     """Calculate clinical severity scores."""
#     df = df.copy()

#     # SIRS criteria (Systemic Inflammatory Response Syndrome)
#     if "Temp" in df.columns:
#         df["sirs_temp"] = ((df["Temp"] < 36) | (df["Temp"] > 38)).astype(int)
#     if "HR" in df.columns:
#         df["sirs_hr"] = (df["HR"] > 90).astype(int)
#     if "Resp" in df.columns:
#         df["sirs_resp"] = (df["Resp"] > 20).astype(int)
#     if "WBC" in df.columns:
#         df["sirs_wbc"] = ((df["WBC"] < 4) | (df["WBC"] > 12)).astype(int)

#     # Calculate total SIRS score if all components exist
#     sirs_components = ["sirs_temp", "sirs_hr", "sirs_resp", "sirs_wbc"]
#     if all(comp in df.columns for comp in sirs_components):
#         df["sirs_score"] = df[sirs_components].sum(axis=1)

#     return df


# def preprocess_data(df):
#     """
#     Execute complete preprocessing pipeline for sepsis prediction.

#     Steps:
#     1. Drop redundant/highly missing columns
#     2. Handle missing values
#     3. Engineer vital sign features
#     4. Calculate severity scores
#     5. Scale numeric features
#     6. Clean up intermediate columns

#     Args:
#         df (pd.DataFrame): Raw input dataframe

#     Returns:
#         pd.DataFrame: Preprocessed dataframe ready for modeling
#     """
#     logger.info("Starting preprocessing pipeline")

#     # Make a copy to avoid modifying original
#     df = df.copy()

#     # Step 1: Drop redundant columns
#     df = drop_redundant_columns(df)
#     logger.info(f"After dropping columns: {df.shape[1]} features remaining")

#     # Step 2: Handle missing values
#     df = handle_missing_values(df)
#     logger.info("Completed missing value imputation")

#     # Step 3: Engineer vital features
#     df = engineer_vital_features(df)
#     logger.info("Created vital sign features")

#     # Step 4: Calculate severity scores
#     df = calculate_severity_scores(df)
#     logger.info("Calculated clinical severity scores")

#     # Step 5: Scale numeric features
#     numeric_cols = df.select_dtypes(include=[np.number]).columns
#     numeric_cols = numeric_cols.drop(
#         ["Patient_ID", "Hour", "SepsisLabel"], errors="ignore"
#     )

#     if len(numeric_cols) > 0:
#         scaler = RobustScaler()
#         df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

#     # Step 6: One-hot encode categorical variables
#     categorical_cols = ["Gender", "Unit1", "Unit2"]
#     existing_cats = [col for col in categorical_cols if col in df.columns]

#     for col in existing_cats:
#         dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
#         df = pd.concat([df, dummies], axis=1)
#         df.drop(columns=[col], inplace=True)

#     logger.info(f"Final preprocessed data shape: {df.shape}")

#     return df

# feature_engineering.py

import os
import time

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, cpu_count, delayed
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler

from src.logger_config import get_logger

logger = get_logger("sepsis_prediction.preprocessing")


def parallel_impute(data, imputer, n_jobs):
    """
    Perform parallel imputation using IterativeImputer with improved monitoring and convergence.
    """

    # Convert to numpy array first if it's a DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.values

    # Get actual number of jobs based on CPU cores
    n_jobs = min(cpu_count(), n_jobs) if n_jobs > 0 else cpu_count()
    chunk_size = len(data) // n_jobs

    logger.info(f"Starting parallel imputation using {n_jobs} CPU cores")
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Chunk size: {chunk_size} rows")

    # Create chunks with overlap to improve imputation at boundaries
    overlap = 100  # Number of rows to overlap between chunks
    chunks = []
    convergence_info = []

    for i in range(n_jobs):
        start = max(0, i * chunk_size - overlap)
        end = min(len(data), (i + 1) * chunk_size + overlap)
        chunks.append(data[start:end])

    def impute_chunk(chunk, chunk_id):
        # Calculate valid min/max bounds for non-missing values
        chunk_min = np.nanmin(chunk, axis=0)
        chunk_max = np.nanmax(chunk, axis=0)

        # Handle columns with all missing values or invalid bounds
        valid_bounds = chunk_min < chunk_max
        if not np.all(valid_bounds):
            logger.warning(
                f"Chunk {chunk_id}: Found {np.sum(~valid_bounds)} columns with invalid bounds"
            )
            # For invalid bounds, use reasonable defaults based on data type
            chunk_min[~valid_bounds] = -1e6
            chunk_max[~valid_bounds] = 1e6

        # Configure chunk-specific imputer with monitoring
        chunk_imputer = IterativeImputer(
            max_iter=50,
            tol=1e-3,
            random_state=42 + chunk_id,
            initial_strategy="mean",  # Changed to mean for better stability
            min_value=chunk_min,
            max_value=chunk_max,
            verbose=2,
            imputation_order="ascending",
            n_nearest_features=5,  # Limit number of predictors for speed
        )

        start_time = time.time()
        imputed_chunk = chunk_imputer.fit_transform(chunk)

        # Collect convergence information
        n_iter_reached = getattr(chunk_imputer, "n_iter_", 0)
        tolerance_reached = getattr(chunk_imputer, "_imputed_mask", None)

        convergence_info.append(
            {
                "chunk_id": chunk_id,
                "n_iterations": n_iter_reached,
                "time_taken": time.time() - start_time,
                "shape": chunk.shape,
                "missing_pct": np.isnan(chunk).mean() * 100,
            }
        )

        return imputed_chunk, convergence_info[-1]

    # Execute parallel imputation with progress monitoring
    results = Parallel(n_jobs=n_jobs, verbose=10, backend="loky")(
        delayed(impute_chunk)(chunk, i) for i, chunk in enumerate(chunks)
    )

    # Extract results and convergence info
    imputed_chunks, conv_info = zip(*results)

    # Remove overlap regions and combine chunks
    final_chunks = []
    for i, chunk in enumerate(imputed_chunks):
        if i == 0:
            final_chunks.append(chunk[:-overlap] if len(imputed_chunks) > 1 else chunk)
        elif i == len(imputed_chunks) - 1:
            final_chunks.append(chunk[overlap:])
        else:
            final_chunks.append(chunk[overlap:-overlap])

    # Log convergence information
    logger.info("Imputation Convergence Summary:")
    for info in conv_info:
        logger.info(
            f"Chunk {info['chunk_id']}: "
            f"Iterations={info['n_iterations']}, "
            f"Time={info['time_taken']:.2f}s, "
            f"Missing={info['missing_pct']:.2f}%"
        )

    combined_result = np.vstack(final_chunks)
    logger.info(f"Parallel imputation completed. Final shape: {combined_result.shape}")

    return combined_result


def handle_missing_values(df):
    """Handle missing values with improved parallel imputation."""
    df = df.copy()
    logger.info("Starting missing value handling with improved monitoring")

    # Forward fill vital signs within patient groups (up to 4 hours)
    vital_signs = ["HR", "O2Sat", "Temp", "SBP", "DBP", "MAP", "Resp"]
    existing_vitals = [col for col in vital_signs if col in df.columns]

    if existing_vitals:
        df[existing_vitals] = df.groupby("Patient_ID")[existing_vitals].transform(
            lambda x: x.ffill(limit=4)
        )
        logger.debug(
            f"Completed forward-filling for {len(existing_vitals)} vital signs"
        )

    # Log missing value statistics before imputation
    missing_stats = df.isnull().sum() / len(df) * 100
    logger.info("Missing value percentages before imputation:")
    for col, pct in missing_stats[missing_stats > 0].items():
        logger.info(f"{col}: {pct:.2f}%")

    # Create missing indicators for important lab values
    lab_values = ["BUN", "Creatinine", "Glucose", "WBC", "Platelets"]
    for col in lab_values:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isnull().astype(int)

    # Use parallel iterative imputation for remaining numeric values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols.drop(
        ["Patient_ID", "Hour", "SepsisLabel"]
        + [f"{col}_missing" for col in lab_values],
        errors="ignore",
    )

    # Check for columns with too many missing values
    missing_pct = df[numeric_cols].isnull().mean()
    too_many_missing = missing_pct > 0.99  # Columns with >99% missing
    if any(too_many_missing):
        logger.warning("Dropping columns with >99% missing values:")
        for col in numeric_cols[too_many_missing]:
            logger.warning(f"{col}: {missing_pct[col]*100:.2f}% missing")
        numeric_cols = numeric_cols[~too_many_missing]

    logger.info(f"Starting parallel imputation for {len(numeric_cols)} numeric columns")

    # Validate data before imputation
    numeric_data = df[numeric_cols]
    if numeric_data.empty:
        logger.error("No numeric columns to impute after filtering")
        raise ValueError("No valid columns for imputation")

    # Log statistics about the data before imputation
    logger.info("\nNumeric columns statistics before imputation:")
    stats = numeric_data.agg(["count", "mean", "std", "min", "max"]).round(3)
    for col in numeric_cols:
        logger.info(f"\n{col}:")
        logger.info(f"Count: {stats.loc['count', col]}")
        logger.info(f"Mean: {stats.loc['mean', col]}")
        logger.info(f"Std: {stats.loc['std', col]}")
        logger.info(f"Min: {stats.loc['min', col]}")
        logger.info(f"Max: {stats.loc['max', col]}")

    try:
        imputed_data = parallel_impute(numeric_data, None, n_jobs=-1)
        df[numeric_cols] = pd.DataFrame(
            imputed_data, index=df.index, columns=numeric_cols
        )

        # Validate imputed data
        if df[numeric_cols].isnull().any().any():
            logger.error("Some values still missing after imputation")
            missing_cols = df[numeric_cols].columns[df[numeric_cols].isnull().any()]
            for col in missing_cols:
                logger.error(f"{col}: {df[col].isnull().mean()*100:.2f}% still missing")
            raise ValueError("Imputation failed to fill all missing values")

        logger.info("Completed parallel imputation successfully")
    except Exception as e:
        logger.error(f"Error during parallel imputation: {str(e)}")
        raise

    # Log missing value statistics after imputation
    missing_stats_after = df.isnull().sum() / len(df) * 100
    logger.info("Missing value percentages after imputation:")
    for col, pct in missing_stats_after[missing_stats_after > 0].items():
        logger.info(f"{col}: {pct:.2f}%")

    return df


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

    # 2. Analyze missing values
    missing_rates = df.isnull().mean()
    high_missing_cols = missing_rates[missing_rates > missing_threshold].index.tolist()
    if high_missing_cols:
        dropped_columns["high_missing"] = {
            col: f"{missing_rates[col]:.2%} missing" for col in high_missing_cols
        }

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

    # Combine all columns to drop
    all_drops = (
        dropped_columns.get("index", [])
        + list(dropped_columns.get("high_missing", {}).keys())
        + list(dropped_columns.get("high_correlation", {}).keys())
        + list(dropped_columns.get("clinical_redundancy", {}).keys())
    )

    # Log the analysis
    logger.info("\nColumn Drop Analysis:")
    logger.info("-" * 50)

    if dropped_columns.get("index"):
        logger.info("\nDropped index column:")
        logger.info(f"  - {dropped_columns['index'][0]}")

    if dropped_columns.get("high_missing"):
        logger.info("\nDropped due to high missing rate:")
        for col, rate in dropped_columns["high_missing"].items():
            logger.info(f"  - {col}: {rate}")

    if dropped_columns.get("high_correlation"):
        logger.info("\nDropped due to high correlation:")
        for col, info in dropped_columns["high_correlation"].items():
            logger.info(
                f"  - {col} (correlated with {info['correlated_with']}, r={info['correlation']})"
            )

    if dropped_columns.get("clinical_redundancy"):
        logger.info("\nDropped due to clinical redundancy:")
        for col, info in dropped_columns["clinical_redundancy"].items():
            logger.info(f"  - {col} (keeping {info['kept_instead']} instead)")

    # Remove the columns
    df = df.drop(columns=all_drops, errors="ignore")

    # Log retention statistics
    kept_cols = len(df.columns)
    total_cols = len(df.columns) + len(all_drops)
    logger.info(
        f"\nRetained {kept_cols}/{total_cols} columns ({kept_cols/total_cols:.1%})"
    )

    return df, dropped_columns


def engineer_vital_features(df):
    """Engineer features from vital signs."""
    df = df.copy()

    if "HR" in df.columns and "SBP" in df.columns:
        df["shock_index"] = df["HR"] / df["SBP"].clip(lower=1)

    if "SBP" in df.columns and "DBP" in df.columns:
        df["pulse_pressure"] = df["SBP"] - df["DBP"]

    patient_groups = df.groupby("Patient_ID")
    vital_signs = ["HR", "O2Sat", "Temp", "MAP", "Resp"]
    vital_signs = [vs for vs in vital_signs if vs in df.columns]

    for vital in vital_signs:
        df[f"{vital}_rate"] = patient_groups[vital].transform(
            lambda x: x.diff() / df["Hour"].diff()
        )

        df[f"{vital}_rolling_mean_6h"] = patient_groups[vital].transform(
            lambda x: x.rolling(6, min_periods=1).mean()
        )
        df[f"{vital}_rolling_std_6h"] = patient_groups[vital].transform(
            lambda x: x.rolling(6, min_periods=1).std()
        )

    return df


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

    return df


def engineer_basic_vital_features(df):
    """Engineer features that don't require imputed data."""
    df = df.copy()

    # Simple ratios and differences that can handle NaN
    if all(col in df.columns for col in ["HR", "SBP"]):
        df["shock_index"] = df["HR"] / df["SBP"].clip(lower=1)

    if all(col in df.columns for col in ["SBP", "DBP"]):
        df["pulse_pressure"] = df["SBP"] - df["DBP"]

    # Basic time-based features using existing data
    patient_groups = df.groupby("Patient_ID")
    vital_signs = ["HR", "O2Sat", "Temp", "MAP", "Resp"]
    vital_signs = [vs for vs in vital_signs if vs in df.columns]

    for vital in vital_signs:
        # Rate of change (can handle some missing values)
        df[f"{vital}_rate"] = patient_groups[vital].transform(
            lambda x: x.diff() / df["Hour"].diff()
        )

    return df


def engineer_advanced_vital_features(df):
    """Engineer features that require complete (imputed) data."""
    df = df.copy()

    patient_groups = df.groupby("Patient_ID")
    vital_signs = ["HR", "O2Sat", "Temp", "MAP", "Resp"]
    vital_signs = [vs for vs in vital_signs if vs in df.columns]

    for vital in vital_signs:
        # Rolling statistics (require complete data)
        df[f"{vital}_rolling_mean_6h"] = patient_groups[vital].transform(
            lambda x: x.rolling(6, min_periods=1).mean()
        )
        df[f"{vital}_rolling_std_6h"] = patient_groups[vital].transform(
            lambda x: x.rolling(6, min_periods=1).std()
        )

        # Baseline deviations (require complete data)
        baseline = patient_groups[vital].transform("first")
        df[f"{vital}_baseline_dev"] = df[vital] - baseline

    return df


def preprocess_pipeline(train_df, val_df=None, test_df=None):
    """
    Preprocess all datasets ensuring consistent transformation across splits.

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data
    val_df : pd.DataFrame, optional
        Validation data
    test_df : pd.DataFrame, optional
        Test data

    Returns:
    --------
    tuple
        Processed dataframes (train, val, test)
    """
    logger.info("Starting preprocessing pipeline")

    # Store transformers that need to be fit on training data
    transformers = {}

    # 1. Initial cleaning and column analysis (based on training data only)
    logger.info("Analyzing columns based on training data")
    train_df, drop_analysis = analyze_and_drop_columns(train_df)
    if val_df is not None:
        val_df = val_df.drop(
            columns=drop_analysis.get("dropped_columns", []), errors="ignore"
        )
    if test_df is not None:
        test_df = test_df.drop(
            columns=drop_analysis.get("dropped_columns", []), errors="ignore"
        )

    # 2. Basic feature engineering (features that don't require complete data)
    logger.info("Creating basic features")
    train_df = engineer_basic_vital_features(train_df)
    if val_df is not None:
        val_df = engineer_basic_vital_features(val_df)
    if test_df is not None:
        test_df = engineer_basic_vital_features(test_df)

    # 3. Handle missing values (fit on training data)
    logger.info("Handling missing values")
    train_df, imputer = handle_missing_values(train_df, fit=True)
    transformers["imputer"] = imputer

    if val_df is not None:
        val_df = handle_missing_values(val_df, imputer=imputer)
    if test_df is not None:
        test_df = handle_missing_values(test_df, imputer=imputer)

    # 4. Advanced feature engineering
    logger.info("Creating advanced features")
    train_df = engineer_advanced_vital_features(train_df)
    if val_df is not None:
        val_df = engineer_advanced_vital_features(val_df)
    if test_df is not None:
        test_df = engineer_advanced_vital_features(test_df)

    # 5. Calculate severity scores
    train_df = calculate_severity_scores(train_df)
    if val_df is not None:
        val_df = calculate_severity_scores(val_df)
    if test_df is not None:
        test_df = calculate_severity_scores(test_df)

    # 6. Handle categorical variables (fit on training data)
    categorical_cols = ["Gender", "Unit1", "Unit2"]
    existing_cats = [col for col in categorical_cols if col in train_df.columns]

    if existing_cats:
        logger.info("Encoding categorical variables")
        for col in existing_cats:
            dummies = pd.get_dummies(train_df[col], prefix=col, drop_first=True)
            train_df = pd.concat([train_df.drop(columns=[col]), dummies], axis=1)

            if val_df is not None:
                val_dummies = pd.get_dummies(val_df[col], prefix=col, drop_first=True)
                # Ensure same columns as training
                for col in dummies.columns:
                    if col not in val_dummies.columns:
                        val_dummies[col] = 0
                val_df = pd.concat(
                    [val_df.drop(columns=[col]), val_dummies[dummies.columns]], axis=1
                )

            if test_df is not None:
                test_dummies = pd.get_dummies(test_df[col], prefix=col, drop_first=True)
                # Ensure same columns as training
                for col in dummies.columns:
                    if col not in test_dummies.columns:
                        test_dummies[col] = 0
                test_df = pd.concat(
                    [test_df.drop(columns=[col]), test_dummies[dummies.columns]], axis=1
                )

    # 7. Scale numeric features (fit on training data)
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols.drop(
        ["Patient_ID", "Hour", "SepsisLabel"], errors="ignore"
    )

    if len(numeric_cols) > 0:
        logger.info("Scaling numeric features")
        scaler = RobustScaler()
        train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
        transformers["scaler"] = scaler

        if val_df is not None:
            val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
        if test_df is not None:
            test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

    logger.info(f"Final shapes - Train: {train_df.shape}")
    if val_df is not None:
        logger.info(f"Validation: {val_df.shape}")
    if test_df is not None:
        logger.info(f"Test: {test_df.shape}")

    # Save transformers for future use
    os.makedirs("models/transformers", exist_ok=True)
    for name, transformer in transformers.items():
        joblib.dump(transformer, f"models/transformers/{name}.pkl")

    return train_df, val_df, test_df
