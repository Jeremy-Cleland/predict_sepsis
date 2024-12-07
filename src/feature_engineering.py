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
            max_iter=100,  # Increased from 50
            tol=1e-2,  # Relaxed from 1e-3
            random_state=42 + chunk_id,
            initial_strategy="mean",  # Changed to mean for better stability
            min_value=chunk_min,
            max_value=chunk_max,
            verbose=0,
            imputation_order="ascending",
            n_nearest_features=3,  # Reduced from 5 for faster convergence
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
    # ! Change Verbose to 10 for extensive logging
    results = Parallel(n_jobs=n_jobs, verbose=0, backend="loky")(
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

    #! Add for Logging
    # Log missing value statistics before imputation
    # missing_stats = df.isnull().sum() / len(df) * 100
    # logger.info("Missing value percentages before imputation:")
    # for col, pct in missing_stats[missing_stats > 0].items():
    #     logger.info(f"{col}: {pct:.2f}%")

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

    #! Add for Logging
    # Log statistics about the data before imputation
    # logger.info("\nNumeric columns statistics before imputation:")
    # stats = numeric_data.agg(["count", "mean", "std", "min", "max"]).round(3)
    # for col in numeric_cols:
    #     logger.info(f"\n{col}:")
    #     logger.info(f"Count: {stats.loc['count', col]}")
    #     logger.info(f"Mean: {stats.loc['mean', col]}")
    #     logger.info(f"Std: {stats.loc['std', col]}")
    #     logger.info(f"Min: {stats.loc['min', col]}")
    #     logger.info(f"Max: {stats.loc['max', col]}")

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

    # ! Add for Logging
    # # Log the analysis
    # logger.info("\nColumn Drop Analysis:")
    # logger.info("-" * 50)

    # if dropped_columns.get("index"):
    #     logger.info("\nDropped index column:")
    #     logger.info(f"  - {dropped_columns['index'][0]}")

    # if dropped_columns.get("high_missing"):
    #     logger.info("\nDropped due to high missing rate:")
    #     for col, rate in dropped_columns["high_missing"].items():
    #         logger.info(f"  - {col}: {rate}")

    # if dropped_columns.get("high_correlation"):
    #     logger.info("\nDropped due to high correlation:")
    #     for col, info in dropped_columns["high_correlation"].items():
    #         logger.info(
    #             f"  - {col} (correlated with {info['correlated_with']}, r={info['correlation']})"
    #         )

    # if dropped_columns.get("clinical_redundancy"):
    #     logger.info("\nDropped due to clinical redundancy:")
    #     for col, info in dropped_columns["clinical_redundancy"].items():
    #         logger.info(f"  - {col} (keeping {info['kept_instead']} instead)")

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


def engineer_basic_vital_features(df, n_jobs=-1):
    """Parallel implementation of basic vital feature engineering."""
    df = df.copy()

    # Simple calculations that don't need grouping
    if all(col in df.columns for col in ["HR", "SBP"]):
        df["shock_index"] = df["HR"] / df["SBP"].clip(lower=1)

    if all(col in df.columns for col in ["SBP", "DBP"]):
        df["pulse_pressure"] = df["SBP"] - df["DBP"]

    # Prepare for parallel processing
    vital_signs = ["HR", "O2Sat", "Temp", "MAP", "Resp"]
    vital_signs = [vs for vs in vital_signs if vs in df.columns]

    def process_patient_group(group_data):
        """Process a single patient group."""
        result = pd.DataFrame(index=group_data.index)

        for vital in vital_signs:
            # Rate of change calculation
            result[f"{vital}_rate"] = (
                group_data[vital].diff() / group_data["Hour"].diff()
            )

        return result

    # Split data by patient for parallel processing
    patient_groups = [group for _, group in df.groupby("Patient_ID")]

    # Process groups in parallel
    # ! Change Verbose to 10 for extensive logging
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

    return df


# def engineer_advanced_vital_features(df):
#     """Engineer features that require complete (imputed) data."""
#     df = df.copy()

#     patient_groups = df.groupby("Patient_ID")
#     vital_signs = ["HR", "O2Sat", "Temp", "MAP", "Resp"]
#     vital_signs = [vs for vs in vital_signs if vs in df.columns]

#     for vital in vital_signs:
#         # Rolling statistics (require complete data)
#         df[f"{vital}_rolling_mean_6h"] = patient_groups[vital].transform(
#             lambda x: x.rolling(6, min_periods=1).mean()
#         )
#         df[f"{vital}_rolling_std_6h"] = patient_groups[vital].transform(
#             lambda x: x.rolling(6, min_periods=1).std()
#         )

#         # Baseline deviations (require complete data)
#         baseline = patient_groups[vital].transform("first")
#         df[f"{vital}_baseline_dev"] = df[vital] - baseline

#     return df


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
    # ! Change Verbose to 10 for extensive logging
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


def encode_categorical_variables(
    df, categorical_cols, is_training=True, training_categories=None
):
    """
    Encode categorical variables consistently across train/val/test sets.

    Parameters:
    -----------
    df : pd.DataFrame
        Data to encode
    categorical_cols : list
        List of categorical columns to encode
    is_training : bool
        Whether this is the training set
    training_categories : dict, optional
        Dictionary of categories from training set for consistent encoding

    Returns:
    --------
    pd.DataFrame, dict
        Encoded dataframe and dictionary of categories (if training)
    """
    df = df.copy()
    categories = {} if is_training else training_categories

    for col in categorical_cols:
        if col not in df.columns:
            continue

        # For numeric categorical variables (like Gender), first convert to string
        df[col] = df[col].astype(str)

        if is_training:
            # Store unique categories from training
            categories[col] = sorted(df[col].unique())

        # Create dummy variables
        dummies = pd.get_dummies(df[col], prefix=col)

        # If not training, ensure consistent columns with training set
        if not is_training:
            for cat in categories[col]:
                dummy_col = f"{col}_{cat}"
                if dummy_col not in dummies.columns:
                    dummies[dummy_col] = 0
            # Only keep columns that were in training
            dummies = dummies[[f"{col}_{cat}" for cat in categories[col]]]

        # Remove original column and add encoded columns
        df = df.drop(columns=[col])
        df = pd.concat([df, dummies], axis=1)

    if is_training:
        return df, categories
    return df


def preprocess_pipeline(train_df, val_df=None, test_df=None, n_jobs=-1):
    """Parallel version of preprocessing pipeline."""
    logger.info("Starting parallel preprocessing pipeline")

    # Initialize transformers dictionary
    transformers = {}

    # 1. Initial cleaning and column analysis
    logger.info("Analyzing columns based on training data")
    train_df, drop_analysis = analyze_and_drop_columns(train_df)

    # Store dropped columns for consistent preprocessing
    transformers["dropped_columns"] = drop_analysis

    if val_df is not None:
        val_df = val_df.drop(
            columns=drop_analysis.get("dropped_columns", []), errors="ignore"
        )
    if test_df is not None:
        test_df = test_df.drop(
            columns=drop_analysis.get("dropped_columns", []), errors="ignore"
        )

    # 2. Basic feature engineering in parallel
    logger.info("Creating basic features in parallel")
    train_df = engineer_basic_vital_features(train_df, n_jobs=n_jobs)
    if val_df is not None:
        val_df = engineer_basic_vital_features(val_df, n_jobs=n_jobs)
    if test_df is not None:
        test_df = engineer_basic_vital_features(test_df, n_jobs=n_jobs)

    # 3. Parallel missing value handling
    # ! Change Verbose to 10 for extensive logging
    logger.info("Handling missing values in parallel")
    train_chunks = np.array_split(train_df, os.cpu_count())
    with Parallel(n_jobs=n_jobs, verbose=0, backend="threading") as parallel:
        processed_chunks = parallel(
            delayed(handle_missing_values)(chunk) for chunk in train_chunks
        )
    train_df = pd.concat(processed_chunks)

    # 4. Advanced feature engineering in parallel
    logger.info("Creating advanced features in parallel")
    train_df = engineer_advanced_vital_features(train_df, n_jobs=n_jobs)
    if val_df is not None:
        val_df = engineer_advanced_vital_features(val_df, n_jobs=n_jobs)
    if test_df is not None:
        test_df = engineer_advanced_vital_features(test_df, n_jobs=n_jobs)

    # 5. Calculate severity scores
    train_df = calculate_severity_scores(train_df)
    if val_df is not None:
        val_df = calculate_severity_scores(val_df)
    if test_df is not None:
        test_df = calculate_severity_scores(test_df)

    # 6. Handle categorical variables
    categorical_cols = ["Gender", "Unit1", "Unit2"]
    existing_cats = [col for col in categorical_cols if col in train_df.columns]

    if existing_cats:
        logger.info("Encoding categorical variables")
        # Encode training data and get categories
        train_df, categories = encode_categorical_variables(
            train_df, existing_cats, is_training=True
        )

        # Encode validation and test data using training categories
        if val_df is not None:
            val_df = encode_categorical_variables(
                val_df, existing_cats, is_training=False, training_categories=categories
            )
        if test_df is not None:
            test_df = encode_categorical_variables(
                test_df,
                existing_cats,
                is_training=False,
                training_categories=categories,
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
