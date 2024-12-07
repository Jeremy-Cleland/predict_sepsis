# src/feature_engineering.py

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler, StandardScaler


def validate_dataframe(df, stage_name=""):
    """Validate DataFrame has required columns and proper data types."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Input must be a pandas DataFrame, got {type(df)} instead")

    required_columns = {"Patient_ID", "Hour"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns at {stage_name}: {missing_columns}. Available columns: {df.columns.tolist()}"
        )

    return df


def create_vital_signs_features(df):
    """Create features from vital signs measurements."""
    df = validate_dataframe(df, "create_vital_signs_features")
    df = df.copy()

    vital_signs = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
    grouped = df.groupby("Patient_ID", group_keys=False)

    for vital in vital_signs:
        if vital in df.columns:
            # Calculate rolling statistics for each vital sign
            df[f"{vital}_rolling_mean_3"] = (
                grouped[vital]
                .rolling(window=3, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            df[f"{vital}_rolling_std_3"] = (
                grouped[vital]
                .rolling(window=3, min_periods=1)
                .std()
                .reset_index(0, drop=True)
            )

            df[f"{vital}_rate_change"] = grouped[vital].diff().reset_index(0, drop=True)
            df[f"{vital}_zscore"] = grouped[vital].transform(
                lambda x: stats.zscore(x, nan_policy="omit")
            )
            df[f"{vital}_abnormal"] = abs(df[f"{vital}_zscore"]) > 2

    return df


def create_lab_features(df):
    """Create features from laboratory measurements."""
    df = validate_dataframe(df, "create_lab_features")
    df = df.copy()

    lab_tests = [
        "BUN",
        "Creatinine",
        "Glucose",
        "Calcium",
        "Hct",
        "Hgb",
        "WBC",
        "Platelets",
    ]

    grouped = df.groupby("Patient_ID", group_keys=False)

    for lab in lab_tests:
        if lab in df.columns:
            df[f"{lab}_trend"] = grouped[lab].diff().reset_index(0, drop=True)
            df[f"{lab}_pct_change"] = (
                grouped[lab].pct_change().reset_index(0, drop=True)
            )

            # Flag critical values based on medical thresholds
            if lab == "WBC":
                df[f"{lab}_critical"] = (df[lab] < 4) | (df[lab] > 12)
            elif lab == "Platelets":
                df[f"{lab}_critical"] = (df[lab] < 150) | (df[lab] > 450)
            elif lab == "Hct":
                df[f"{lab}_critical"] = (df[lab] < 36) | (df[lab] > 50)

    return df


def create_temporal_features(df):
    """Create time-based features."""
    df = validate_dataframe(df, "create_temporal_features")
    df = df.copy()

    df["hour_of_day"] = df["Hour"] % 24
    df["time_of_day"] = pd.cut(
        df["hour_of_day"],
        bins=[0, 6, 12, 18, 24],
        labels=["night", "morning", "afternoon", "evening"],
    )

    grouped = df.groupby("Patient_ID", group_keys=False)
    df["time_in_hospital"] = grouped["Hour"].transform("max")
    df["normalized_time"] = df["Hour"] / df["time_in_hospital"]
    df["measurements_count"] = grouped["Hour"].transform("count")

    return df


def calculate_sofa_proxy(df):
    """Calculate a proxy SOFA score based on available measurements."""
    df = validate_dataframe(df, "calculate_sofa_proxy")
    df = df.copy()

    sofa_score = pd.DataFrame(index=df.index)

    if "O2Sat" in df.columns:
        sofa_score["resp"] = pd.cut(
            df["O2Sat"], bins=[0, 91, 94, 97, 101], labels=[3, 2, 1, 0]
        )

    if "MAP" in df.columns:
        sofa_score["cv"] = pd.cut(
            df["MAP"], bins=[0, 60, 70, 80, 300], labels=[3, 2, 1, 0]
        )

    if "Platelets" in df.columns:
        sofa_score["coag"] = pd.cut(
            df["Platelets"], bins=[0, 50, 100, 150, 1500], labels=[3, 2, 1, 0]
        )

    if "Creatinine" in df.columns:
        sofa_score["renal"] = pd.cut(
            df["Creatinine"], bins=[0, 1.2, 2.0, 3.5, 30], labels=[0, 1, 2, 3]
        )

    df["sofa_proxy"] = sofa_score.mean(axis=1) * sofa_score.count(axis=1)
    return df


def handle_missing_values(df):
    """Enhanced missing value handling with improved error checking."""
    df = validate_dataframe(df, "handle_missing_values")
    df = df.copy()

    try:
        # Forward fill within patient groups
        df = df.groupby("Patient_ID", group_keys=False).ffill()

        # Backward fill for remaining missing values
        df = df.groupby("Patient_ID", group_keys=False).bfill()

        # Fill remaining missing values with medians
        if "Gender" in df.columns and "Age" in df.columns:
            # Fill missing ages with median first
            df["Age"] = df["Age"].fillna(df["Age"].median())

            # Group by gender and age quartiles for more accurate imputation
            age_bins = pd.qcut(df["Age"], q=4)
            df = df.groupby(["Gender", age_bins], group_keys=False).transform(
                lambda x: x.fillna(x.median())
                if x.dtype.kind in "biufc"
                else x.fillna(x.mode().iloc[0])
            )
        else:
            # Fallback if Gender or Age is not available
            df = df.transform(
                lambda x: x.fillna(x.median())
                if x.dtype.kind in "biufc"
                else x.fillna(x.mode().iloc[0])
            )

    except Exception as e:
        raise ValueError(f"Error in handle_missing_values: {str(e)}")

    return df


def identify_high_risk_conditions(df):
    """Create features for high-risk medical conditions."""
    df = validate_dataframe(df, "identify_high_risk_conditions")
    df = df.copy()

    if "Temp" in df.columns:
        df["high_fever"] = df["Temp"] > 38.5

    if "HR" in df.columns:
        df["tachycardia"] = df["HR"] > 100

    if "MAP" in df.columns:
        df["hypotension"] = df["MAP"] < 65

    if "Creatinine" in df.columns:
        df["kidney_dysfunction"] = df["Creatinine"] > 1.5

    if "Platelets" in df.columns:
        df["coagulation_dysfunction"] = df["Platelets"] < 100

    risk_columns = [
        "high_fever",
        "tachycardia",
        "hypotension",
        "kidney_dysfunction",
        "coagulation_dysfunction",
    ]

    # Only include risk columns that exist in the DataFrame
    available_risk_columns = [col for col in risk_columns if col in df.columns]
    df["risk_score"] = df[available_risk_columns].sum(axis=1)

    return df


def preprocess_data(df):
    """Enhanced preprocessing pipeline with improved error handling."""
    try:
        # Initial validation
        df = validate_dataframe(df, "initial preprocessing")
        original_patient_ids = set(df["Patient_ID"].unique())
        df = df.copy()

        # Handle missing values
        df = handle_missing_values(df)

        # Create features
        df = create_vital_signs_features(df)
        df = create_lab_features(df)
        df = create_temporal_features(df)
        df = calculate_sofa_proxy(df)
        df = identify_high_risk_conditions(df)

        # One-hot encode categorical variables
        categorical_columns = ["Gender", "time_of_day"]
        for col in categorical_columns:
            if col in df.columns:
                df = pd.get_dummies(df, columns=[col], prefix=col)

        # Log transform skewed numerical features
        skewed_columns = ["BUN", "Creatinine", "Glucose", "WBC", "Platelets"]
        for col in skewed_columns:
            if col in df.columns:
                # Add small constant to handle zeros
                df[col] = np.log1p(df[col])

        # Scale numerical features
        numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns
        # Exclude Patient_ID and SepsisLabel from scaling
        cols_to_scale = numerical_columns.difference(["Patient_ID", "SepsisLabel"])
        if len(cols_to_scale) > 0:
            scaler = RobustScaler()
            df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

        # Verify Patient_IDs are preserved
        current_patient_ids = set(df["Patient_ID"].unique())
        if current_patient_ids != original_patient_ids:
            raise ValueError(
                f"Patient_ID mismatch after preprocessing. "
                f"Lost IDs: {original_patient_ids - current_patient_ids}, "
                f"New IDs: {current_patient_ids - original_patient_ids}"
            )

        # Final cleaning
        df = df.dropna()
        df.columns = df.columns.astype(str)

        return df

    except Exception as e:
        raise RuntimeError(f"Error in preprocessing pipeline: {str(e)}")


# src/feature_engineering.py

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler

# Constants for clinical variables
VITAL_SIGNS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LAB_TESTS = [
    "BUN",
    "Creatinine",
    "Glucose",
    "Calcium",
    "Hct",
    "Hgb",
    "WBC",
    "Platelets",
]


def create_missingness_features(df, columns):
    """Create features based on missingness patterns."""
    grouped = df.groupby("Patient_ID")

    for col in columns:
        if col in df.columns:
            # Measurement frequency
            df[f"{col}_measure_freq"] = grouped[col].transform(
                lambda x: x.notnull().cumsum()
            )

            # Time since last measurement
            df[f"{col}_time_since_last"] = grouped[col].transform(
                lambda x: x.isnull().cumsum()
            )

            # Difference from last measurement
            df[f"{col}_value_diff"] = grouped[col].transform(lambda x: x.diff())

    return df


def create_sliding_window_features(df, window_size=6):
    """Create sliding window statistics for vital signs."""
    grouped = df.groupby("Patient_ID")

    for vital in VITAL_SIGNS:
        if vital in df.columns:
            # Calculate rolling statistics
            df[f"{vital}_rolling_max"] = (
                grouped[vital].rolling(window_size, min_periods=1).max()
            )
            df[f"{vital}_rolling_min"] = (
                grouped[vital].rolling(window_size, min_periods=1).min()
            )
            df[f"{vital}_rolling_mean"] = (
                grouped[vital].rolling(window_size, min_periods=1).mean()
            )
            df[f"{vital}_rolling_std"] = (
                grouped[vital].rolling(window_size, min_periods=1).std()
            )
            df[f"{vital}_rolling_trend"] = (
                grouped[vital]
                .rolling(window_size, min_periods=2)
                .apply(lambda x: stats.linregress(range(len(x)), x)[0])
            )

    return df


def calculate_clinical_scores(df):
    """Calculate clinical scoring systems (NEWS, SOFA components, qSOFA)."""
    # NEWS Score components
    if "HR" in df.columns:
        df["HR_score"] = pd.cut(
            df["HR"],
            bins=[-np.inf, 40, 50, 90, 110, 130, np.inf],
            labels=[3, 1, 0, 1, 2, 3],
        )

    if "Temp" in df.columns:
        df["Temp_score"] = pd.cut(
            df["Temp"], bins=[-np.inf, 35, 36, 38, 39, np.inf], labels=[3, 1, 0, 1, 2]
        )

    if "Resp" in df.columns:
        df["Resp_score"] = pd.cut(
            df["Resp"], bins=[-np.inf, 8, 11, 20, 24, np.inf], labels=[3, 1, 0, 2, 3]
        )

    # SOFA Score components
    if "MAP" in df.columns:
        df["MAP_score"] = (df["MAP"] < 70).astype(int)

    if "Platelets" in df.columns:
        df["Platelet_score"] = pd.cut(
            df["Platelets"], bins=[-np.inf, 50, 100, 150, np.inf], labels=[3, 2, 1, 0]
        )

    if "Creatinine" in df.columns:
        df["Creatinine_score"] = pd.cut(
            df["Creatinine"], bins=[-np.inf, 1.2, 2.0, 3.5, np.inf], labels=[0, 1, 2, 3]
        )

    # qSOFA components
    if all(x in df.columns for x in ["SBP", "Resp"]):
        df["qSOFA"] = ((df["SBP"] <= 100) & (df["Resp"] >= 22)).astype(int)

    return df


def handle_missing_values(df):
    """Enhanced missing value handling with forward fill and grouping."""
    # Forward fill within patient groups
    df = df.groupby("Patient_ID").ffill()

    # For any remaining missing values, fill with median of similar patients
    df = df.groupby(["Gender", pd.qcut(df["Age"], q=4)]).transform(
        lambda x: x.fillna(x.median())
    )

    return df


def create_interaction_features(df):
    """Create interaction features between vital signs and lab values."""
    if all(x in df.columns for x in ["MAP", "Lactate"]):
        df["shock_index"] = df["Lactate"] / df["MAP"]

    if all(x in df.columns for x in ["WBC", "Temp"]):
        df["infection_index"] = df["WBC"] * df["Temp"]

    if all(x in df.columns for x in ["BUN", "Creatinine"]):
        df["bun_creatinine_ratio"] = df["BUN"] / df["Creatinine"]

    return df


def preprocess_data(df):
    """Complete preprocessing pipeline combining all approaches."""
    # Make a copy to avoid modifying original data
    df = df.copy()

    # Handle missing values first
    df = handle_missing_values(df)

    # Create missingness features
    df = create_missingness_features(df, VITAL_SIGNS + LAB_TESTS)

    # Create sliding window features for vital signs
    df = create_sliding_window_features(df)

    # Calculate clinical scores
    df = calculate_clinical_scores(df)

    # Create interaction features
    df = create_interaction_features(df)

    # One-hot encode categorical variables
    categorical_columns = (
        ["Gender", "time_of_day"] if "time_of_day" in df.columns else ["Gender"]
    )
    df = pd.get_dummies(df, columns=categorical_columns)

    # Log transform highly skewed numerical features
    skewed_features = ["BUN", "Creatinine", "Glucose", "WBC", "Platelets"]
    for col in skewed_features:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    # Scale numerical features
    numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns
    scaler = RobustScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Drop unnecessary columns
    columns_to_drop = ["Unit1", "Unit2", "Patient_ID"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # Final cleaning
    df = df.dropna()
    df.columns = df.columns.astype(str)

    return df
