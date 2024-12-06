# src/data_processing.py

import pandas as pd


def load_data(filepath):
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath)


def split_data(combined_df):
    """Split the combined data into training and testing datasets based on Patient_ID length."""
    rows_to_drop_train = combined_df.loc[
        combined_df["Patient_ID"].apply(lambda x: len(str(x)) == 6)
    ]
    df_train = combined_df.drop(rows_to_drop_train.index)
    df_train.to_csv("data/processed/data_part1.csv", index=False)

    rows_to_drop_test = combined_df.loc[
        combined_df["Patient_ID"].apply(lambda x: len(str(x)) != 6)
    ]
    df_test = combined_df.drop(rows_to_drop_test.index)
    df_test.to_csv("data/processed/data_part2.csv", index=False)

    return df_train, df_test


def load_processed_data(train_path, test_path):
    """Load processed training and testing data."""
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test


def get_data_ready(df):
    """Transform the dataframe into a format compatible with the model."""
    from .feature_engineering import (
        preprocess_data,
    )  # Local import to prevent circular dependency

    df = preprocess_data(df)
    return df
