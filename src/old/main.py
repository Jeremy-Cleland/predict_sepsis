# main.py

import os

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import (
    evaluate_model,
    get_data_ready,
    load_data,
    load_processed_data,
    predict_xgboost,
    preprocess_data,
    setup_logger,
    split_data,
    train_knn,
    train_logistic_regression,
    train_naive_bayes,
    train_random_forest,
    train_xgboost,
)


def main():
    # Set up logger
    logger = setup_logger()
    logger.info("Starting Sepsis Prediction Pipeline")

    # Step 1: Load and split the data
    logger.info("Loading raw data")
    combined_df = load_data("data/raw/Dataset.csv")
    logger.info("Splitting data into training and testing sets")
    df_train, df_test = split_data(combined_df)

    # Step 2: Load processed data
    logger.info("Loading processed training and testing data")
    df_train, df_test = load_processed_data(
        "data/processed/data_part1.csv", "data/processed/data_part2.csv"
    )

    # Step 3: Preprocess training data
    logger.info("Preprocessing training data")
    df_train_impute = preprocess_data(df_train)

    # Step 4: Handle class imbalance (Oversampling with SMOTE)
    X = df_train_impute.drop("SepsisLabel", axis=1)
    y = df_train_impute["SepsisLabel"]

    logger.info(f"Number of SepsisLabel=1: {y.sum()}")
    logger.info(f"Number of SepsisLabel=0: {len(y) - y.sum()}")

    # Initialize SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    logger.info(f"After SMote, number of SepsisLabel=1: {y_resampled.sum()}")
    logger.info(
        f"After SMOTE, number of SepsisLabel=0: {len(y_resampled) - y_resampled.sum()}"
    )

    # Step 5: Train-Test Split
    logger.info("Splitting data into training and validation sets")
    X_train, X_val, y_train, y_val = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42
    )
    logger.debug("Converted all column names to strings")

    # Step 6: Train Models using Pipelines
    # 6.1 Random Forest with Pipeline
    logger.info("Training Random Forest model with Pipeline")
    pipeline = ImbPipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=20,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    evaluate_model(y_val, y_pred, model_name="Random_Forest", logger=logger)

    # 6.2 Naive Bayes
    logger.info("Training Naive Bayes model")
    nb_model = train_naive_bayes(X_train, y_train)
    nb_predictions = nb_model.predict(X_val)
    evaluate_model(y_val, nb_predictions, model_name="Naive_Bayes", logger=logger)

    # 6.3 K-Nearest Neighbors
    logger.info("Training K-Nearest Neighbors model")
    knn_model = train_knn(X_train, y_train, n_neighbors=5)  # Changed from 10 to 5
    knn_predictions = knn_model.predict(X_val)
    evaluate_model(y_val, knn_predictions, model_name="KNN", logger=logger)

    # 6.4 Logistic Regression
    logger.info("Training Logistic Regression model")
    lr_model = train_logistic_regression(X_train, y_train)
    lr_predictions = lr_model.predict(X_val)
    evaluate_model(
        y_val, lr_predictions, model_name="Logistic_Regression", logger=logger
    )

    # 6.5 XGBoost
    logger.info("Training XGBoost model")
    xgb_params = {
        "max_depth": 6,
        "min_child_weight": 1,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": ["auc", "error"],
        "alpha": 1,
        "lambda": 1,
        "tree_method": "hist",
        "random_state": 42,
    }
    xgb_model = train_xgboost(X_train, y_train, params=xgb_params, num_round=100)
    xgb_predictions = predict_xgboost(xgb_model, X_val)
    evaluate_model(y_val, xgb_predictions, model_name="XGBoost", logger=logger)

    # Step 7: Select Best Model (Assuming Random Forest performed best)
    logger.info("Selecting the best model (Random Forest)")
    best_model = pipeline  # Using the trained pipeline

    # Step 8: Test on Test Data
    logger.info("Preprocessing test data")
    df_test_preprocessed = get_data_ready(df_test)
    X_test_final = df_test_preprocessed.drop("SepsisLabel", axis=1)
    y_test_final = df_test_preprocessed["SepsisLabel"]

    # Ensure column alignment
    X_test_final = X_test_final[X_train.columns]
    logger.debug("Aligned test data columns with training data")

    # Make Predictions
    logger.info("Making predictions on test data")
    final_predictions = best_model.predict(X_test_final)

    # Evaluate
    evaluate_model(
        y_test_final,
        final_predictions,
        model_name="Final_Model_Test_Data",
        logger=logger,
    )

    # Step 9: Save the Best Model
    logger.info("Saving the best model (Random Forest Pipeline)")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "random_forest_pipeline.pkl")
    joblib.dump(best_model, model_path)
    logger.info(f"Model saved to {model_path}")

    logger.info("Sepsis Prediction Pipeline completed successfully.")


if __name__ == "__main__":
    main()
