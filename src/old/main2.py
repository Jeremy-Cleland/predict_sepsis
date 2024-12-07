# main.py

import logging
import os

import joblib
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier  # Add this import
from sklearn.preprocessing import StandardScaler

from src import (
    evaluate_model,
    load_data,
    predict_xgboost,
    preprocess_data,
    split_data,
    train_logistic_regression,
    train_xgboost,
)
from src.utils import setup_logger


def main():
    # Set up logger
    logger = setup_logger()
    logger.info("Starting Sepsis Prediction Pipeline")

    # Step 1: Load and split the data into train/val/test
    logger.info("Loading raw data")
    combined_df = load_data("data/raw/Dataset.csv")
    logger.info("Splitting data into training, validation, and testing sets")
    df_train, df_val, df_test = split_data(combined_df, train_size=0.7, val_size=0.15)

    # Step 2: Preprocess all datasets
    logger.info("Preprocessing all datasets")
    df_train_processed = preprocess_data(df_train)
    df_val_processed = preprocess_data(df_val)
    df_test_processed = preprocess_data(df_test)

    # Step 3: Prepare features and targets
    X_train = df_train_processed.drop("SepsisLabel", axis=1)
    y_train = df_train_processed["SepsisLabel"]
    X_val = df_val_processed.drop("SepsisLabel", axis=1)
    y_val = df_val_processed["SepsisLabel"]
    X_test = df_test_processed.drop("SepsisLabel", axis=1)
    y_test = df_test_processed["SepsisLabel"]

    # Step 4: Handle class imbalance (SMOTE) - Only on training data
    logger.info("Applying SMOTE to training data")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    logger.info(
        f"Original class distribution - Class 1: {y_train.sum()}, Class 0: {len(y_train) - y_train.sum()}"
    )
    logger.info(
        f"Resampled class distribution - Class 1: {y_train_resampled.sum()}, Class 0: {len(y_train_resampled) - y_train_resampled.sum()}"
    )

    # Step 5: Train and evaluate models
    models = {}
    predictions = {}

    # 5.1 Random Forest with Pipeline
    logger.info("Training Random Forest model")
    from sklearn.ensemble import RandomForestClassifier

    rf_pipeline = ImbPipeline(
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
    rf_pipeline.fit(X_train_resampled, y_train_resampled)
    models["random_forest"] = rf_pipeline
    predictions["random_forest"] = rf_pipeline.predict(X_val)

    # 5.2 Naive Bayes
    logger.info("Training Naive Bayes model")
    nb_pipeline = ImbPipeline([("scaler", StandardScaler()), ("nb", GaussianNB())])
    nb_pipeline.fit(X_train_resampled, y_train_resampled)
    models["naive_bayes"] = nb_pipeline
    predictions["naive_bayes"] = nb_pipeline.predict(X_val)

    # 5.3 KNN - Fixed pipeline implementation
    logger.info("Training KNN model")
    knn_pipeline = ImbPipeline(
        [
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=5,
                    weights="distance",
                    algorithm="auto",
                    leaf_size=30,
                    p=2,
                    n_jobs=-1,
                    metric="minkowski",
                ),
            ),
        ]
    )
    knn_pipeline.fit(X_train_resampled, y_train_resampled)
    models["knn"] = knn_pipeline
    predictions["knn"] = knn_pipeline.predict(X_val)

    # 5.4 Logistic Regression
    logger.info("Training Logistic Regression model")
    lr_model = train_logistic_regression(X_train_resampled, y_train_resampled)
    lr_pipeline = ImbPipeline([("scaler", StandardScaler()), ("lr", lr_model)])
    lr_pipeline.fit(X_train_resampled, y_train_resampled)
    models["logistic_regression"] = lr_pipeline
    predictions["logistic_regression"] = lr_pipeline.predict(X_val)

    # 5.5 XGBoost
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

    # Create validation dataset for XGBoost
    dtrain = xgb.DMatrix(X_train_resampled, label=y_train_resampled)
    dval = xgb.DMatrix(X_val, label=y_val)
    watchlist = [(dtrain, "train"), (dval, "eval")]

    xgb_model = train_xgboost(
        X_train_resampled,
        y_train_resampled,
        params=xgb_params,
        num_round=100,
        eval_set=watchlist,
    )
    models["xgboost"] = xgb_model
    predictions["xgboost"] = predict_xgboost(xgb_model, X_val)

    # Step 6: Evaluate all models on validation set
    best_score = 0
    best_model_name = None

    for model_name, y_pred in predictions.items():
        logger.info(f"\nEvaluating {model_name}")
        score = evaluate_model(y_val, y_pred, model_name=model_name, logger=logger)
        if score > best_score:
            best_score = score
            best_model_name = model_name

    # Step 7: Final evaluation on test set
    logger.info(f"\nPerforming final evaluation with best model: {best_model_name}")
    best_model = models[best_model_name]
    final_predictions = best_model.predict(X_test)
    evaluate_model(
        y_test, final_predictions, model_name=f"Final_{best_model_name}", logger=logger
    )

    # Step 8: Save the best model
    logger.info(f"Saving the best model ({best_model_name})")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"best_model_{best_model_name}.pkl")
    joblib.dump(best_model, model_path)
    logger.info(f"Model saved to {model_path}")

    logger.info("Sepsis Prediction Pipeline completed successfully.")


if __name__ == "__main__":
    main()
