# main.py

import logging
import os
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
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


def train_and_evaluate_model(
    model_name: str,
    pipeline: ImbPipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    logger: logging.Logger,
) -> tuple[float, Dict[str, Any]]:
    """Train and evaluate a model, with proper logging."""
    logger.info(f"\nTraining {model_name}...")

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_val)

    # Calculate metrics
    metrics = {
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred),
        "Recall": recall_score(y_val, y_pred),
        "F1 Score": f1_score(y_val, y_pred),
        "AUC-ROC": roc_auc_score(y_val, y_pred),
        "Mean Absolute Error": mean_absolute_error(y_val, y_pred),
        "Root Mean Squared Error": np.sqrt(mean_squared_error(y_val, y_pred)),
    }

    # Log metrics
    logger.info(f"{model_name} Evaluation Results:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name:<20} : {value:.4f}")

    # Save evaluation artifacts without additional logging
    evaluate_model(
        y_val,
        y_pred,
        model_name=model_name.lower().replace(" ", "_"),
        logger=None,  # Prevent double logging from evaluate_model
    )

    # Log save completion
    logger.info(f"Saved evaluation artifacts for {model_name}\n")

    return metrics["F1 Score"], pipeline


def main():
    # Set up logger
    logger = setup_logger()
    logger.info("Starting Sepsis Prediction Pipeline")

    try:
        # Step 1: Load and split the data
        logger.info("Loading raw data")
        combined_df = load_data("data/raw/Dataset.csv")
        logger.info("Splitting data into training, validation, and testing sets")
        df_train, df_val, df_test = split_data(
            combined_df, train_size=0.7, val_size=0.15
        )

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

        # Step 4: Handle class imbalance (SMOTEENN) - Only on training data
        logger.info("Applying SMOTEENN to training data")

        from imblearn.combine import SMOTEENN

        smote_enn = SMOTEENN(sampling_strategy=0.4, random_state=42)
        X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

        logger.info(
            f"Original class distribution - Class 1: {y_train.sum()}, Class 0: {len(y_train) - y_train.sum()}"
        )
        logger.info(
            f"Resampled class distribution - Class 1: {y_train_resampled.sum()}, Class 0: {len(y_train_resampled) - y_train_resampled.sum()}"
        )

        # Step 5: Train and evaluate models
        models = {}
        best_score = 0
        best_model_name = None

        # 5.1 Random Forest
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
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        score, models["Random Forest"] = train_and_evaluate_model(
            "Random Forest",
            rf_pipeline,
            X_train_resampled,
            y_train_resampled,
            X_val,
            y_val,
            logger,
        )
        if score > best_score:
            best_score = score
            best_model_name = "Random Forest"

        # 5.2 Naive Bayes
        nb_pipeline = ImbPipeline([("scaler", StandardScaler()), ("nb", GaussianNB())])
        score, models["Naive Bayes"] = train_and_evaluate_model(
            "Naive Bayes",
            nb_pipeline,
            X_train_resampled,
            y_train_resampled,
            X_val,
            y_val,
            logger,
        )
        if score > best_score:
            best_score = score
            best_model_name = "Naive Bayes"

        # 5.3 KNN
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
        score, models["KNN"] = train_and_evaluate_model(
            "KNN",
            knn_pipeline,
            X_train_resampled,
            y_train_resampled,
            X_val,
            y_val,
            logger,
        )
        if score > best_score:
            best_score = score
            best_model_name = "KNN"

        # 5.4 Logistic Regression
        lr_model = train_logistic_regression(X_train_resampled, y_train_resampled)
        lr_pipeline = ImbPipeline([("scaler", StandardScaler()), ("lr", lr_model)])
        score, models["Logistic Regression"] = train_and_evaluate_model(
            "Logistic Regression",
            lr_pipeline,
            X_train_resampled,
            y_train_resampled,
            X_val,
            y_val,
            logger,
        )
        if score > best_score:
            best_score = score
            best_model_name = "Logistic Regression"

        # # 5.5 XGBoost
        # logger.info("Training XGBoost model")
        # xgb_params = {
        #     "max_depth": 6,
        #     "min_child_weight": 1,
        #     "eta": 0.05,  # Reduced learning rate
        #     "subsample": 0.8,
        #     "colsample_bytree": 0.8,
        #     "objective": "binary:logistic",
        #     "eval_metric": ["auc", "error"],
        #     "alpha": 1,
        #     "lambda": 1,
        #     "tree_method": "hist",
        #     "random_state": 42,
        # }

        # dtrain = xgb.DMatrix(X_train_resampled, label=y_train_resampled)
        # dval = xgb.DMatrix(X_val, label=y_val)
        # watchlist = [(dtrain, "train"), (dval, "eval")]

        # xgb_model = train_xgboost(
        #     X_train_resampled,
        #     y_train_resampled,
        #     params=xgb_params,
        #     num_round=200,  # Increased number of rounds
        #     eval_set=watchlist,
        # )

        # y_pred = predict_xgboost(xgb_model, X_val)
        # score = f1_score(y_val, y_pred)
        # models["XGBoost"] = xgb_model

        # if score > best_score:
        #     best_score = score
        #     best_model_name = "XGBoost"

        # evaluate_model(y_val, y_pred, model_name="xgboost", logger=logger)

        # 5.5 XGBoost
        try:
            logger.info("Training XGBoost model")

            # Convert data to numpy arrays
            X_train_array = (
                X_train_resampled.values
                if hasattr(X_train_resampled, "values")
                else X_train_resampled
            )
            y_train_array = (
                y_train_resampled.values
                if hasattr(y_train_resampled, "values")
                else y_train_resampled
            )
            X_val_array = X_val.values if hasattr(X_val, "values") else X_val
            y_val_array = y_val.values if hasattr(y_val, "values") else y_val

            # XGBoost parameters
            xgb_params = {
                "max_depth": 6,
                "min_child_weight": 1,
                "eta": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "binary:logistic",
                "eval_metric": ["auc", "error"],
                "alpha": 1,
                "lambda": 1,
                "tree_method": "hist",
                "random_state": 42,
            }

            # Create DMatrix objects
            dtrain = xgb.DMatrix(X_train_array, label=y_train_array)
            dval = xgb.DMatrix(X_val_array, label=y_val_array)

            # Prepare watchlist for evaluation
            watchlist = [(dtrain, "train"), (dval, "eval")]

            # Train XGBoost model
            xgb_model = xgb.train(
                params=xgb_params,
                dtrain=dtrain,
                num_boost_round=200,
                evals=watchlist,
                early_stopping_rounds=20,
                verbose_eval=10,
            )

            # Make predictions
            y_pred_proba = xgb_model.predict(dval)
            y_pred = (y_pred_proba > 0.5).astype(int)

            # Calculate metrics
            score = f1_score(y_val, y_pred)
            models["XGBoost"] = xgb_model

            # Update best model if necessary
            if score > best_score:
                best_score = score
                best_model_name = "XGBoost"

            # Log XGBoost specific metrics
            xgb_metrics = {
                "Accuracy": accuracy_score(y_val, y_pred),
                "Precision": precision_score(y_val, y_pred),
                "Recall": recall_score(y_val, y_pred),
                "F1 Score": score,
                "AUC-ROC": roc_auc_score(y_val, y_pred_proba),
                "Mean Absolute Error": mean_absolute_error(y_val, y_pred),
                "Root Mean Squared Error": np.sqrt(mean_squared_error(y_val, y_pred)),
            }

            logger.info("XGBoost Evaluation Results:")
            for metric_name, value in xgb_metrics.items():
                logger.info(f"  {metric_name:<20} : {value:.4f}")

            # Save feature importance plot
            importance_scores = xgb_model.get_score(importance_type="weight")
            importance_df = pd.DataFrame(
                {
                    "feature": list(importance_scores.keys()),
                    "importance": list(importance_scores.values()),
                }
            ).sort_values("importance", ascending=False)

            # Plot feature importance
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.bar(importance_df["feature"], importance_df["importance"])
            plt.xticks(rotation=45, ha="right")
            plt.title("XGBoost Feature Importance")
            plt.tight_layout()
            plt.savefig("reports/evaluations/xgboost_feature_importance.png")
            plt.close()

            # Save evaluation results
            evaluate_model(y_val, y_pred, model_name="xgboost", logger=logger)

        except Exception as e:
            logger.error(f"Error in XGBoost training: {str(e)}", exc_info=True)
            logger.info("Skipping XGBoost due to error")

        # # Step 6: Final evaluation on test set
        # logger.info(f"\nPerforming final evaluation with best model: {best_model_name}")
        # best_model = models[best_model_name]

        # if best_model_name == "XGBoost":
        #     final_predictions = predict_xgboost(best_model, X_test)
        # else:
        #     final_predictions = best_model.predict(X_test)

        # evaluate_model(
        #     y_test,
        #     final_predictions,
        #     model_name=f"final_{best_model_name.lower().replace(' ', '_')}",
        #     logger=logger,
        # )

        # Step 6: Final evaluation on test set
        logger.info(f"\nPerforming final evaluation with best model: {best_model_name}")
        best_model = models[best_model_name]

        if best_model_name == "XGBoost":
            # Convert test data to numpy array
            X_test_array = X_test.values if hasattr(X_test, "values") else X_test
            dtest = xgb.DMatrix(X_test_array)
            final_predictions_proba = best_model.predict(dtest)
            final_predictions = (final_predictions_proba > 0.5).astype(int)
        else:
            final_predictions = best_model.predict(X_test)

        evaluate_model(
            y_test,
            final_predictions,
            model_name=f"final_{best_model_name.lower().replace(' ', '_')}",
            logger=logger,
        )

        # Step 7: Save the best model
        logger.info(f"Saving the best model ({best_model_name})")
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(
            model_dir, f"best_model_{best_model_name.lower().replace(' ', '_')}.pkl"
        )
        joblib.dump(best_model, model_path)
        logger.info(f"Model saved to {model_path}")

        logger.info("Sepsis Prediction Pipeline completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
