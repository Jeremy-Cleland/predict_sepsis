# # main.py

# import gc
# import logging
# import os
# from typing import Any

# import joblib
# import numpy as np
# import pandas as pd
# import xgboost as xgb
# from imblearn.combine import SMOTEENN
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline as ImbPipeline
# from imblearn.under_sampling import EditedNearestNeighbours
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (
#     accuracy_score,
#     f1_score,
#     mean_absolute_error,
#     mean_squared_error,
#     precision_score,
#     recall_score,
#     roc_auc_score,
# )
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler

# from src import (
#     evaluate_model,
#     load_data,
#     preprocess_data,
#     split_data,
#     train_logistic_regression,
# )
# from src.evaluation import (
#     plot_class_distribution,
#     plot_cumulative_gains,
#     plot_feature_correlation_heatmap,
#     plot_learning_curves,
# )
# from src.utils import setup_logger


# def train_and_evaluate_model(
#     model_name: str,
#     pipeline: ImbPipeline,
#     X_train: pd.DataFrame,
#     y_train: pd.Series,
#     X_val: pd.DataFrame,
#     y_val: pd.Series,
#     logger: logging.Logger,
# ) -> tuple[float, Any]:
#     """Train and evaluate a model, with proper logging and plotting."""
#     logger.info(f"\nTraining {model_name}...")

#     try:
#         # Train the model
#         pipeline.fit(X_train, y_train)

#         # Make predictions
#         y_pred = pipeline.predict(X_val)

#         # Get predicted probabilities if available
#         y_pred_proba = None
#         if hasattr(pipeline, "predict_proba"):
#             y_pred_proba = pipeline.predict_proba(X_val)[:, 1]

#         # Calculate metrics
#         metrics = {
#             "Accuracy": accuracy_score(y_val, y_pred),
#             "Precision": precision_score(y_val, y_pred, zero_division=0),
#             "Recall": recall_score(y_val, y_pred, zero_division=0),
#             "F1 Score": f1_score(y_val, y_pred, zero_division=0),
#             "AUC-ROC": roc_auc_score(y_val, y_pred_proba)
#             if y_pred_proba is not None
#             else None,
#             "Mean Absolute Error": mean_absolute_error(y_val, y_pred),
#             "Root Mean Squared Error": np.sqrt(mean_squared_error(y_val, y_pred)),
#         }

#         # Log metrics
#         logger.info(f"{model_name} Evaluation Results:")
#         for metric_name, value in metrics.items():
#             if value is not None:
#                 logger.info(f"  {metric_name:<25} : {value:.4f}")
#             else:
#                 logger.info(f"  {metric_name:<25} : Not Available")

#         # Save evaluation artifacts without additional logging
#         # Pass the trained model for feature importance and SHAP
#         model = pipeline.named_steps.get(model_name.lower().replace(" ", "_"))

#         evaluate_model(
#             y_val,
#             y_pred,
#             model_name=model_name.lower().replace(" ", "_"),
#             logger=None,  # Prevent double logging from evaluate_model
#             y_pred_proba=y_pred_proba,
#             model=model,
#         )

#         # Plot Learning Curves
#         plot_learning_curves(
#             estimator=pipeline,
#             X=X_train,
#             y=y_train,
#             model_name=model_name,
#             report_dir="reports/evaluations",
#             cv=5,
#             scoring="f1",
#         )

#         # Log save completion
#         logger.info(f"Saved evaluation artifacts for {model_name}\n")

#         return metrics["F1 Score"], model

#     finally:
#         # Clear memory by deleting the pipeline and invoking garbage collection
#         del pipeline
#         gc.collect()


# def main():
#     # Set up logger
#     logger = setup_logger()
#     logger.info("Starting Sepsis Prediction Pipeline")

#     try:
#         # Step 1: Load and split the data
#         logger.info("Loading raw data")
#         combined_df = load_data("data/raw/Dataset.csv")
#         logger.info("Splitting data into training, validation, and testing sets")
#         df_train, df_val, df_test = split_data(
#             combined_df, train_size=0.7, val_size=0.15
#         )

#         # Plot Class Distribution before resampling
#         plot_class_distribution(
#             y=df_train["SepsisLabel"],
#             model_name="Original Training Data",
#             report_dir="reports/evaluations",
#             title_suffix="_before_resampling",
#         )

#         # Step 2: Preprocess all datasets
#         logger.info("Preprocessing all datasets")
#         df_train_processed = preprocess_data(df_train)
#         del df_train
#         gc.collect()
#         df_val_processed = preprocess_data(df_val)
#         del df_val
#         gc.collect()
#         df_test_processed = preprocess_data(df_test)
#         del df_test
#         gc.collect()

#         # Plot Class Distribution after resampling will be done after resampling

#         # Step 3: Prepare features and targets
#         X_train = df_train_processed.drop("SepsisLabel", axis=1)
#         y_train = df_train_processed["SepsisLabel"]
#         X_val = df_val_processed.drop("SepsisLabel", axis=1)
#         y_val = df_val_processed["SepsisLabel"]
#         X_test = df_test_processed.drop("SepsisLabel", axis=1)
#         y_test = df_test_processed["SepsisLabel"]

#         # Plot Feature Correlation Heatmap
#         plot_feature_correlation_heatmap(
#             df=X_train, model_name="Training Data", report_dir="reports/evaluations"
#         )

#         # Step 4: Handle class imbalance (SMOTEENN) - Only on training data
#         logger.info("Applying SMOTEENN to training data")

#         # Initialize KNN for SMOTE
#         knn_smote = KNeighborsClassifier(n_jobs=-1)

#         # Configure SMOTE with the KNN estimator
#         smote = SMOTE(sampling_strategy=0.3, random_state=42, k_neighbors=knn_smote)

#         # Configure ENN
#         enn = EditedNearestNeighbours(n_jobs=-1)

#         # Combine them in SMOTEENN
#         smote_enn = SMOTEENN(smote=smote, enn=enn, random_state=42)

#         # Resample the training data
#         X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

#         # Plot Class Distribution after resampling
#         plot_class_distribution(
#             y=y_train_resampled,
#             model_name="Resampled Training Data",
#             report_dir="reports/evaluations",
#             title_suffix="_after_resampling",
#         )

#         logger.info(
#             f"Original class distribution - Class 1: {y_train.sum()}, Class 0: {len(y_train) - y_train.sum()}"
#         )
#         logger.info(
#             f"Resampled class distribution - Class 1: {y_train_resampled.sum()}, Class 0: {len(y_train_resampled) - y_train_resampled.sum()}"
#         )

#         # Step 5: Train and evaluate models
#         models = {}
#         best_score = 0
#         best_model_name = None

#         # 5.1 Random Forest
#         rf_pipeline = ImbPipeline(
#             [
#                 ("scaler", StandardScaler()),
#                 (
#                     "random_forest",
#                     RandomForestClassifier(
#                         n_estimators=100,
#                         max_depth=20,
#                         random_state=42,
#                         class_weight="balanced",
#                         n_jobs=-1,
#                     ),
#                 ),
#             ]
#         )
#         score, models["Random Forest"] = train_and_evaluate_model(
#             "Random Forest",
#             rf_pipeline,
#             X_train_resampled,
#             y_train_resampled,
#             X_val,
#             y_val,
#             logger,
#         )
#         if score > best_score:
#             best_score = score
#             best_model_name = "Random Forest"

#         # After training Random Forest, delete the pipeline and collect garbage
#         del rf_pipeline
#         gc.collect()

#         # 5.2 Naive Bayes
#         nb_pipeline = ImbPipeline(
#             [("scaler", StandardScaler()), ("naive_bayes", GaussianNB())]
#         )
#         score, models["Naive Bayes"] = train_and_evaluate_model(
#             "Naive Bayes",
#             nb_pipeline,
#             X_train_resampled,
#             y_train_resampled,
#             X_val,
#             y_val,
#             logger,
#         )
#         if score > best_score:
#             best_score = score
#             best_model_name = "Naive Bayes"

#         # After training Naive Bayes, delete the pipeline and collect garbage
#         del nb_pipeline
#         gc.collect()

#         # 5.3 KNN
#         knn_pipeline = ImbPipeline(
#             [
#                 ("scaler", StandardScaler()),
#                 (
#                     "knn",
#                     KNeighborsClassifier(
#                         n_neighbors=5,
#                         weights="distance",
#                         algorithm="auto",
#                         leaf_size=30,
#                         p=2,
#                         n_jobs=-1,
#                         metric="minkowski",
#                     ),
#                 ),
#             ]
#         )
#         score, models["KNN"] = train_and_evaluate_model(
#             "KNN",
#             knn_pipeline,
#             X_train_resampled,
#             y_train_resampled,
#             X_val,
#             y_val,
#             logger,
#         )
#         if score > best_score:
#             best_score = score
#             best_model_name = "KNN"

#         # After training KNN, delete the pipeline and collect garbage
#         del knn_pipeline
#         gc.collect()

#         # 5.4 Logistic Regression
#         lr_model = train_logistic_regression(X_train_resampled, y_train_resampled)
#         lr_pipeline = ImbPipeline(
#             [("scaler", StandardScaler()), ("logistic_regression", lr_model)]
#         )
#         score, models["Logistic Regression"] = train_and_evaluate_model(
#             "Logistic Regression",
#             lr_pipeline,
#             X_train_resampled,
#             y_train_resampled,
#             X_val,
#             y_val,
#             logger,
#         )
#         if score > best_score:
#             best_score = score
#             best_model_name = "Logistic Regression"

#         # After training Logistic Regression, delete the pipeline and collect garbage
#         del lr_pipeline
#         gc.collect()

#         # 5.5 XGBoost
#         try:
#             logger.info("Training XGBoost model")

#             # Convert data to numpy arrays using .to_numpy() for consistency
#             X_train_array = X_train_resampled.to_numpy()
#             y_train_array = y_train_resampled.to_numpy()
#             X_val_array = X_val.to_numpy()
#             y_val_array = y_val.to_numpy()

#             # XGBoost parameters
#             xgb_params = {
#                 "max_depth": 6,
#                 "min_child_weight": 1,
#                 "eta": 0.05,
#                 "subsample": 0.8,
#                 "colsample_bytree": 0.8,
#                 "objective": "binary:logistic",
#                 "eval_metric": ["auc", "error"],
#                 "alpha": 1,
#                 "lambda": 1,
#                 "tree_method": "hist",
#                 "random_state": 42,
#             }

#             # Create DMatrix objects
#             dtrain = xgb.DMatrix(X_train_array, label=y_train_array)
#             dval = xgb.DMatrix(X_val_array, label=y_val_array)

#             # Prepare watchlist for evaluation
#             watchlist = [(dtrain, "train"), (dval, "eval")]

#             # Train XGBoost model
#             xgb_model = xgb.train(
#                 params=xgb_params,
#                 dtrain=dtrain,
#                 num_boost_round=200,
#                 evals=watchlist,
#                 early_stopping_rounds=20,
#                 verbose_eval=False,  # Suppress training logs
#             )

#             # Make predictions
#             y_pred_proba = xgb_model.predict(dval)
#             y_pred = (y_pred_proba > 0.5).astype(int)

#             # Calculate metrics (only F1 Score for selection)
#             score = f1_score(y_val, y_pred)
#             models["XGBoost"] = xgb_model

#             # Update best model if necessary
#             if score > best_score:
#                 best_score = score
#                 best_model_name = "XGBoost"

#             # Log XGBoost specific metrics using evaluate_model
#             evaluate_model(
#                 y_val,
#                 y_pred,
#                 model_name="xgboost",
#                 logger=None,
#                 y_pred_proba=y_pred_proba,
#                 model=xgb_model,
#             )

#             # Plot Learning Curves for XGBoost (if applicable)
#             # Note: Learning Curves for boosting models can be time-consuming
#             # Consider limiting the number of training examples or iterations
#             # plot_learning_curves(...)  # Implement if desired

#             # Log the artifact saving
#             logger.info(f"Saved evaluation artifacts for XGBoost\n")

#             # After training XGBoost, delete DMatrix objects and model if not needed
#             del dtrain, dval, xgb_model
#             gc.collect()

#         except Exception as e:
#             logger.error(f"Error in XGBoost training: {str(e)}", exc_info=True)
#             logger.info("Skipping XGBoost due to error")

#         # Step 6: Final evaluation on test set
#         logger.info(f"\nPerforming final evaluation with best model: {best_model_name}")
#         best_model = models[best_model_name]

#         if best_model_name == "XGBoost":
#             # Convert test data to numpy array using .to_numpy()
#             X_test_array = X_test.to_numpy()
#             dtest = xgb.DMatrix(X_test_array)
#             final_predictions_proba = best_model.predict(dtest)
#             final_predictions = (final_predictions_proba > 0.5).astype(int)

#             # Clean up DMatrix
#             del dtest
#             gc.collect()

#             # Evaluate using y_pred_proba
#             evaluate_model(
#                 y_test,
#                 final_predictions,
#                 model_name=f"final_{best_model_name.lower().replace(' ', '_')}",
#                 logger=logger,
#                 y_pred_proba=final_predictions_proba,
#                 model=best_model,
#             )
#         else:
#             final_predictions = best_model.predict(X_test)

#             # Evaluate without y_pred_proba
#             evaluate_model(
#                 y_test,
#                 final_predictions,
#                 model_name=f"final_{best_model_name.lower().replace(' ', '_')}",
#                 logger=logger,
#                 model=best_model,
#             )

#         # Step 7: Save the best model
#         logger.info(f"Saving the best model ({best_model_name})")
#         model_dir = "models"
#         os.makedirs(model_dir, exist_ok=True)
#         model_path = os.path.join(
#             model_dir, f"best_model_{best_model_name.lower().replace(' ', '_')}.pkl"
#         )
#         joblib.dump(best_model, model_path)
#         logger.info(f"Model saved to {model_path}")

#         # Optionally, delete the best model from memory if no longer needed
#         del best_model
#         gc.collect()

#         logger.info("Sepsis Prediction Pipeline completed successfully.")

#     except Exception as e:
#         logger.error(f"An error occurred: {str(e)}", exc_info=True)
#         raise


# if __name__ == "__main__":
#     main()


# main.py

import gc
import logging
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import EditedNearestNeighbours
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
    preprocess_data,
    split_data,
    train_logistic_regression,
)
from src.evaluation import plot_class_distribution, plot_feature_correlation_heatmap
from src.utils import setup_logger


def train_and_evaluate_model(
    model_name: str,
    pipeline: ImbPipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    logger: logging.Logger,
) -> tuple[float, Any]:
    """Train and evaluate a model, with proper logging and plotting."""
    logger.info(f"\nTraining {model_name}...")

    try:
        # Train the model
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred = pipeline.predict(X_val)

        # Get predicted probabilities if available
        y_pred_proba = None
        if hasattr(pipeline, "predict_proba"):
            y_pred_proba = pipeline.predict_proba(X_val)[:, 1]

        # Calculate metrics
        metrics = {
            "Accuracy": accuracy_score(y_val, y_pred),
            "Precision": precision_score(y_val, y_pred, zero_division=0),
            "Recall": recall_score(y_val, y_pred, zero_division=0),
            "F1 Score": f1_score(y_val, y_pred, zero_division=0),
            "AUC-ROC": roc_auc_score(y_val, y_pred_proba)
            if y_pred_proba is not None
            else None,
            "Mean Absolute Error": mean_absolute_error(y_val, y_pred),
            "Root Mean Squared Error": np.sqrt(mean_squared_error(y_val, y_pred)),
        }

        # Log metrics
        logger.info(f"{model_name} Evaluation Results:")
        for metric_name, value in metrics.items():
            if value is not None:
                logger.info(f"  {metric_name:<25} : {value:.4f}")
            else:
                logger.info(f"  {metric_name:<25} : Not Available")

        # Save evaluation artifacts without additional logging
        # Pass the trained model for feature importance and SHAP
        model = pipeline.named_steps.get(model_name.lower().replace(" ", "_"))

        # Assuming X_val is the feature set for SHAP plots
        evaluate_model(
            y_val,
            y_pred,
            model_name=model_name.lower().replace(" ", "_"),
            logger=None,  # Prevent double logging from evaluate_model
            y_pred_proba=y_pred_proba,
            model=model,
            X_features=X_val,  # Pass feature set for SHAP
        )

        # Log save completion
        logger.info(f"Saved evaluation artifacts for {model_name}\n")

        # Plot Learning Curves
        evaluate_model(
            y_val,
            y_pred,
            model_name=model_name.lower().replace(" ", "_"),
            report_dir="reports/evaluations",
            logger=None,
            y_pred_proba=y_pred_proba,
            model=model,
            X_features=X_val,  # Pass feature set for SHAP
        )

        return metrics["F1 Score"], model

    finally:
        # Clear memory by deleting the pipeline and invoking garbage collection
        del pipeline
        gc.collect()


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

        # Plot Class Distribution before resampling
        plot_class_distribution(
            y=df_train["SepsisLabel"],
            model_name="Original Training Data",
            report_dir="reports/evaluations",
            title_suffix="_before_resampling",
        )

        # Step 2: Preprocess all datasets
        logger.info("Preprocessing all datasets")
        df_train_processed = preprocess_data(df_train)
        del df_train
        gc.collect()
        df_val_processed = preprocess_data(df_val)
        del df_val
        gc.collect()
        df_test_processed = preprocess_data(df_test)
        del df_test
        gc.collect()

        # Step 3: Prepare features and targets
        X_train = df_train_processed.drop("SepsisLabel", axis=1)
        y_train = df_train_processed["SepsisLabel"]
        X_val = df_val_processed.drop("SepsisLabel", axis=1)
        y_val = df_val_processed["SepsisLabel"]
        X_test = df_test_processed.drop("SepsisLabel", axis=1)
        y_test = df_test_processed["SepsisLabel"]

        # Plot Feature Correlation Heatmap
        plot_feature_correlation_heatmap(
            df=X_train,
            model_name="Training Data",
            report_dir="reports/evaluations",
        )

        # Step 4: Handle class imbalance (SMOTEENN) - Only on training data
        logger.info("Applying SMOTEENN to training data")

        # Configure SMOTE with the correct k_neighbors parameter
        smote = SMOTE(sampling_strategy=0.3, random_state=42, k_neighbors=5)

        # Configure ENN
        enn = EditedNearestNeighbours(n_jobs=-1)

        # Combine them in SMOTEENN
        smote_enn = SMOTEENN(smote=smote, enn=enn, random_state=42)

        # Resample the training data
        X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

        # Plot Class Distribution after resampling
        plot_class_distribution(
            y=y_train_resampled,
            model_name="Resampled Training Data",
            report_dir="reports/evaluations",
            title_suffix="_after_resampling",
        )

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
                    "random_forest",
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

        # After training Random Forest, delete the pipeline and collect garbage
        del rf_pipeline
        gc.collect()

        # 5.2 Naive Bayes
        nb_pipeline = ImbPipeline(
            [("scaler", StandardScaler()), ("naive_bayes", GaussianNB())]
        )
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

        # After training Naive Bayes, delete the pipeline and collect garbage
        del nb_pipeline
        gc.collect()

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

        # After training KNN, delete the pipeline and collect garbage
        del knn_pipeline
        gc.collect()

        # 5.4 Logistic Regression
        lr_model = train_logistic_regression(X_train_resampled, y_train_resampled)
        lr_pipeline = ImbPipeline(
            [("scaler", StandardScaler()), ("logistic_regression", lr_model)]
        )
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

        # After training Logistic Regression, delete the pipeline and collect garbage
        del lr_pipeline
        gc.collect()

        # 5.5 XGBoost
        try:
            logger.info("Training XGBoost model")

            # Convert data to numpy arrays using .to_numpy() for consistency
            X_train_array = X_train_resampled.to_numpy()
            y_train_array = y_train_resampled.to_numpy()
            X_val_array = X_val.to_numpy()
            y_val_array = y_val.to_numpy()

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
                verbose_eval=False,  # Suppress training logs
            )

            # Make predictions
            y_pred_proba = xgb_model.predict(dval)
            y_pred = (y_pred_proba > 0.5).astype(int)

            # Calculate metrics (only F1 Score for selection)
            score = f1_score(y_val, y_pred)
            models["XGBoost"] = xgb_model

            # Update best model if necessary
            if score > best_score:
                best_score = score
                best_model_name = "XGBoost"

            # Log XGBoost specific metrics using evaluate_model
            evaluate_model(
                y_val,
                y_pred,
                model_name="xgboost",
                logger=None,
                y_pred_proba=y_pred_proba,
                model=xgb_model,
                X_features=X_val,  # Pass feature set for SHAP
            )

            # Log the artifact saving
            logger.info(f"Saved evaluation artifacts for XGBoost\n")

            # After training XGBoost, delete DMatrix objects and model if not needed
            del dtrain, dval, xgb_model
            gc.collect()

        except Exception as e:
            logger.error(f"Error in XGBoost training: {str(e)}", exc_info=True)
            logger.info("Skipping XGBoost due to error")

        # Step 6: Final evaluation on test set
        logger.info(f"\nPerforming final evaluation with best model: {best_model_name}")
        best_model = models[best_model_name]

        if best_model_name == "XGBoost":
            # Convert test data to numpy array using .to_numpy()
            X_test_array = X_test.to_numpy()
            dtest = xgb.DMatrix(X_test_array)
            final_predictions_proba = best_model.predict(dtest)
            final_predictions = (final_predictions_proba > 0.5).astype(int)

            # Clean up DMatrix
            del dtest
            gc.collect()

            # Evaluate using y_pred_proba
            evaluate_model(
                y_test,
                final_predictions,
                model_name=f"final_{best_model_name.lower().replace(' ', '_')}",
                logger=logger,
                y_pred_proba=final_predictions_proba,
                model=best_model,
                X_features=X_test,  # Pass feature set for SHAP
            )
        else:
            final_predictions = best_model.predict(X_test)

            # Evaluate without y_pred_proba
            evaluate_model(
                y_test,
                final_predictions,
                model_name=f"final_{best_model_name.lower().replace(' ', '_')}",
                logger=logger,
                model=best_model,
                X_features=X_test,  # Pass feature set for SHAP if applicable
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

        # Optionally, delete the best model from memory if no longer needed
        del best_model
        gc.collect()

        logger.info("Sepsis Prediction Pipeline completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
