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
        # ! change back to 0.3
        smote = SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=5)

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

# def main():
#     # Set up logger

#     logger.info("Starting Sepsis Prediction Pipeline")

#     try:
#         # Step 1: Load and split the data
#         logger.info("Loading raw data")
#         combined_df = load_data("data/raw/Dataset.csv")
#         logger.info("Splitting data into training, validation, and testing sets")
#         df_train, df_val, df_test = split_data(
#             combined_df, train_size=0.7, val_size=0.15
#         )

#         # Plot initial class distribution
#         plot_class_distribution(
#             y=df_train["SepsisLabel"],
#             model_name="Original Training Data",
#             report_dir="reports/evaluations",
#             title_suffix="_before_resampling",
#         )

#         # Step 2: Preprocess all datasets together with parallel processing
#         logger.info("Preprocessing all datasets")
#         df_train_processed, df_val_processed, df_test_processed = preprocess_pipeline(
#             train_df=df_train,
#             val_df=df_val,
#             test_df=df_test,
#             n_jobs=-1,
#         )

#         # Clean up original dataframes
#         del df_train, df_val, df_test, combined_df
#         gc.collect()

#         # Rest of your main function remains the same...
#         # Step 3: Prepare features and targets
#         X_train = df_train_processed.drop("SepsisLabel", axis=1)
#         y_train = df_train_processed["SepsisLabel"]
#         X_val = df_val_processed.drop("SepsisLabel", axis=1)
#         y_val = df_val_processed["SepsisLabel"]
#         X_test = df_test_processed.drop("SepsisLabel", axis=1)
#         y_test = df_test_processed["SepsisLabel"]

#         # Plot feature correlation heatmap
#         plot_feature_correlation_heatmap(
#             df=X_train,
#             model_name="Training Data",
#             report_dir="reports/evaluations",
#         )

#         # Step 4: Handle class imbalance with SMOTEENN

#         logger.info("Applying SMOTEENN to training data")
#         smote_enn = SMOTEENN(
#             smote=SMOTE(sampling_strategy=0.3, random_state=42, k_neighbors=5),
#             enn=EditedNearestNeighbours(),
#             random_state=42,
#         )
#         X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

#         # Plot resampled class distribution
#         plot_class_distribution(
#             y=y_train_resampled,
#             model_name="Resampled Training Data",
#             report_dir="reports/evaluations",
#             title_suffix="_after_resampling",
#         )

#         # 5.1 Random Forest
#         models = {}
#         best_score = 0
#         best_model_name = None

#         from scipy.stats import randint, uniform
#         from sklearn.model_selection import RandomizedSearchCV

#         # Define the Random Forest pipeline
#         rf_pipeline = ImbPipeline(
#             [
#                 (
#                     "random_forest",
#                     RandomForestClassifier(
#                         random_state=42,
#                         class_weight="balanced",
#                         n_jobs=-1,
#                     ),
#                 ),
#             ]
#         )

#         # Define the hyperparameter grid
#         param_dist = {
#             "random_forest__n_estimators": randint(100, 500),
#             "random_forest__max_depth": [None] + list(range(10, 31, 5)),
#             "random_forest__min_samples_split": randint(2, 11),
#             "random_forest__min_samples_leaf": randint(1, 5),
#             "random_forest__max_features": ["sqrt", "log2"],  # Removed 'auto'
#         }

#         # Initialize RandomizedSearchCV
#         random_search = RandomizedSearchCV(
#             estimator=rf_pipeline,
#             param_distributions=param_dist,
#             n_iter=10,  # Number of parameter settings sampled
#             scoring="f1",
#             cv=5,  # 5-fold cross-validation
#             verbose=2,
#             random_state=42,
#             n_jobs=-1,
#             error_score=0,  # If a parameter set raises an error, continue and set the score as 0
#         )

#         # Fit RandomizedSearchCV
#         logger.info("Starting hyperparameter tuning for Random Forest")
#         random_search.fit(X_train_resampled, y_train_resampled)
#         logger.info(f"Best parameters found: {random_search.best_params_}")
#         logger.info(f"Best F1 Score from tuning: {random_search.best_score_:.4f}")

#         # Evaluate the best estimator on the validation set
#         best_rf_model = random_search.best_estimator_
#         y_pred = best_rf_model.predict(X_val)
#         y_pred_proba = best_rf_model.predict_proba(X_val)[:, 1]

#         # Use the unified evaluation function
#         metrics = evaluate_model(
#             y_true=y_val,
#             y_pred=y_pred,
#             model_name="random_forest_tuned",
#             y_pred_proba=y_pred_proba,
#             model=best_rf_model,
#         )

#         # Update best score and model if necessary
#         if metrics["F1 Score"] > best_score:
#             best_score = metrics["F1 Score"]
#             best_model_name = "Random Forest (Tuned)"
#             models["Random Forest (Tuned)"] = best_rf_model

#         # Save the best parameters and metrics
#         best_rf_params = random_search.best_params_
#         with open(
#             os.path.join("reports/evaluations", "random_forest_tuned_params.json"), "w"
#         ) as f:
#             json.dump(best_rf_params, f, indent=4)

#         logger.info("Hyperparameter tuning for Random Forest completed successfully.")

#         del rf_pipeline
#         gc.collect()

#         # 5.2 Naive Bayes
#         nb_pipeline = ImbPipeline(
#             [
#                 ("scaler", StandardScaler()),
#                 ("naive_bayes", GaussianNB()),
#             ]
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

#         del nb_pipeline
#         gc.collect()

#         # 5.3 KNN
#         knn_pipeline = ImbPipeline(
#             [
#                 ("scaler", StandardScaler()),
#                 (
#                     "knn",
#                     KNeighborsClassifier(
#                         n_neighbors=5, weights="distance", algorithm="auto", n_jobs=-1
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

#             # Convert data to numpy arrays
#             X_train_array = X_train_resampled.to_numpy()
#             y_train_array = y_train_resampled.to_numpy()
#             X_val_array = X_val.to_numpy()
#             y_val_array = y_val.to_numpy()

#             # Create DMatrix objects
#             dtrain = xgb.DMatrix(X_train_array, label=y_train_array)
#             dval = xgb.DMatrix(X_val_array, label=y_val_array)

#             # Train XGBoost model
#             xgb_model = xgb.train(
#                 params={
#                     "max_depth": 6,
#                     "min_child_weight": 1,
#                     "eta": 0.05,
#                     "subsample": 0.8,
#                     "colsample_bytree": 0.8,
#                     "objective": "binary:logistic",
#                     "eval_metric": ["auc", "error"],
#                     "tree_method": "hist",
#                     "alpha": 1,
#                     "lambda": 1,
#                     "random_state": 42,
#                 },
#                 dtrain=dtrain,
#                 num_boost_round=200,
#                 evals=[(dtrain, "train"), (dval, "eval")],
#                 early_stopping_rounds=20,
#                 verbose_eval=False,
#             )

#             # Make predictions
#             y_pred_proba = xgb_model.predict(dval)
#             y_pred = (y_pred_proba > 0.5).astype(int)

#             # Evaluate XGBoost model
#             metrics = evaluate_model(
#                 y_true=y_val,
#                 y_pred=y_pred,
#                 model_name="xgboost",
#                 y_pred_proba=y_pred_proba,
#                 model=xgb_model,
#             )

#             score = metrics["F1 Score"]
#             models["XGBoost"] = xgb_model

#             if score > best_score:
#                 best_score = score
#                 best_model_name = "XGBoost"

#             del dtrain, dval
#             gc.collect()

#         except Exception as e:
#             logger.error(f"Error in XGBoost training: {str(e)}", exc_info=True)

#         # Step 6: Final evaluation on test set
#         logger.info(f"\nPerforming final evaluation with best model: {best_model_name}")
#         best_model = models[best_model_name]

#         if best_model_name == "XGBoost":
#             # Handle XGBoost predictions
#             X_test_array = X_test.to_numpy()
#             dtest = xgb.DMatrix(X_test_array)
#             final_predictions_proba = best_model.predict(dtest)
#             final_predictions = (final_predictions_proba > 0.5).astype(int)

#             # Evaluate
#             evaluate_model(
#                 y_true=y_test,
#                 y_pred=final_predictions,
#                 model_name=f"final_{best_model_name.lower()}",
#                 y_pred_proba=final_predictions_proba,
#                 model=best_model,
#             )

#             del dtest
#             gc.collect()
#         else:
#             # Handle other models
#             final_predictions = best_model.predict(X_test)
#             final_predictions_proba = (
#                 best_model.predict_proba(X_test)[:, 1]
#                 if hasattr(best_model, "predict_proba")
#                 else None
#             )

#             # Evaluate
#             evaluate_model(
#                 y_true=y_test,
#                 y_pred=final_predictions,
#                 model_name=f"final_{best_model_name.lower()}",
#                 y_pred_proba=final_predictions_proba,
#                 model=best_model,
#             )

#         # Step 7: Save the best model
#         logger.info(f"Saving the best model ({best_model_name})")
#         model_dir = "models"
#         os.makedirs(model_dir, exist_ok=True)
#         model_path = os.path.join(
#             model_dir, f"best_model_{best_model_name.lower()}.pkl"
#         )
#         joblib.dump(best_model, model_path)
#         logger.info(f"Model saved to {model_path}")

#         # Clean up
#         del best_model
#         gc.collect()

#         logger.info("Sepsis Prediction Pipeline completed successfully.")

#     except Exception as e:
#         logger.error(f"An error occurred: {str(e)}", exc_info=True)
#         raise


def main():
    logger.info("Starting Sepsis Prediction Pipeline")

    try:
        # Step 1: Load and split the data
        logger.info("Loading raw data")
        combined_df = load_data("data/raw/Dataset.csv")
        logger.info("Splitting data into training, validation, and testing sets")
        df_train, df_val, df_test = split_data(
            combined_df, train_size=0.7, val_size=0.15
        )

        # Plot initial class distribution
        plot_class_distribution(
            y=df_train["SepsisLabel"],
            model_name="Original Training Data",
            report_dir="reports/evaluations",
            title_suffix="_before_resampling",
        )

        # Step 2: Preprocess all datasets together with parallel processing
        logger.info("Preprocessing all datasets")
        df_train_processed, df_val_processed, df_test_processed = preprocess_pipeline(
            train_df=df_train,
            val_df=df_val,
            test_df=df_test,
            n_jobs=-1,
        )

        # **Add Logging After Encoding Here**
        logger.info(
            f"Training set features after encoding: {list(df_train_processed.columns)}"
        )
        logger.info(
            f"Validation set features after encoding: {list(df_val_processed.columns)}"
        )
        logger.info(
            f"Test set features after encoding: {list(df_test_processed.columns)}"
        )

        # Clean up original dataframes
        del df_train, df_val, df_test, combined_df
        gc.collect()

        # Step 3: Prepare features and targets
        X_train = df_train_processed.drop("SepsisLabel", axis=1)
        y_train = df_train_processed["SepsisLabel"]
        X_val = df_val_processed.drop("SepsisLabel", axis=1)
        y_val = df_val_processed["SepsisLabel"]
        X_test = df_test_processed.drop("SepsisLabel", axis=1)
        y_test = df_test_processed["SepsisLabel"]

        # Plot feature correlation heatmap
        plot_feature_correlation_heatmap(
            df=X_train,
            model_name="Training Data",
            report_dir="reports/evaluations",
        )

        # Log feature names
        logger.info(f"Training data features: {list(X_train.columns)}")
        logger.info(f"Validation data features: {list(X_val.columns)}")
        logger.info(f"Test data features: {list(X_test.columns)}")

        # Check for feature consistency
        train_features = set(X_train.columns)
        val_features = set(X_val.columns)
        test_features = set(X_test.columns)

        if train_features != val_features:
            logger.error("Feature mismatch between training and validation sets.")
            logger.error(f"Training features: {train_features}")
            logger.error(f"Validation features: {val_features}")
            raise ValueError("Training and validation feature sets do not match.")

        if train_features != test_features:
            logger.error("Feature mismatch between training and test sets.")
            logger.error(f"Training features: {train_features}")
            logger.error(f"Test features: {test_features}")
            raise ValueError("Training and test feature sets do not match.")

        # Step 4: Handle class imbalance with SMOTEENN
        logger.info("Applying SMOTEENN to training data")
        smote_enn = SMOTEENN(
            smote=SMOTE(sampling_strategy=0.3, random_state=42, k_neighbors=5),
            enn=EditedNearestNeighbours(),
            random_state=42,
        )
        X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

        # Plot resampled class distribution
        plot_class_distribution(
            y=y_train_resampled,
            model_name="Resampled Training Data",
            report_dir="reports/evaluations",
            title_suffix="_after_resampling",
        )

        # Step 5.1: Random Forest Hyperparameter Tuning
        models = {}
        best_score = 0
        best_model_name = None

        from scipy.stats import randint, uniform

        # Define the Random Forest pipeline without scaler
        rf_pipeline = ImbPipeline(
            [
                (
                    "random_forest",
                    RandomForestClassifier(
                        random_state=42,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        # Define the hyperparameter grid
        param_dist = {
            "random_forest__n_estimators": randint(100, 500),
            "random_forest__max_depth": [None] + list(range(10, 31, 5)),
            "random_forest__min_samples_split": randint(2, 11),
            "random_forest__min_samples_leaf": randint(1, 5),
            "random_forest__max_features": ["sqrt", "log2"],  # Removed 'auto'
        }

        # Define stratified cross-validation
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Initialize RandomizedSearchCV with stratified CV, increased verbosity, and error handling
        random_search = RandomizedSearchCV(
            estimator=rf_pipeline,
            param_distributions=param_dist,
            n_iter=10,  # Number of parameter settings sampled
            scoring="f1",
            cv=cv_strategy,  # Use stratified CV
            verbose=3,  # Increased verbosity for detailed logging
            random_state=42,
            n_jobs=-1,
            error_score="raise",  # Raise exceptions to capture errors
        )

        # Fit RandomizedSearchCV with exception handling
        try:
            logger.info("Starting hyperparameter tuning for Random Forest")
            random_search.fit(X_train_resampled, y_train_resampled)
            logger.info(f"Best parameters found: {random_search.best_params_}")
            logger.info(f"Best F1 Score from tuning: {random_search.best_score_:.4f}")
        except Exception as e:
            logger.error(f"Hyperparameter tuning failed: {str(e)}", exc_info=True)
            # Optionally, adjust the hyperparameter grid here
            raise  # Re-raise the exception after logging

        # Evaluate the best estimator on the validation set
        best_rf_model = random_search.best_estimator_
        y_pred = best_rf_model.predict(X_val)
        y_pred_proba = best_rf_model.predict_proba(X_val)[:, 1]

        # Use the unified evaluation function
        metrics = evaluate_model(
            y_true=y_val,
            y_pred=y_pred,
            model_name="random_forest_tuned",
            y_pred_proba=y_pred_proba,
            model=best_rf_model,
        )

        # Update best score and model if necessary
        if metrics["F1 Score"] > best_score:
            best_score = metrics["F1 Score"]
            best_model_name = "Random Forest (Tuned)"
            models["Random Forest (Tuned)"] = best_rf_model

        # Save the best parameters and metrics
        best_rf_params = random_search.best_params_
        with open(
            os.path.join("reports/evaluations", "random_forest_tuned_params.json"), "w"
        ) as f:
            json.dump(best_rf_params, f, indent=4)

        logger.info("Hyperparameter tuning for Random Forest completed successfully.")

        del rf_pipeline
        gc.collect()

        # Step 5.2: Naive Bayes
        nb_pipeline = ImbPipeline(
            [
                ("naive_bayes", GaussianNB()),
            ]
        )
        score, models["Naive Bayes"] = train_and_evaluate_model(
            "Naive Bayes",
            nb_pipeline,
            X_train_resampled,
            y_train_resampled,
            X_val,
            y_val,
        )
        if score > best_score:
            best_score = score
            best_model_name = "Naive Bayes"

        del nb_pipeline
        gc.collect()

        # Step 5.3: KNN
        knn_pipeline = ImbPipeline(
            [
                (
                    "knn",
                    KNeighborsClassifier(
                        n_neighbors=5, weights="distance", algorithm="auto", n_jobs=-1
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
        )
        if score > best_score:
            best_score = score
            best_model_name = "KNN"

        del knn_pipeline
        gc.collect()

        # Step 5.4: Logistic Regression
        lr_model = train_logistic_regression(X_train_resampled, y_train_resampled)
        lr_pipeline = ImbPipeline([("logistic_regression", lr_model)])
        score, models["Logistic Regression"] = train_and_evaluate_model(
            "Logistic Regression",
            lr_pipeline,
            X_train_resampled,
            y_train_resampled,
            X_val,
            y_val,
        )
        if score > best_score:
            best_score = score
            best_model_name = "Logistic Regression"

        del lr_pipeline
        gc.collect()

        # Step 5.5: XGBoost
        try:
            logger.info("Training XGBoost model")

            # Convert data to numpy arrays
            X_train_array = X_train_resampled.to_numpy()
            y_train_array = y_train_resampled.to_numpy()
            X_val_array = X_val.to_numpy()
            y_val_array = y_val.to_numpy()

            # Create DMatrix objects
            dtrain = xgb.DMatrix(X_train_array, label=y_train_array)
            dval = xgb.DMatrix(X_val_array, label=y_val_array)

            # Train XGBoost model
            xgb_model = xgb.train(
                params={
                    "max_depth": 6,
                    "min_child_weight": 1,
                    "eta": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "objective": "binary:logistic",
                    "eval_metric": ["auc", "error"],
                    "tree_method": "hist",
                    "alpha": 1,
                    "lambda": 1,
                    "random_state": 42,
                },
                dtrain=dtrain,
                num_boost_round=200,
                evals=[(dtrain, "train"), (dval, "eval")],
                early_stopping_rounds=20,
                verbose_eval=1,  # Enable verbose evaluation
            )

            # Make predictions
            y_pred_proba = xgb_model.predict(dval)
            y_pred = (y_pred_proba > 0.5).astype(int)

            # Evaluate XGBoost model
            metrics = evaluate_model(
                y_true=y_val,
                y_pred=y_pred,
                model_name="xgboost",
                y_pred_proba=y_pred_proba,
                model=xgb_model,
            )

            score = metrics["F1 Score"]
            models["XGBoost"] = xgb_model

            if score > best_score:
                best_score = score
                best_model_name = "XGBoost"

            del dtrain, dval
            gc.collect()

        except Exception as e:
            logger.error(f"Error in XGBoost training: {str(e)}", exc_info=True)

        # Step 6: Final evaluation on test set
        logger.info(f"\nPerforming final evaluation with best model: {best_model_name}")
        best_model = models[best_model_name]

        if best_model_name == "XGBoost":
            # Handle XGBoost predictions
            X_test_array = X_test.to_numpy()
            dtest = xgb.DMatrix(X_test_array)
            final_predictions_proba = best_model.predict(dtest)
            final_predictions = (final_predictions_proba > 0.5).astype(int)

            # Evaluate
            evaluate_model(
                y_true=y_test,
                y_pred=final_predictions,
                model_name=f"final_{best_model_name.lower()}",
                y_pred_proba=final_predictions_proba,
                model=best_model,
            )

            del dtest
            gc.collect()
        else:
            # Handle other models
            final_predictions = best_model.predict(X_test)
            final_predictions_proba = (
                best_model.predict_proba(X_test)[:, 1]
                if hasattr(best_model, "predict_proba")
                else None
            )

            # Evaluate
            evaluate_model(
                y_true=y_test,
                y_pred=final_predictions,
                model_name=f"final_{best_model_name.lower()}",
                y_pred_proba=final_predictions_proba,
                model=best_model,
            )

        # Step 7: Save the best model
        logger.info(f"Saving the best model ({best_model_name})")
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(
            model_dir, f"best_model_{best_model_name.lower()}.pkl"
        )
        joblib.dump(best_model, model_path)
        logger.info(f"Model saved to {model_path}")

        # Clean up
        del best_model
        gc.collect()

        logger.info("Sepsis Prediction Pipeline completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
