# main1.py

import gc
import json
import os
import time
from typing import Any, Tuple
from datetime import datetime

from scipy.stats import randint, uniform

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src import (
    evaluate_model,  # Import the updated unified evaluation function
    load_data,
    preprocess_pipeline,
    split_data,
    train_logistic_regression,
)
from src.evaluation import plot_class_distribution, plot_feature_correlation_heatmap

from src.logging_utils import log_phase, log_step, log_memory, log_dataframe_info
from src.logger_config import setup_logger

# Initialize a temporary logger before Run ID is generated
temp_logger = setup_logger(
    name="sepsis_prediction.main.temp",
    log_file="logs/sepsis_prediction_temp.log",
    use_json=False,
)


def train_and_evaluate_model(
    model_name: str,
    pipeline: ImbPipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    run_report_dir: str,
) -> Tuple[float, Any]:
    """Train and evaluate a model using the unified evaluation function."""
    logger.info(f"\nTraining {model_name}...")
    start_time = time.time()

    try:
        # Log memory before training
        log_memory(logger, f"Before training {model_name}")

        # Train the model
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred = pipeline.predict(X_val)

        # Get the actual model from the pipeline
        model = None
        for step_name, estimator in pipeline.named_steps.items():
            if isinstance(
                estimator,
                (
                    RandomForestClassifier,
                    GaussianNB,
                    KNeighborsClassifier,
                    LogisticRegression,
                ),
            ):
                model = estimator
                break

        if model is None:
            logger.warning(f"No compatible model found in pipeline for {model_name}.")
            return 0.0, None

        # Get predicted probabilities if available
        y_pred_proba = None
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_val)[:, 1]

        # Use the unified evaluation function
        metrics = evaluate_model(
            y_true=y_val,
            y_pred=y_pred,
            model_name=model_name.lower().replace(" ", "_"),
            y_pred_proba=y_pred_proba,
            model=model,
            report_dir=run_report_dir,
        )

        # Log memory after training
        log_memory(logger, f"After training {model_name}")

        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Completed training {model_name} in {duration:.2f} seconds.")

        return metrics.get("F1 Score", 0.0), model

    except Exception as e:
        logger.error(
            f"Error in training and evaluation of {model_name}: {e}", exc_info=True
        )
        return 0.0, None

    finally:
        # Clear memory
        del pipeline
        gc.collect()


@log_phase(temp_logger)
def main():
    # Generate a unique Run ID based on the current timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_logger.info(f"Run ID: {run_id}")

    try:
        # Define base directories
        base_log_dir = "logs"
        base_report_dir = "reports/evaluations"
        base_model_dir = "models"

        # Create unique subdirectories for the run
        run_log_dir = os.path.join(base_log_dir, run_id)
        run_report_dir = os.path.join(base_report_dir, run_id)
        run_model_dir = os.path.join(base_model_dir, run_id)

        os.makedirs(run_log_dir, exist_ok=True)
        os.makedirs(run_report_dir, exist_ok=True)
        os.makedirs(run_model_dir, exist_ok=True)

        # Re-initialize logger to use the run-specific log file
        global logger
        logger = setup_logger(
            name=f"sepsis_prediction.main.{run_id}",
            log_file=os.path.join(run_log_dir, f"sepsis_prediction_{run_id}.log"),
            use_json=False,
        )
        logger.info(f"Starting Sepsis Prediction Pipeline with Run ID: {run_id}")

        # Step 1: Load and split the data
        with log_step(logger, "Data Loading and Splitting"):
            logger.info("Loading raw data from data/raw/Dataset.csv")
            combined_df = load_data("data/raw/Dataset.csv")
            log_dataframe_info(logger, combined_df, "Initial")

            logger.info("Splitting data into training, validation, and testing sets")
            df_train, df_val, df_test = split_data(
                combined_df, train_size=0.7, val_size=0.15
            )

            # Plot initial class distribution
            plot_class_distribution(
                y=df_train["SepsisLabel"],
                model_name="Original Training Data",
                report_dir=run_report_dir,
                title_suffix="_before_resampling",
            )

            # Log memory after data splitting
            log_memory(logger, "After Data Loading and Splitting")

        # Step 2: Preprocess all datasets together with parallel processing
        with log_step(logger, "Data Preprocessing"):
            logger.info("Preprocessing all datasets")
            log_memory(logger, "Before Preprocessing")
            df_train_processed, df_val_processed, df_test_processed = (
                preprocess_pipeline(
                    train_df=df_train,
                    val_df=df_val,
                    test_df=df_test,
                    n_jobs=-1,
                )
            )
            log_memory(logger, "After Preprocessing")

            log_dataframe_info(logger, df_train_processed, "Training Processed")
            log_dataframe_info(logger, df_val_processed, "Validation Processed")
            log_dataframe_info(logger, df_test_processed, "Test Processed")

            # Clean up original dataframes
            del df_train, df_val, df_test, combined_df
            gc.collect()
            logger.info("Cleaned up original dataframes after preprocessing")
            log_memory(logger, "After Cleaning Up Original Dataframes")

        # Step 3: Prepare features and targets
        with log_step(logger, "Feature and Target Preparation"):
            log_memory(logger, "Before Feature and Target Preparation")
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
                report_dir=run_report_dir,
            )

            # Log feature names after preprocessing
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

            log_memory(logger, "After Feature and Target Preparation")

        # Step 4: Handle class imbalance with SMOTETomek
        with log_step(logger, "Handling Class Imbalance with SMOTETomek"):
            logger.info("Applying SMOTETomek to training data")
            log_memory(logger, "Before SMOTETomek")

            smote = SMOTE(sampling_strategy=0.3, random_state=42, k_neighbors=5)
            tomek = TomekLinks()
            smote_tomek = SMOTETomek(smote=smote, tomek=tomek, random_state=42)
            X_train_resampled, y_train_resampled = smote_tomek.fit_resample(
                X_train, y_train
            )

            # Ensure 'Patient_ID' is not in the resampled training data
            if "Patient_ID" in X_train_resampled.columns:
                logger.warning(
                    "'Patient_ID' found in resampled training data. Dropping it now."
                )
                X_train_resampled = X_train_resampled.drop(columns=["Patient_ID"])
            else:
                logger.info(
                    "'Patient_ID' successfully excluded from resampled training data."
                )

            # Similarly, ensure 'Patient_ID' is not in validation and test sets
            for dataset_name, dataset in [("validation", X_val), ("test", X_test)]:
                if "Patient_ID" in dataset.columns:
                    logger.warning(
                        f"'Patient_ID' found in {dataset_name} data. Dropping it now."
                    )
                    dataset = dataset.drop(columns=["Patient_ID"])
                    if dataset_name == "validation":
                        X_val = dataset
                    else:
                        X_test = dataset
                else:
                    logger.info(
                        f"'Patient_ID' successfully excluded from {dataset_name} data."
                    )

            # Plot resampled class distribution
            plot_class_distribution(
                y=y_train_resampled,
                model_name="Resampled Training Data",
                report_dir=run_report_dir,
                title_suffix="_after_resampling",
            )

            log_memory(logger, "After SMOTETomek")

        # Step 5: Model Training
        with log_step(logger, "Model Training"):
            # Initialize tracking variables
            models = {}
            best_score = 0
            best_model_name = None

            # Define models and their configurations inside main to access runtime variables
            models_config = {
                "Random Forest (Tuned)": {
                    "pipeline": Pipeline(
                        [
                            (
                                "classifier",
                                RandomForestClassifier(
                                    random_state=42,
                                    class_weight="balanced",
                                    n_jobs=-1,
                                ),
                            ),
                        ]
                    ),
                    "param_dist": {
                        "classifier__n_estimators": randint(
                            100, 500
                        ),  # Increased upper limit
                        "classifier__max_depth": [None]
                        + list(range(10, 31, 5)),  # Expanded range
                        "classifier__min_samples_split": randint(
                            2, 12
                        ),  # Expanded range
                        "classifier__min_samples_leaf": randint(1, 6),  # Expanded range
                        "classifier__max_features": [
                            "sqrt",
                            "log2",
                            0.2,
                            0.5,
                        ],  # Added float options
                    },
                    "scoring": "f1",
                    "cv": StratifiedKFold(
                        n_splits=5, shuffle=True, random_state=42
                    ),  # Reduced folds
                },
                "Naive Bayes": {
                    "pipeline": ImbPipeline(
                        [
                            ("naive_bayes", GaussianNB()),
                        ]
                    ),
                    "scoring": "f1",
                },
                "KNN": {
                    "pipeline": ImbPipeline(
                        [
                            (
                                "knn",
                                KNeighborsClassifier(
                                    n_neighbors=5,
                                    weights="distance",
                                    algorithm="auto",
                                    n_jobs=-1,
                                ),
                            ),
                        ]
                    ),
                    "scoring": "f1",
                },
                "Logistic Regression": {
                    "pipeline": ImbPipeline(
                        [
                            (
                                "logistic_regression",
                                train_logistic_regression(
                                    X_train_resampled, y_train_resampled
                                ),
                            ),
                        ]
                    ),
                    "scoring": "f1",
                },
                "XGBoost": {
                    "pipeline": ImbPipeline(
                        [
                            (
                                "xgboost",
                                xgb.XGBClassifier(
                                    use_label_encoder=False,
                                    eval_metric="logloss",
                                    random_state=42,
                                    n_jobs=-1,
                                ),
                            ),
                        ]
                    ),
                    "param_dist": {
                        "xgboost__n_estimators": randint(100, 300),
                        "xgboost__max_depth": randint(3, 10),
                        "xgboost__learning_rate": uniform(0.01, 0.2),
                        "xgboost__subsample": uniform(0.5, 0.3),
                        "xgboost__colsample_bytree": uniform(0.5, 0.3),
                        "xgboost__gamma": uniform(0, 3),
                    },
                    "scoring": "f1",
                    "cv": StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                },
                "Ensemble_RF_XGBoost": {  # Integrated Ensemble Model
                    "pipeline": VotingClassifier(
                        estimators=[
                            (
                                "rf",
                                RandomForestClassifier(
                                    n_estimators=300,
                                    max_depth=20,
                                    min_samples_split=4,
                                    min_samples_leaf=2,
                                    max_features="sqrt",
                                    class_weight="balanced",
                                    random_state=42,
                                    n_jobs=-1,
                                ),
                            ),
                            (
                                "xgb",
                                xgb.XGBClassifier(
                                    n_estimators=300,
                                    max_depth=10,
                                    learning_rate=0.1,
                                    subsample=0.5,
                                    colsample_bytree=0.5,
                                    gamma=2.9,
                                    scale_pos_weight=y_train_resampled.value_counts()[0]
                                    / y_train_resampled.value_counts()[1],
                                    use_label_encoder=False,
                                    eval_metric="logloss",
                                    random_state=42,
                                    n_jobs=-1,
                                ),
                            ),
                        ],
                        voting="soft",  # Use soft voting to leverage predicted probabilities
                    ),
                    "scoring": "f1",
                    "cv": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                },
            }

            # Log memory before model training
            log_memory(logger, "Before Model Training")

            # Total models to train
            total_models = len(models_config)

            for idx, (model_name, config) in enumerate(models_config.items(), 1):
                pipeline = config.get("pipeline")
                scoring = config.get("scoring", "f1")
                cv_strategy = config.get(
                    "cv", StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                )

                logger.info(
                    f"Starting training for {model_name} (Model {idx}/{total_models})"
                )
                log_memory(logger, f"Before training {model_name}")

                if model_name in ["Random Forest (Tuned)", "XGBoost"]:
                    # Hyperparameter tuning for Random Forest and XGBoost
                    param_dist = config.get("param_dist", {})
                    n_iter = 20 if model_name == "Random Forest (Tuned)" else 10

                    random_search = RandomizedSearchCV(
                        estimator=pipeline,
                        param_distributions=param_dist,
                        n_iter=n_iter,
                        scoring=scoring,
                        cv=cv_strategy,
                        verbose=3,
                        random_state=42,
                        n_jobs=-1,
                        error_score="raise",
                    )

                    try:
                        logger.info(
                            f"Starting hyperparameter tuning for {model_name} (Model {idx}/{total_models})"
                        )
                        log_memory(logger, f"Before tuning {model_name}")
                        start_tuning = time.time()
                        random_search.fit(X_train_resampled, y_train_resampled)
                        end_tuning = time.time()
                        tuning_duration = end_tuning - start_tuning
                        logger.info(
                            f"Best parameters found: {random_search.best_params_}"
                        )
                        logger.info(
                            f"Best {scoring.capitalize()} from tuning: {random_search.best_score_:.4f}"
                        )
                        logger.info(
                            f"Completed hyperparameter tuning for {model_name} in {tuning_duration:.2f} seconds."
                        )
                        log_memory(logger, f"After tuning {model_name}")
                    except Exception as e:
                        logger.error(
                            f"Hyperparameter tuning failed for {model_name}: {e}",
                            exc_info=True,
                        )
                        continue  # Skip to next model

                    # Evaluate the best estimator on the validation set
                    best_estimator = random_search.best_estimator_
                    y_pred = best_estimator.predict(X_val)
                    y_pred_proba = best_estimator.predict_proba(X_val)[:, 1]

                    # Evaluate
                    metrics = evaluate_model(
                        y_true=y_val,
                        y_pred=y_pred,
                        model_name=model_name.lower().replace(" ", "_"),
                        y_pred_proba=y_pred_proba,
                        model=best_estimator,
                        report_dir=run_report_dir,
                    )

                    # Update best score and model
                    if metrics.get("F1 Score", 0.0) > best_score:
                        best_score = metrics["F1 Score"]
                        best_model_name = model_name
                        models[model_name] = best_estimator

                    # Save hyperparameters with Run ID
                    best_params = random_search.best_params_
                    with open(
                        os.path.join(
                            run_report_dir,
                            f"{model_name.lower().replace(' ', '_')}_params_{run_id}.json",
                        ),
                        "w",
                    ) as f:
                        json.dump(best_params, f, indent=4)

                    logger.info(
                        f"Hyperparameter tuning for {model_name} completed successfully."
                    )
                    logger.info(
                        f"Completed training {model_name} (Model {idx}/{total_models})"
                    )

                elif model_name == "Ensemble_RF_XGBoost":
                    # Ensemble model training
                    try:
                        logger.info(
                            f"Training ensemble model: {model_name} (Model {idx}/{total_models})"
                        )
                        log_memory(logger, f"Before training {model_name}")

                        # Fit the ensemble
                        pipeline.fit(X_train_resampled, y_train_resampled)
                        logger.info(f"Completed training ensemble model: {model_name}")
                        log_memory(logger, f"After training {model_name}")

                        # Make predictions
                        y_pred = pipeline.predict(X_val)
                        y_pred_proba = pipeline.predict_proba(X_val)[:, 1]

                        # Evaluate
                        metrics = evaluate_model(
                            y_true=y_val,
                            y_pred=y_pred,
                            model_name=model_name.lower().replace(" ", "_"),
                            y_pred_proba=y_pred_proba,
                            model=pipeline,
                            report_dir=run_report_dir,
                        )

                        # Update best score and model
                        if metrics.get("F1 Score", 0.0) > best_score:
                            best_score = metrics["F1 Score"]
                            best_model_name = model_name
                            models[model_name] = pipeline

                        # Save hyperparameters with Run ID
                        # Note: VotingClassifier doesn't have hyperparameters to save in the same way
                        ensemble_params = pipeline.get_params()
                        with open(
                            os.path.join(
                                run_report_dir,
                                f"{model_name.lower().replace(' ', '_')}_params_{run_id}.json",
                            ),
                            "w",
                        ) as f:
                            json.dump(ensemble_params, f, indent=4)

                        logger.info(
                            f"Completed training ensemble model: {model_name} (Model {idx}/{total_models})"
                        )

                    except Exception as e:
                        logger.error(
                            f"Ensemble training failed for {model_name}: {e}",
                            exc_info=True,
                        )
                        continue  # Skip to next model

                else:
                    # Train and evaluate other models
                    logger.info(f"Training model {idx}/{total_models}: {model_name}")
                    log_memory(logger, f"Before training {model_name}")
                    start_training = time.time()
                    score, trained_model = train_and_evaluate_model(
                        model_name,
                        config["pipeline"],
                        X_train_resampled,
                        y_train_resampled,
                        X_val,
                        y_val,
                        run_report_dir,
                    )
                    end_training = time.time()
                    training_duration = end_training - start_training
                    logger.info(
                        f"Completed training {model_name} in {training_duration:.2f} seconds."
                    )
                    log_memory(logger, f"After training {model_name}")

                    if score > best_score:
                        best_score = score
                        best_model_name = model_name
                        models[model_name] = trained_model

                # Save the trained model
                if model_name in models:
                    model_save_path = os.path.join(
                        run_model_dir,
                        f"{model_name.lower().replace(' ', '_')}_{run_id}.pkl",
                    )
                    joblib.dump(models[model_name], model_save_path)
                    logger.info(f"Saved model '{model_name}' to {model_save_path}")

            # Step 6: Final evaluation on test set
            with log_step(logger, "Final Evaluation on Test Set"):
                log_memory(logger, "Before Final Evaluation")
                if best_model_name is not None:
                    logger.info(
                        f"\nPerforming final evaluation with best model: {best_model_name}"
                    )
                    best_model = models[best_model_name]

                    if (
                        best_model_name == "XGBoost"
                        or best_model_name == "Ensemble_RF_XGBoost"
                    ):
                        # For XGBoost and Ensemble, use predict_proba if available
                        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                        final_predictions = (y_pred_proba > 0.5).astype(int)
                    else:
                        # For other models
                        y_pred_proba = (
                            best_model.predict_proba(X_test)[:, 1]
                            if hasattr(best_model, "predict_proba")
                            else None
                        )
                        final_predictions = best_model.predict(X_test)

                    # Evaluate
                    final_metrics = evaluate_model(
                        y_true=y_test,
                        y_pred=final_predictions,
                        model_name=f"final_{best_model_name.lower()}",
                        y_pred_proba=y_pred_proba,
                        model=best_model,
                        report_dir=run_report_dir,
                    )

                    # Step 7: Save the best model with Run ID
                    with log_step(logger, "Saving the Best Model"):
                        logger.info(f"Saving the best model ({best_model_name})")
                        model_path = os.path.join(
                            run_model_dir,
                            f"best_model_{best_model_name.lower()}_{run_id}.pkl",
                        )
                        joblib.dump(best_model, model_path)
                        logger.info(f"Model saved to {model_path}")

                    with open(
                        os.path.join(run_report_dir, "final_metrics.json"), "w"
                    ) as f:
                        json.dump(final_metrics, f, indent=4)
                        # Clean up
                    del best_model
                    gc.collect()
                    logger.info("Cleaned up best model from memory")
                    log_memory(logger, "After Final Evaluation and Model Saving")

                    logger.info("Sepsis Prediction Pipeline completed successfully.")

                else:
                    logger.warning("No model was successfully trained and evaluated.")
                    log_memory(logger, "After Final Evaluation - No Model Trained")

    except Exception as e:
        temp_logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

    finally:
        # Remove temporary logger
        temp_logger.handlers.clear()


if __name__ == "__main__":
    main()
