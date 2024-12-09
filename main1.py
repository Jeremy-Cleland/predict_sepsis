# main1.py

import gc
import json
import logging
import os
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src import (
    evaluate_model,  # Import the updated unified evaluation function
    load_data,
    preprocess_data,
    split_data,
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
) -> Tuple[float, Any]:
    """Train and evaluate a model using the unified evaluation function."""
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

        # Get the actual model from the pipeline
        model = pipeline.named_steps.get(model_name.lower().replace(" ", "_"))

        # Use the unified evaluation function
        metrics = evaluate_model(
            y_true=y_val,
            y_pred=y_pred,
            model_name=model_name.lower().replace(" ", "_"),
            y_pred_proba=y_pred_proba,
            model=model,
        )

        return metrics["F1 Score"], model

    finally:
        # Clear memory
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

        # Plot initial class distribution
        plot_class_distribution(
            y=df_train["SepsisLabel"],
            model_name="Original Training Data",
            report_dir="reports/evaluations",
            title_suffix="_before_resampling",
        )

        # Step 2: Preprocess all datasets
        logger.info("Preprocessing all datasets")
        df_train_processed = preprocess_data(df_train)
        df_val_processed = preprocess_data(df_val)
        df_test_processed = preprocess_data(df_test)

        # Clean up original dataframes
        del df_train, df_val, df_test
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

        # Step 4: Handle class imbalance with SMOTEENN

        logger.info("Applying SMOTEENN to training data")
        smote_enn = SMOTEENN(
            smote=SMOTE(sampling_strategy=0.4, random_state=42, k_neighbors=6),
            enn=EditedNearestNeighbours(
                n_jobs=-1,
                n_neighbors=3,
            ),
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

        # 5.1 Random Forest
        models = {}
        best_score = 0
        best_model_name = None

        from scipy.stats import randint, uniform
        from sklearn.model_selection import RandomizedSearchCV

        # Define the Random Forest pipeline
        rf_pipeline = ImbPipeline(
            [
                ("scaler", StandardScaler()),
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
            "random_forest__max_features": ["sqrt", "log2", None],
        }

        # Initialize RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=rf_pipeline,
            param_distributions=param_dist,
            n_iter=50,  # Number of parameter settings sampled
            scoring="f1",
            cv=5,  # 5-fold cross-validation
            verbose=2,
            random_state=42,
            n_jobs=-1,
        )

        # Fit RandomizedSearchCV
        logger.info("Starting hyperparameter tuning for Random Forest")
        random_search.fit(X_train_resampled, y_train_resampled)
        logger.info(f"Best parameters found: {random_search.best_params_}")
        logger.info(f"Best F1 Score from tuning: {random_search.best_score_:.4f}")

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

        # 5.2 Naive Bayes with optimization
        nb_pipeline = ImbPipeline(
            [("scaler", StandardScaler()), ("naive_bayes", GaussianNB())]
        )

        nb_param_dist = {
            "naive_bayes__var_smoothing": np.logspace(-11, -5, 100),
            "naive_bayes__priors": [None, [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]],
        }

        nb_random_search = RandomizedSearchCV(
            estimator=nb_pipeline,
            param_distributions=nb_param_dist,
            n_iter=20,
            scoring="f1",
            cv=5,
            verbose=2,
            random_state=42,
            n_jobs=-1,
        )

        logger.info("Starting hyperparameter tuning for Naive Bayes")
        nb_random_search.fit(X_train_resampled, y_train_resampled)
        logger.info(f"Best NB parameters: {nb_random_search.best_params_}")
        logger.info(f"Best NB F1 score: {nb_random_search.best_score_:.4f}")

        # Evaluate the best NB model
        best_nb_model = nb_random_search.best_estimator_
        y_pred = best_nb_model.predict(X_val)
        y_pred_proba = best_nb_model.predict_proba(X_val)[:, 1]

        metrics = evaluate_model(
            y_true=y_val,
            y_pred=y_pred,
            model_name="naive_bayes_tuned",
            y_pred_proba=y_pred_proba,
            model=best_nb_model,
        )

        if metrics["F1 Score"] > best_score:
            best_score = metrics["F1 Score"]
            best_model_name = "Naive Bayes (Tuned)"
            models["Naive Bayes (Tuned)"] = best_nb_model

        # Save NB parameters
        with open(
            os.path.join("reports/evaluations", "naive_bayes_tuned_params.json"), "w"
        ) as f:
            json.dump(nb_random_search.best_params_, f, indent=4)

        del nb_pipeline, nb_random_search
        gc.collect()

        # 5.3 KNN with optimization
        knn_pipeline = ImbPipeline(
            [("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_jobs=-1))]
        )

        knn_param_dist = {
            "knn__n_neighbors": randint(3, 20),
            "knn__weights": ["uniform", "distance"],
            "knn__algorithm": ["auto", "ball_tree", "kd_tree"],
            "knn__leaf_size": randint(20, 50),
            "knn__p": [1, 2],
        }

        knn_random_search = RandomizedSearchCV(
            estimator=knn_pipeline,
            param_distributions=knn_param_dist,
            n_iter=50,
            scoring="f1",
            cv=5,
            verbose=2,
            random_state=42,
            n_jobs=-1,
        )

        logger.info("Starting hyperparameter tuning for KNN")
        knn_random_search.fit(X_train_resampled, y_train_resampled)
        logger.info(f"Best KNN parameters: {knn_random_search.best_params_}")
        logger.info(f"Best KNN F1 score: {knn_random_search.best_score_:.4f}")

        # Evaluate the best KNN model
        best_knn_model = knn_random_search.best_estimator_
        y_pred = best_knn_model.predict(X_val)
        y_pred_proba = best_knn_model.predict_proba(X_val)[:, 1]

        metrics = evaluate_model(
            y_true=y_val,
            y_pred=y_pred,
            model_name="knn_tuned",
            y_pred_proba=y_pred_proba,
            model=best_knn_model,
        )

        if metrics["F1 Score"] > best_score:
            best_score = metrics["F1 Score"]
            best_model_name = "KNN (Tuned)"
            models["KNN (Tuned)"] = best_knn_model

        # Save KNN parameters
        with open(
            os.path.join("reports/evaluations", "knn_tuned_params.json"), "w"
        ) as f:
            json.dump(knn_random_search.best_params_, f, indent=4)

        del knn_pipeline, knn_random_search
        gc.collect()

        # 5.4 Logistic Regression with optimization
        lr_pipeline = ImbPipeline(
            [
                ("scaler", StandardScaler()),
                ("logistic_regression", LogisticRegression(random_state=42)),
            ]
        )

        lr_param_dist = {
            "logistic_regression__C": uniform(0.001, 100),
            "logistic_regression__penalty": ["l1", "l2", "elasticnet"],
            "logistic_regression__solver": ["saga"],
            "logistic_regression__l1_ratio": uniform(0, 1),
            "logistic_regression__max_iter": [500, 1000, 1500],
            "logistic_regression__class_weight": [
                "balanced",
                None,
                {0: 1, 1: 2},
                {0: 1, 1: 3},
            ],
            "logistic_regression__tol": [1e-5, 1e-4, 1e-3],
        }

        lr_random_search = RandomizedSearchCV(
            estimator=lr_pipeline,
            param_distributions=lr_param_dist,
            n_iter=50,
            scoring="f1",
            cv=5,
            verbose=2,
            random_state=42,
            n_jobs=-1,
        )

        logger.info("Starting hyperparameter tuning for Logistic Regression")
        lr_random_search.fit(X_train_resampled, y_train_resampled)
        logger.info(f"Best LR parameters: {lr_random_search.best_params_}")
        logger.info(f"Best LR F1 score: {lr_random_search.best_score_:.4f}")

        # Evaluate the best LR model
        best_lr_model = lr_random_search.best_estimator_
        y_pred = best_lr_model.predict(X_val)
        y_pred_proba = best_lr_model.predict_proba(X_val)[:, 1]

        metrics = evaluate_model(
            y_true=y_val,
            y_pred=y_pred,
            model_name="logistic_regression_tuned",
            y_pred_proba=y_pred_proba,
            model=best_lr_model,
        )

        if metrics["F1 Score"] > best_score:
            best_score = metrics["F1 Score"]
            best_model_name = "Logistic Regression (Tuned)"
            models["Logistic Regression (Tuned)"] = best_lr_model

        # Save LR parameters
        with open(
            os.path.join(
                "reports/evaluations", "logistic_regression_tuned_params.json"
            ),
            "w",
        ) as f:
            json.dump(lr_random_search.best_params_, f, indent=4)

        del lr_pipeline, lr_random_search
        gc.collect()

        # 5.5 XGBoost
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
                xgb_params={
                    "max_depth": 8,  # Increased from 6
                    "min_child_weight": 1,
                    "eta": 0.03,  # Decreased from 0.05
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "objective": "binary:logistic",
                    "eval_metric": ["auc", "error"],
                    "tree_method": "hist",
                    "alpha": 1,
                    "lambda": 1,
                    "gamma": 0.1,  # Added gamma parameter
                    "random_state": 42,
                },
                dtrain=dtrain,
                num_boost_round=500,  # Increased from 200
                evals=[(dtrain, "train"), (dval, "eval")],
                early_stopping_rounds=20,
                verbose_eval=False,
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


if __name__ == "__main__":
    main()
