# src/evaluation.py

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve

from .utils import setup_logger

# Corrected the log_file path
logger = setup_logger("logs/evaluation.log")


def setup_plot_style():
    """Set up the plotting style."""
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (8, 6),
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "font.size": 12,
            "lines.linewidth": 2,
        }
    )


def evaluate_model(
    y_true,
    y_pred,
    model_name,
    report_dir="reports/evaluations",
    logger=None,
    y_pred_proba=None,
    model=None,
    X_features=None,  # Added parameter for SHAP plots
):
    """
    Calculate, print, plot, and save evaluation metrics and various plots.

    Parameters:
    -----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    model_name : str
        Name of the model.
    report_dir : str, optional
        Directory to save evaluation reports, by default "reports/evaluations".
    logger : logging.Logger, optional
        Logger for logging information, by default None.
    y_pred_proba : array-like, optional
        Predicted probabilities for the positive class, used for AUC-ROC and other probabilistic plots, by default None.
    model : sklearn estimator or xgboost.Booster, optional
        The trained model to extract feature importances from, by default None.
    X_features : pd.DataFrame or np.ndarray, optional
        Feature set used for SHAP plots, by default None.
    """

    setup_plot_style()

    # Ensure report directory exists
    os.makedirs(report_dir, exist_ok=True)

    # Calculate metrics
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "Mean Absolute Error": mean_absolute_error(y_true, y_pred),
        "Root Mean Squared Error": np.sqrt(mean_squared_error(y_true, y_pred)),
    }

    # Calculate AUC-ROC using probabilities if provided, else use binary predictions
    if y_pred_proba is not None:
        try:
            auc_roc = roc_auc_score(y_true, y_pred_proba)
            metrics["AUC-ROC"] = auc_roc
        except ValueError as e:
            if logger:
                logger.warning(f"Cannot calculate AUC-ROC: {e}")
            else:
                print(f"Warning: Cannot calculate AUC-ROC: {e}")
            metrics["AUC-ROC"] = None
    else:
        try:
            auc_roc = roc_auc_score(y_true, y_pred)
            metrics["AUC-ROC"] = auc_roc
        except ValueError as e:
            if logger:
                logger.warning(f"Cannot calculate AUC-ROC: {e}")
            else:
                print(f"Warning: Cannot calculate AUC-ROC: {e}")
            metrics["AUC-ROC"] = None

    # Log or print metrics
    if logger:
        logger.info(f"{model_name} Evaluation:")
        for metric, value in metrics.items():
            if value is not None:
                logger.info(f"  {metric:<25} : {value:.4f}")
            else:
                logger.info(f"  {metric:<25} : Not Available")
    else:
        print(f"{model_name} Evaluation:")
        for metric, value in metrics.items():
            if value is not None:
                print(f"  {metric:<25} : {value:.4f}")
            else:
                print(f"  {metric:<25} : Not Available")

    # Plot Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, model_name, report_dir, normalize=False)
    plot_confusion_matrix(y_true, y_pred, model_name, report_dir, normalize=True)

    # Plot ROC Curve
    if y_pred_proba is not None:
        plot_roc_curve(y_true, y_pred_proba, model_name, report_dir)

    # Plot Precision-Recall Curve
    if y_pred_proba is not None:
        plot_precision_recall_curve(y_true, y_pred_proba, model_name, report_dir)

    # Plot Calibration Curve
    if y_pred_proba is not None:
        plot_calibration_curve(y_true, y_pred_proba, model_name, report_dir)

    # Plot Cumulative Gains and Lift Chart
    if y_pred_proba is not None:
        plot_cumulative_gains(y_true, y_pred_proba, model_name, report_dir)

    # Plot Feature Importance
    if model is not None:
        plot_feature_importance(model, model_name, report_dir)

    # Plot SHAP Summary Plot
    if model is not None and y_pred_proba is not None and X_features is not None:
        plot_shap_summary(y_true, model, model_name, report_dir, X_features)

    # Save metrics to JSON
    metrics_path = os.path.join(report_dir, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    if logger:
        logger.info(f"Saved evaluation metrics to {metrics_path}\n")
    else:
        print(f"Saved evaluation metrics to {metrics_path}\n")


def plot_confusion_matrix(y_true, y_pred, model_name, report_dir, normalize=False):
    """Plot and save the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(
        f"Confusion Matrix for {model_name} {'(Normalized)' if normalize else ''}"
    )
    plt.tight_layout()

    # Save confusion matrix plot
    cm_type = "confusion_matrix_normalized" if normalize else "confusion_matrix"
    cm_path = os.path.join(report_dir, f"{model_name}_{cm_type}.png")
    plt.savefig(cm_path)
    plt.close()
    # Removed unused variables: title, suffix, log_msg


def plot_roc_curve(y_true, y_pred_proba, model_name, report_dir):
    """Plot and save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Receiver Operating Characteristic for {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()

    # Save ROC plot
    roc_path = os.path.join(report_dir, f"{model_name}_roc_curve.png")
    plt.savefig(roc_path)
    plt.close()


def plot_precision_recall_curve(y_true, y_pred_proba, model_name, report_dir):
    """Plot and save the Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    average_precision = average_precision_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        color="purple",
        lw=2,
        label=f"PR curve (AP = {average_precision:.2f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve for {model_name}")
    plt.legend(loc="lower left")
    plt.tight_layout()

    # Save Precision-Recall plot
    pr_path = os.path.join(report_dir, f"{model_name}_precision_recall_curve.png")
    plt.savefig(pr_path)
    plt.close()


def plot_calibration_curve(y_true, y_pred_proba, model_name, report_dir):
    """Plot and save the Calibration curve."""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=10
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        mean_predicted_value, fraction_of_positives, "s-", label="Calibration curve"
    )
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Curve for {model_name}")
    plt.legend(loc="upper left")
    plt.tight_layout()

    # Save Calibration plot
    calib_path = os.path.join(report_dir, f"{model_name}_calibration_curve.png")
    plt.savefig(calib_path)
    plt.close()


def plot_feature_importance(model, model_name, report_dir):
    """Plot and save feature importance."""
    plt.figure(figsize=(10, 8))

    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_names = (
                model.get_booster().feature_names
                if hasattr(model, "get_booster")
                else None
            )
            if feature_names is None:
                feature_names = [f"Feature {i}" for i in range(len(importances))]

            feature_importance_df = pd.DataFrame(
                {"feature": feature_names, "importance": importances}
            ).sort_values(by="importance", ascending=False)

            sns.barplot(
                x="importance",
                y="feature",
                data=feature_importance_df.head(20),
                hue="feature",
                dodge=False,  # Prevents overlapping bars
                palette="viridis",
            )
            plt.legend([], [], frameon=False)  # Removes the legend
            plt.title(f"Top 20 Feature Importances for {model_name}")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.tight_layout()

            # Save Feature Importance plot
            fi_path = os.path.join(report_dir, f"{model_name}_feature_importance.png")
            plt.savefig(fi_path)
            plt.close()

        elif isinstance(model, xgb.Booster):
            importances = model.get_score(importance_type="weight")
            importance_df = pd.DataFrame(
                {
                    "feature": list(importances.keys()),
                    "importance": list(importances.values()),
                }
            ).sort_values("importance", ascending=False)

            sns.barplot(
                x="importance",
                y="feature",
                data=importance_df.head(20),
                palette="viridis",
            )
            plt.title(f"Top 20 Feature Importances for {model_name}")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.tight_layout()

            # Save Feature Importance plot
            fi_path = os.path.join(report_dir, f"{model_name}_feature_importance.png")
            plt.savefig(fi_path)
            plt.close()
        else:
            raise AttributeError("Model does not have feature_importances_ attribute.")

    except Exception as e:
        if logger:
            logger.warning(f"Could not plot feature importances for {model_name}: {e}")
        else:
            print(f"Warning: Could not plot feature importances for {model_name}: {e}")


def plot_learning_curve_custom(
    estimator, X, y, model_name, report_dir, cv=5, scoring="f1"
):
    """Plot and save the learning curve."""
    plt.figure(figsize=(8, 6))
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        scoring=scoring,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )

    plt.xlabel("Training examples")
    plt.ylabel(scoring.capitalize())
    plt.title(f"Learning Curve for {model_name}")
    plt.legend(loc="best")
    plt.tight_layout()

    # Save Learning Curve plot
    lc_path = os.path.join(report_dir, f"{model_name}_learning_curve.png")
    plt.savefig(lc_path)
    plt.close()


def plot_feature_correlation(
    df,
    model_name,
    report_dir,
    top_n=20,
):
    """Plot and save the feature correlation heatmap."""
    plt.figure(figsize=(12, 10))
    corr = df.corr().abs()
    # Select upper triangle
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    # Find features with correlation greater than a threshold
    threshold = 0.6
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Compute correlation matrix of selected features
    corr_selected = corr.drop(columns=to_drop).drop(index=to_drop)

    sns.heatmap(corr_selected, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(f"Feature Correlation Heatmap for {model_name}")
    plt.tight_layout()

    # Save Feature Correlation Heatmap
    fc_path = os.path.join(report_dir, f"{model_name}_feature_correlation_heatmap.png")
    plt.savefig(fc_path)
    plt.close()


def plot_cumulative_gains(y_true, y_pred_proba, model_name, report_dir):
    """Plot and save the Cumulative Gains and Lift Chart."""
    # Create a dataframe with true values and predicted probabilities
    data = pd.DataFrame({"y_true": y_true, "y_pred_proba": y_pred_proba})
    data = data.sort_values(by="y_pred_proba", ascending=False).reset_index(drop=True)
    data["cumulative_true"] = data["y_true"].cumsum()
    total_true = data["y_true"].sum()
    data["gain"] = data["cumulative_true"] / total_true
    data["percentage"] = np.linspace(0, 1, len(data))

    # Cumulative Gains Chart
    plt.figure(figsize=(8, 6))
    plt.plot(data["percentage"], data["gain"], label="Model", color="blue")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Baseline", color="gray")
    plt.xlabel("Percentage of Sample")
    plt.ylabel("Gain")
    plt.title(f"Cumulative Gains Chart for {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()

    # Save Cumulative Gains Chart
    cg_path = os.path.join(report_dir, f"{model_name}_cumulative_gains.png")
    plt.savefig(cg_path)
    plt.close()

    # Lift Chart
    plt.figure(figsize=(8, 6))
    plt.plot(
        data["percentage"],
        data["gain"] / data["percentage"],
        label="Model",
        color="green",
    )
    plt.plot([0, 1], [1, 1], linestyle="--", label="Baseline", color="gray")
    plt.xlabel("Percentage of Sample")
    plt.ylabel("Lift")
    plt.title(f"Lift Chart for {model_name}")
    plt.legend(loc="upper left")
    plt.tight_layout()

    # Save Lift Chart
    lift_path = os.path.join(report_dir, f"{model_name}_lift_chart.png")
    plt.savefig(lift_path)
    plt.close()


def plot_shap_summary(y_true, model, model_name, report_dir, X_features):
    """Plot and save the SHAP summary plot."""
    try:
        # Initialize the SHAP explainer
        if isinstance(model, xgb.Booster):
            # For XGBoost Booster
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_features)
        else:
            # For scikit-learn models
            explainer = shap.Explainer(model, X_features)
            shap_values = explainer(X_features)

        plt.figure()
        shap.summary_plot(
            shap_values, X_features, show=False, plot_type="bar", max_display=20
        )
        plt.title(f"SHAP Summary Plot for {model_name}")
        plt.tight_layout()

        # Save SHAP summary plot
        shap_path = os.path.join(report_dir, f"{model_name}_shap_summary.png")
        plt.savefig(shap_path, bbox_inches="tight")
        plt.close()

    except Exception as e:
        if logger:
            logger.warning(f"Could not plot SHAP values for {model_name}: {e}")
        else:
            print(f"Warning: Could not plot SHAP values for {model_name}: {e}")


def plot_class_distribution(y, model_name, report_dir, title_suffix=""):
    """Plot and save the class distribution."""
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y, hue=y, palette="viridis", legend=False)
    plt.title(f"Class Distribution for {model_name} {title_suffix}")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()

    # Save the plot
    suffix = title_suffix.replace(" ", "_") if title_suffix else ""
    cd_path = os.path.join(report_dir, f"{model_name}_class_distribution_{suffix}.png")
    plt.savefig(cd_path)
    plt.close()


def plot_feature_correlation_heatmap(
    df,
    model_name,
    report_dir,
    top_n=20,
):
    """Plot and save the feature correlation heatmap."""
    plt.figure(figsize=(12, 10))
    corr = df.corr().abs()
    # Select upper triangle
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    # Find features with correlation greater than a threshold
    threshold = 0.6
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Compute correlation matrix of selected features
    corr_selected = corr.drop(columns=to_drop).drop(index=to_drop)

    sns.heatmap(corr_selected, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(f"Feature Correlation Heatmap for {model_name}")
    plt.tight_layout()

    # Save Feature Correlation Heatmap
    fc_path = os.path.join(report_dir, f"{model_name}_feature_correlation_heatmap.png")
    plt.savefig(fc_path)
    plt.close()
