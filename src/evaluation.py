# src/evaluation.py

import json
import logging
import os
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
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

from src.logger_config import setup_logger

# Initialize logger with default log level ("INFO") and JSON formatting enabled
logger = setup_logger(
    name="sepsis_prediction.evaluation",
    log_file="logs/evaluation.log",
    use_json=True,  # Ensure JSON logging is enabled
)


def setup_plot_style():
    """Set up the plotting style for all visualizations."""
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
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    report_dir: str = "reports/evaluations",
    y_pred_proba: Optional[np.ndarray] = None,
    model: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Comprehensive model evaluation function that calculates metrics and generates essential visualization plots.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted target values
    model_name : str
        Name of the model being evaluated
    report_dir : str, optional
        Directory to save evaluation reports
    y_pred_proba : np.ndarray, optional
        Predicted probabilities for the positive class
    model : Any, optional
        Trained model object for feature importance analysis

    Returns
    -------
    Dict[str, float]
        Dictionary containing all calculated metrics
    """
    try:
        setup_plot_style()
        os.makedirs(report_dir, exist_ok=True)

        # Calculate basic metrics
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, zero_division=0),
            "Mean Absolute Error": mean_absolute_error(y_true, y_pred),
            "Root Mean Squared Error": np.sqrt(mean_squared_error(y_true, y_pred)),
        }

        # Calculate AUC-ROC if probabilities are available
        if y_pred_proba is not None:
            try:
                metrics["AUC-ROC"] = roc_auc_score(y_true, y_pred_proba)
            except ValueError as e:
                logger.warning(f"Cannot calculate AUC-ROC: {e}")
                metrics["AUC-ROC"] = None

        # Log metrics
        log_metrics(logger, model_name, metrics)

        # Generate and save essential plots
        generate_evaluation_plots(
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            model=model,
            model_name=model_name,
            report_dir=report_dir,
            X_features=None,  # Assuming you no longer pass X_features
        )

        # Save metrics to JSON
        save_metrics_to_json(metrics, model_name, report_dir, logger)

        return metrics

    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise


def generate_evaluation_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray],
    model: Any,
    model_name: str,
    report_dir: str,
    X_features: Optional[Union[pd.DataFrame, np.ndarray]],
) -> None:
    """Generate essential evaluation plots for the model."""
    try:
        # Confusion Matrix (Raw and Normalized)
        plot_confusion_matrix(y_true, y_pred, model_name, report_dir, normalize=False)
        plot_confusion_matrix(y_true, y_pred, model_name, report_dir, normalize=True)

        # ROC and Precision-Recall Curves
        if y_pred_proba is not None:
            plot_roc_curve(y_true, y_pred_proba, model_name, report_dir)
            plot_precision_recall_curve(y_true, y_pred_proba, model_name, report_dir)

        # Feature Importance
        if model is not None:
            plot_feature_importance(model, model_name, report_dir)

    except Exception as e:
        logger.error(f"Error generating plots: {str(e)}")
        raise


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    report_dir: str,
    normalize: bool = False,
) -> None:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, normalize="true" if normalize else None)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(
        f"Confusion Matrix for {model_name} {'(Normalized)' if normalize else ''}"
    )
    plt.tight_layout()

    cm_type = "confusion_matrix_normalized" if normalize else "confusion_matrix"
    save_plot(plt, report_dir, f"{model_name}_{cm_type}.png")


def plot_roc_curve(
    y_true: np.ndarray, y_pred_proba: np.ndarray, model_name: str, report_dir: str
) -> None:
    """Plot and save ROC curve."""
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
    plt.title(f"ROC Curve for {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()

    save_plot(plt, report_dir, f"{model_name}_roc_curve.png")


def plot_precision_recall_curve(
    y_true: np.ndarray, y_pred_proba: np.ndarray, model_name: str, report_dir: str
) -> None:
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

    save_plot(plt, report_dir, f"{model_name}_precision_recall_curve.png")


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

            # Generate a color palette based on the number of features
            palette = sns.color_palette(
                "viridis", n_colors=len(feature_importance_df.head(20))
            )

            sns.barplot(
                x="importance",
                y="feature",
                data=feature_importance_df.head(20),
                palette=palette,
                dodge=False,
            )
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

            # Generate a color palette based on the number of features
            palette = sns.color_palette("viridis", n_colors=len(importance_df.head(20)))

            sns.barplot(
                x="importance",
                y="feature",
                data=importance_df.head(20),
                palette=palette,
                dodge=False,
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

    except AttributeError as ae:
        logger.warning(f"Feature importances not available for {model_name}: {ae}")
    except Exception as e:
        # Use the global logger
        logger.warning(f"Could not plot feature importances for {model_name}: {e}")


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


def save_plot(plt: plt, report_dir: str, filename: str, **kwargs) -> None:
    """Save plot to file and close it."""
    plt.savefig(os.path.join(report_dir, filename), **kwargs)
    plt.close()


def log_message(
    logger: Optional[logging.Logger], message: str, level: str = "info"
) -> None:
    """Log message using logger if available, otherwise print."""
    if logger:
        getattr(logger, level)(message)
    else:
        print(f"{level.upper()}: {message}")


def log_metrics(
    logger: Optional[logging.Logger], model_name: str, metrics: Dict[str, float]
) -> None:
    """Log evaluation metrics."""
    log_message(logger, f"\n{model_name} Evaluation:")
    for metric, value in metrics.items():
        if value is not None:
            log_message(logger, f"  {metric:<25} : {value:.4f}")
        else:
            log_message(logger, f"  {metric:<25} : Not Available")


def save_metrics_to_json(
    metrics: Dict[str, float],
    model_name: str,
    report_dir: str,
    logger: Optional[logging.Logger],
) -> None:
    """Save metrics to JSON file."""
    metrics_path = os.path.join(report_dir, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    log_message(logger, f"Saved evaluation metrics to {metrics_path}\n")
