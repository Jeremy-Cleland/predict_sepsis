# src/evaluation.py

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    y_true, y_pred, model_name, report_dir="reports/evaluations", logger=None
):
    """Calculate, print, plot, and save evaluation metrics and confusion matrix."""

    # Calculate metrics
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "AUC-ROC": roc_auc_score(y_true, y_pred),
        "Mean Absolute Error": mean_absolute_error(y_true, y_pred),
        "Root Mean Squared Error": np.sqrt(mean_squared_error(y_true, y_pred)),
    }

    # Log or print metrics
    if logger:
        logger.info(f"{model_name} Evaluation:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
    else:
        print(f"{model_name} Evaluation:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.tight_layout()

    # Save confusion matrix plot
    os.makedirs(report_dir, exist_ok=True)
    cm_path = os.path.join(report_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # Save metrics to JSON
    metrics_path = os.path.join(report_dir, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # Log saving paths
    if logger:
        logger.info(f"Saved evaluation metrics to {metrics_path}")
        logger.info(f"Saved confusion matrix plot to {cm_path}\n")
    else:
        print(f"Saved evaluation metrics to {metrics_path}")
        print(f"Saved confusion matrix plot to {cm_path}\n")
