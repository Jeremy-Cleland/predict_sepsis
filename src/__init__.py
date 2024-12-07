# src/__init__.py

from .data_processing import (
    load_data,
    load_processed_data,
    split_data,
)
from .evaluation import (
    evaluate_model,
    plot_calibration_curve,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_cumulative_gains,
    plot_feature_correlation_heatmap,
    plot_feature_importance,
    plot_learning_curve_custom,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_shap_summary,
)
from .feature_engineering import preprocess_data
from .models import (
    predict_xgboost,
    train_knn,
    train_logistic_regression,
    train_naive_bayes,
    train_random_forest,
    train_xgboost,
)
from .utils import (
    corr_matrix,
    diagnostic_plots,
    setup_logger,
    try_gaussian,
)

__all__ = [
    "load_data",
    "load_processed_data",
    "split_data",
    "preprocess_data",
    "train_random_forest",
    "train_naive_bayes",
    "train_knn",
    "train_logistic_regression",
    "train_xgboost",
    "predict_xgboost",
    "evaluate_model",
    "plot_confusion_matrix",
    "plot_confusion_matrix_normalized",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_calibration_curve",
    "plot_cumulative_gains",
    "plot_lift_chart",
    "plot_feature_importance",
    "plot_shap_summary",
    "plot_learning_curve_custom",
    "plot_learning_curves",
    "plot_feature_correlation_heatmap",
    "plot_class_distribution",
    "corr_matrix",
    "diagnostic_plots",
    "try_gaussian",
    "setup_logger",
]
