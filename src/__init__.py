# src/__init__.py

from .data_processing import (
    load_data,
    load_processed_data,
    split_data,
)
from .evaluation import (
    evaluate_model,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_feature_correlation_heatmap,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_roc_curve,
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

# from .logger_config import get_logger, setup_logger

# from .logging_utils import log_phase, log_memory, log_dataframe_info, log_step
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
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_feature_importance",
    "corr_matrix",
    "diagnostic_plots",
    "try_gaussian",
    "setup_logger",
    plot_class_distribution,
    plot_feature_correlation_heatmap,
    # "setup_logger",
    # "get_logger",
    # "log_dataframe_info",
    # "log_memory",
    # "log_phase",
    # "log_step",
]
