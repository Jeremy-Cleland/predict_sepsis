# src/__init__.py

from .data_processing import (
    get_data_ready,  # Ensure this is included
    load_data,
    load_processed_data,
    split_data,
)
from .evaluation import evaluate_model
from .feature_engineering import preprocess_data
from .models import (
    predict_xgboost,
    train_knn,
    train_logistic_regression,
    train_naive_bayes,
    train_random_forest,
    train_xgboost,
)
from .utils import corr_matrix, diagnostic_plots, setup_logger, try_gaussian

__all__ = [
    "load_data",
    "split_data",
    "load_processed_data",
    "get_data_ready",
    "preprocess_data",
    "train_random_forest",
    "train_naive_bayes",
    "train_knn",
    "train_logistic_regression",
    "train_xgboost",
    "predict_xgboost",
    "evaluate_model",
    "corr_matrix",
    "diagnostic_plots",
    "try_gaussian",
    "setup_logger",
]
