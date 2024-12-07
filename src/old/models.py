# src/models.py

import logging

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def train_random_forest(X_train, y_train, n_estimators=300, random_state=0):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_naive_bayes(X_train, y_train):
    """No hyperparameters to tune for GaussianNB - it's naturally simple"""
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train, n_neighbors=5):
    """Reduced n_neighbors for better local sensitivity"""
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights="distance",  # Weight points by distance for better accuracy
        algorithm="auto",  # Automatically choose best algorithm
        leaf_size=30,  # Default leaf size for efficient queries
        p=2,  # Euclidean distance metric
        n_jobs=-1,  # Use all available cores
    )
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,  # Inverse of regularization strength
        penalty="l2",  # Ridge regularization
        solver="lbfgs",  # Efficient solver for medium-sized datasets
        tol=1e-4,  # Tolerance for stopping criteria
        random_state=42,  # For reproducibility
        n_jobs=-1,  # Use all available cores
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, params=None, num_round=100):
    """Train XGBoost model with optional early stopping."""
    if params is None:
        params = {
            "max_depth": 6,  # Slightly deeper trees
            "min_child_weight": 1,  # Minimum sum of instance weight in a child
            "eta": 0.1,  # Lower learning rate for better generalization
            "subsample": 0.8,  # Prevent overfitting
            "colsample_bytree": 0.8,  # Prevent overfitting
            "objective": "binary:logistic",
            "eval_metric": ["auc", "error"],  # Multiple evaluation metrics
            "alpha": 1,  # L1 regularization
            "lambda": 1,  # L2 regularization
            "tree_method": "hist",  # Faster histogram-based algorithm
            "random_state": 42,  # For reproducibility
        }

    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Train without early stopping since we don't have validation data
    bst = xgb.train(params, dtrain, num_boost_round=num_round)
    return bst


def predict_xgboost(bst, X_test):
    dtest = xgb.DMatrix(X_test)
    preds = bst.predict(dtest)
    return (preds >= 0.5).astype(int)  # More efficient prediction conversion
