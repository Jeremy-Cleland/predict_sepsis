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
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train, n_neighbors=10):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, params=None, num_round=100):
    if params is None:
        params = {
            "max_depth": 5,
            "eta": 0.3,
            # "silent": 1,  # Removed deprecated parameter
            "objective": "binary:logistic",
        }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    bst = xgb.train(params, dtrain, num_round)
    return bst


def predict_xgboost(bst, X_test):
    dtest = xgb.DMatrix(X_test)
    preds = bst.predict(dtest)
    predictions = [1 if pred >= 0.5 else 0 for pred in preds]
    return predictions
