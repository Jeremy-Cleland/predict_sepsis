 /Users/jeremy/miniforge3/envs/predictsepsis/bin/python /Users/jeremy/Documents/Development/sepsis_prediction/main.py
2024-12-07 02:20:32,108 - INFO - Starting Sepsis Prediction Pipeline
2024-12-07 02:20:32,110 - INFO - Loading raw data
2024-12-07 02:20:35,055 - INFO - Splitting data into training, validation, and testing sets
2024-12-07 02:20:50,950 - INFO - Preprocessing all datasets
/Users/jeremy/miniforge3/envs/predictsepsis/lib/python3.10/site-packages/sklearn/impute/_iterative.py:825: ConvergenceWarning: [IterativeImputer] Early stopping criterion not reached.
  warnings.warn(
2024-12-07 02:24:14,765 - INFO - Applying SMOTEENN to training data
/Users/jeremy/miniforge3/envs/predictsepsis/lib/python3.10/site-packages/imblearn/over_sampling/_smote/base.py:370: FutureWarning: The parameter `n_jobs` has been deprecated in 0.10 and will be removed in 0.12. You can pass an nearest neighbors estimator where `n_jobs` is already set instead.
  warnings.warn(
2024-12-07 02:27:52,521 - INFO - Original class distribution - Class 1: 10778.0, Class 0: 648977.0
2024-12-07 02:27:52,523 - INFO - Resampled class distribution - Class 1: 194693.0, Class 0: 631799.0
2024-12-07 02:27:52,523 - INFO - 
Training Random Forest...
2024-12-07 02:28:41,568 - INFO - Random Forest Evaluation Results:
2024-12-07 02:28:41,570 - INFO -   Accuracy             : 0.9742
2024-12-07 02:28:41,570 - INFO -   Precision            : 0.3252
2024-12-07 02:28:41,570 - INFO -   Recall               : 0.5377
2024-12-07 02:28:41,570 - INFO -   F1 Score             : 0.4053
2024-12-07 02:28:41,570 - INFO -   AUC-ROC              : 0.7596
2024-12-07 02:28:41,570 - INFO -   Mean Absolute Error  : 0.0258
2024-12-07 02:28:41,570 - INFO -   Root Mean Squared Error : 0.1606
random_forest Evaluation:
Accuracy: 0.9742
Precision: 0.3252
Recall: 0.5377
F1 Score: 0.4053
Mean Absolute Error: 0.0258
Root Mean Squared Error: 0.1606
AUC-ROC: 0.7596
Saved evaluation metrics to reports/evaluations/random_forest_metrics.json
Saved confusion matrix plot to reports/evaluations/random_forest_confusion_matrix.png

2024-12-07 02:28:42,039 - INFO - Saved evaluation artifacts for Random Forest

2024-12-07 02:28:42,141 - INFO - 
Training Naive Bayes...
2024-12-07 02:28:42,931 - INFO - Naive Bayes Evaluation Results:
2024-12-07 02:28:42,931 - INFO -   Accuracy             : 0.9171
2024-12-07 02:28:42,931 - INFO -   Precision            : 0.0792
2024-12-07 02:28:42,931 - INFO -   Recall               : 0.3834
2024-12-07 02:28:42,931 - INFO -   F1 Score             : 0.1313
2024-12-07 02:28:42,931 - INFO -   AUC-ROC              : 0.6547
2024-12-07 02:28:42,931 - INFO -   Mean Absolute Error  : 0.0829
2024-12-07 02:28:42,931 - INFO -   Root Mean Squared Error : 0.2880
naive_bayes Evaluation:
Accuracy: 0.9171
Precision: 0.0792
Recall: 0.3834
F1 Score: 0.1313
Mean Absolute Error: 0.0829
Root Mean Squared Error: 0.2880
AUC-ROC: 0.6547
Saved evaluation metrics to reports/evaluations/naive_bayes_metrics.json
Saved confusion matrix plot to reports/evaluations/naive_bayes_confusion_matrix.png

2024-12-07 02:28:43,180 - INFO - Saved evaluation artifacts for Naive Bayes

2024-12-07 02:28:43,250 - INFO - 
Training KNN...
2024-12-07 02:29:27,512 - INFO - KNN Evaluation Results:
2024-12-07 02:29:27,514 - INFO -   Accuracy             : 0.9654
2024-12-07 02:29:27,514 - INFO -   Precision            : 0.1982
2024-12-07 02:29:27,514 - INFO -   Recall               : 0.3674
2024-12-07 02:29:27,514 - INFO -   F1 Score             : 0.2575
2024-12-07 02:29:27,515 - INFO -   AUC-ROC              : 0.6713
2024-12-07 02:29:27,515 - INFO -   Mean Absolute Error  : 0.0346
2024-12-07 02:29:27,515 - INFO -   Root Mean Squared Error : 0.1861
knn Evaluation:
Accuracy: 0.9654
Precision: 0.1982
Recall: 0.3674
F1 Score: 0.2575
Mean Absolute Error: 0.0346
Root Mean Squared Error: 0.1861
AUC-ROC: 0.6713
Saved evaluation metrics to reports/evaluations/knn_metrics.json
Saved confusion matrix plot to reports/evaluations/knn_confusion_matrix.png

2024-12-07 02:29:27,811 - INFO - Saved evaluation artifacts for KNN

2024-12-07 02:29:56,958 - INFO - 
Training Logistic Regression...
2024-12-07 02:29:59,489 - INFO - Logistic Regression Evaluation Results:
2024-12-07 02:29:59,489 - INFO -   Accuracy             : 0.9145
2024-12-07 02:29:59,489 - INFO -   Precision            : 0.1380
2024-12-07 02:29:59,489 - INFO -   Recall               : 0.8064
2024-12-07 02:29:59,489 - INFO -   F1 Score             : 0.2357
2024-12-07 02:29:59,489 - INFO -   AUC-ROC              : 0.8613
2024-12-07 02:29:59,489 - INFO -   Mean Absolute Error  : 0.0855
2024-12-07 02:29:59,489 - INFO -   Root Mean Squared Error : 0.2924
logistic_regression Evaluation:
Accuracy: 0.9145
Precision: 0.1380
Recall: 0.8064
F1 Score: 0.2357
Mean Absolute Error: 0.0855
Root Mean Squared Error: 0.2924
AUC-ROC: 0.8613
Saved evaluation metrics to reports/evaluations/logistic_regression_metrics.json
Saved confusion matrix plot to reports/evaluations/logistic_regression_confusion_matrix.png

2024-12-07 02:29:59,776 - INFO - Saved evaluation artifacts for Logistic Regression

2024-12-07 02:29:59,879 - INFO - Training XGBoost model
[0]     train-auc:0.90165       train-error:0.23557     eval-auc:0.84440        eval-error:0.01635
[10]    train-auc:0.96043       train-error:0.16106     eval-auc:0.92541        eval-error:0.01940
[20]    train-auc:0.96701       train-error:0.09698     eval-auc:0.92918        eval-error:0.03257
[28]    train-auc:0.97183       train-error:0.07844     eval-auc:0.93511        eval-error:0.03524
xgboost Evaluation:
Accuracy: 0.9648
Precision: 0.2419
Recall: 0.5416
F1 Score: 0.3344
Mean Absolute Error: 0.0352
Root Mean Squared Error: 0.1877
AUC-ROC: 0.9351
Saved evaluation metrics to reports/evaluations/xgboost_metrics.json
Saved confusion matrix plot to reports/evaluations/xgboost_confusion_matrix.png

2024-12-07 02:30:04,761 - INFO - Saved evaluation artifacts for XGBoost

2024-12-07 02:30:04,807 - INFO - 
Performing final evaluation with best model: Random Forest
2024-12-07 02:30:05,173 - INFO - final_random_forest Evaluation:
2024-12-07 02:30:05,173 - INFO - Accuracy: 0.9870
2024-12-07 02:30:05,173 - INFO - Precision: 0.6090
2024-12-07 02:30:05,173 - INFO - Recall: 0.5371
2024-12-07 02:30:05,174 - INFO - F1 Score: 0.5708
2024-12-07 02:30:05,174 - INFO - Mean Absolute Error: 0.0130
2024-12-07 02:30:05,174 - INFO - Root Mean Squared Error: 0.1141
2024-12-07 02:30:05,174 - INFO - AUC-ROC: 0.7657
2024-12-07 02:30:05,298 - INFO - Saved evaluation metrics to reports/evaluations/final_random_forest_metrics.json
2024-12-07 02:30:05,298 - INFO - Saved confusion matrix plot to reports/evaluations/final_random_forest_confusion_matrix.png

2024-12-07 02:30:05,298 - INFO - Saving the best model (Random Forest)
2024-12-07 02:30:05,414 - INFO - Model saved to models/best_model_random_forest.pkl
2024-12-07 02:30:05,460 - INFO - Sepsis Prediction Pipeline completed successfully.