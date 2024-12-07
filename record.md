
# Performance Tracking and Evaluation

## Initial Configuration 

Resampling the minority class (SepsisLabel = 1) to match the majority class (SepsisLabel = 0), resulting in both classes having 574,526 samples.

### Code

```python
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

### Model: **Random Forest**

### Performance Metrics

- **Accuracy:** 0.9452
- **Precision:** 0.1166
- **Recall:** 0.2610
- **F1 Score:** 0.1612
- **AUC-ROC:** 0.6101
- **Mean Absolute Error:** 0.0548
- **Root Mean Squared Error:** 0.2342

---

## (NOT USING)Experiment 1: Using SMOTE with Balanced Ratio

### Code

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

### Changes

- Applied **SMOTE** with a 1:2 sampling strategy (minority:majority).

### Performance Metrics

- **Accuracy:** 0.9493
- **Precision:** 0.1269
- **Recall:** 0.2570
- **F1 Score:** 0.1699
- **AUC-ROC:** 0.6103
- **Mean Absolute Error:** 0.0507
- **Root Mean Squared Error:** 0.2251

---

## Experiment 2: Using Hybrid Sampling (SMOTEENN)

### Code

```python
from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)
```

### Changes

- Combined **SMOTE** with **ENN** to refine the dataset by removing noisy instances.

### Performance Metrics

2024-12-06 23:58:32,899 - INFO - final_random_forest Evaluation:
2024-12-06 23:58:32,900 - INFO - Accuracy: 0.9402
2024-12-06 23:58:32,900 - INFO - Precision: 0.1112
2024-12-06 23:58:32,900 - INFO - Recall: 0.2809
2024-12-06 23:58:32,900 - INFO - F1 Score: 0.1594
2024-12-06 23:58:32,900 - INFO - AUC-ROC: 0.6173
2024-12-06 23:58:32,900 - INFO - Mean Absolute Error: 0.0598
2024-12-06 23:58:32,900 - INFO - Root Mean Squared Error: 0.2446

---

## (NOT USING) Experiment 3: Using ADASYN for Advanced Oversampling

### Code

```python
from imblearn.over_sampling import ADASYN

adasyn = ADASYN(random_state=42)
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)
```

### Changes

- Applied **ADASYN**, which generates synthetic samples proportionally based on the distribution of minority class instances.

### Performance Metrics

Performing final evaluation with best model: Naive Bayes
2024-12-07 00:02:47,977 - INFO - final_naive_bayes Evaluation:
2024-12-07 00:02:47,977 - INFO - Accuracy: 0.8567
2024-12-07 00:02:47,977 - INFO - Precision: 0.0741
2024-12-07 00:02:47,977 - INFO - Recall: 0.5307
2024-12-07 00:02:47,977 - INFO - F1 Score: 0.1301
2024-12-07 00:02:47,977 - INFO - AUC-ROC: 0.6971
2024-12-07 00:02:47,977 - INFO - Mean Absolute Error: 0.1433
2024-12-07 00:02:47,977 - INFO - Root Mean Squared Error: 0.3785

---
## Experiment 4: Using Hybrid Sampling (SMOTEENN) with Custom Sampling Strategy 

### Code

```python

from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN(sampling_strategy=0.4, random_state=42)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)
```

### Changes

- Combined **SMOTE** with **ENN** with a custom sampling strategy of 0.4 to balance the classes effectively.



### Performance Metrics

Performing final evaluation with best model: Random Forest
2024-12-07 00:12:24,166 - INFO - final_random_forest Evaluation:
2024-12-07 00:12:24,167 - INFO - Accuracy: 0.9490
2024-12-07 00:12:24,167 - INFO - Precision: 0.1274
2024-12-07 00:12:24,167 - INFO - Recall: 0.2614
2024-12-07 00:12:24,167 - INFO - F1 Score: 0.1713
2024-12-07 00:12:24,167 - INFO - AUC-ROC: 0.6122
2024-12-07 00:12:24,167 - INFO - Mean Absolute Error: 0.0510
2024-12-07 00:12:24,167 - INFO - Root Mean Squared Error: 0.2259


#### **1.1. Advanced Missing Value Imputation**

- **Objective:** Improve the accuracy of missing value imputation to retain valuable information.

- **Experiment 1: Iterative Imputer**

  ```python
  from sklearn.experimental import enable_iterative_imputer
  from sklearn.impute import IterativeImputer
  
  imputer = IterativeImputer(random_state=42)
  X_train_imputed = imputer.fit_transform(X_train)
  X_val_imputed = imputer.transform(X_val)
  X_test_imputed = imputer.transform(X_test)
  ```

**Full Function**
    
```python
def fill_missing_values(df):
    """Impute missing values using IterativeImputer."""
    # Create a copy of the dataframe
    df_copy = df.copy()
    
    # Separate Patient_ID and categorical columns
    id_column = df_copy['Patient_ID']
    categorical_columns = ['Gender', 'Unit1', 'Unit2']
    categorical_data = df_copy[categorical_columns]
    
    # Get numerical columns for imputation
    numerical_columns = df_copy.select_dtypes(include=[np.number]).columns
    numerical_data = df_copy[numerical_columns]
    
    # Initialize and fit the IterativeImputer
    imputer = IterativeImputer(
        random_state=42,
        max_iter=10,
        initial_strategy='mean',
        skip_complete=True
    )
    
    # Perform imputation on numerical columns
    imputed_numerical = pd.DataFrame(
        imputer.fit_transform(numerical_data),
        columns=numerical_columns,
        index=df_copy.index
    )
    
    # Combine the imputed numerical data with categorical data
    result = pd.concat([id_column, categorical_data, imputed_numerical], axis=1)
    
    return result

```

- **Performance Metrics**




```python
def fill_missing_values(df):
    """Impute missing values using IterativeImputer with optimized convergence criteria."""
    df_copy = df.copy()

    # Separate Patient_ID and categorical columns
    id_column = df_copy["Patient_ID"]
    categorical_columns = ["Gender", "Unit1", "Unit2"]
    categorical_data = df_copy[categorical_columns]

    # Get numerical columns for imputation
    numerical_columns = df_copy.select_dtypes(include=[np.number]).columns
    numerical_data = df_copy[numerical_columns]

    # Initialize and fit the IterativeImputer with optimized parameters
    imputer = IterativeImputer(
        random_state=42,
        max_iter=30,          # Increased from 10 to 30
        tol=1e-3,            # Default is 1e-3
        n_nearest_features=5, # Limit number of features to use for imputation
        initial_strategy="mean",
        min_value=numerical_data.min(),  # Set minimum allowed value
        max_value=numerical_data.max(),  # Set maximum allowed value
        verbose=2,           # Add verbosity to monitor convergence
        imputation_order='ascending'  # Try different orders
    )

    # Perform imputation on numerical columns
    imputed_numerical = pd.DataFrame(
        imputer.fit_transform(numerical_data),
        columns=numerical_columns,
        index=df_copy.index,
    )

    # Combine the imputed numerical data with categorical data
    df = pd.concat([id_column, categorical_data, imputed_numerical], axis=1)

    return df

```

```
# For strict convergence (slower but more accurate)
strict_imputer = IterativeImputer(
    tol=1e-4,
    max_iter=50,
    verbose=2
)

# For moderate convergence (balanced approach)
moderate_imputer = IterativeImputer(
    tol=1e-3,
    max_iter=30,
    verbose=2
)

# For loose convergence (faster but less accurate)
loose_imputer = IterativeImputer(
    tol=1e-2,
    max_iter=20,
    verbose=2
)

```

#### **1.2. Feature Scaling Alternatives**

- **Objective:** Explore different scaling methods to see if they benefit model performance.

- **Experiment 2: Robust Scaler**

  ```python
  from sklearn.preprocessing import RobustScaler
  
  scaler = RobustScaler()
  X_train_scaled = scaler.fit_transform(X_train_imputed)
  X_val_scaled = scaler.transform(X_val_imputed)
  X_test_scaled = scaler.transform(X_test_imputed)
  ```

- **Performance Metrics**

Performing final evaluation with best model: Random Forest
2024-12-07 00:22:41,399 - INFO - final_random_forest Evaluation:
2024-12-07 00:22:41,399 - INFO - Accuracy: 0.9490
2024-12-07 00:22:41,399 - INFO - Precision: 0.1274
2024-12-07 00:22:41,399 - INFO - Recall: 0.2614
2024-12-07 00:22:41,399 - INFO - F1 Score: 0.1713
2024-12-07 00:22:41,399 - INFO - AUC-ROC: 0.6122
2024-12-07 00:22:41,399 - INFO - Mean Absolute Error: 0.0510
2024-12-07 00:22:41,399 - INFO - Root Mean Squared Error: 0.2259

#### **1.3. Encoding Categorical Variables Effectively**

- **Objective:** Improve the representation of categorical variables to capture more information.

- **Experiment 3: Target Encoding for `Unit1` and `Unit2`**

  ```python
  import category_encoders as ce
  
  encoder = ce.TargetEncoder(cols=['Unit1', 'Unit2'])
  X_train_encoded = encoder.fit_transform(X_train_scaled, y_train)
  X_val_encoded = encoder.transform(X_val_scaled)
  X_test_encoded = encoder.transform(X_test_scaled)
  ```

- **Performance Metrics**

  - **Accuracy:** [Update after evaluation]
  - **Precision:** [Update after evaluation]
  - **Recall:** [Update after evaluation]
  - **F1 Score:** [Update after evaluation]
  - **AUC-ROC:** [Update after evaluation]
  - **Mean Absolute Error:** [Update after evaluation]
  - **Root Mean Squared Error:** [Update after evaluation]

---

### **2. Feature Engineering Enhancements**

Enhancing feature engineering can provide the model with more informative inputs.

```python
import category_encoders as ce

def target_encode(df, y, columns, X_val=None, y_val=None, X_test=None):
    """
    Apply Target Encoding for specified categorical columns.

    Args:
        df (pd.DataFrame): Training DataFrame.
        y (pd.Series): Target variable for training.
        columns (list): List of columns to target encode.
        X_val (pd.DataFrame, optional): Validation DataFrame.
        y_val (pd.Series, optional): Target variable for validation.
        X_test (pd.DataFrame, optional): Test DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        Encoded training, validation, and test DataFrames.
    """
    encoder = ce.TargetEncoder(cols=columns)
    df_encoded = encoder.fit_transform(df, y)
    X_val_encoded = encoder.transform(X_val) if X_val is not None else None
    X_test_encoded = encoder.transform(X_test) if X_test is not None else None
    return df_encoded, X_val_encoded, X_test_encoded
```
Usage Example in Preprocessing:

Add this step to your preprocess_data pipeline:

# Target encoding for Unit1 and Unit2

```python
target_columns = ["Unit1", "Unit2"]
X_train_encoded, X_val_encoded, X_test_encoded = target_encode(
    X_train, y_train, target_columns, X_val=X_val, y_val=y_val, X_test=X_test
)
```




















#### **2.1. Temporal Feature Engineering**

- **Objective:** Capture temporal dependencies and trends in the data.

- **Experiment 4: Rolling Statistics and Trend Features**

  ```python
  # Example for Heart Rate (HR)
  df_train['HR_roll_mean'] = df_train.groupby('Patient_ID')['HR'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
  df_train['HR_diff'] = df_train.groupby('Patient_ID')['HR'].transform(lambda x: x.diff())
  
  df_val['HR_roll_mean'] = df_val.groupby('Patient_ID')['HR'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
  df_val['HR_diff'] = df_val.groupby('Patient_ID')['HR'].transform(lambda x: x.diff())
  
  df_test['HR_roll_mean'] = df_test.groupby('Patient_ID')['HR'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
  df_test['HR_diff'] = df_test.groupby('Patient_ID')['HR'].transform(lambda x: x.diff())
  ```

- **Performance Metrics**

  - **Accuracy:** [Update after evaluation]
  - **Precision:** [Update after evaluation]
  - **Recall:** [Update after evaluation]
  - **F1 Score:** [Update after evaluation]
  - **AUC-ROC:** [Update after evaluation]
  - **Mean Absolute Error:** [Update after evaluation]
  - **Root Mean Squared Error:** [Update after evaluation]

#### **2.2. Interaction Features**

- **Objective:** Capture interactions between different vital signs and lab values.

- **Experiment 5: Create Interaction Terms**

  ```python
  # Example Interaction Features
  df_train['HR_O2Sat'] = df_train['HR'] * df_train['O2Sat']
  df_train['Temp_MAP'] = df_train['Temp'] * df_train['MAP']
  
  df_val['HR_O2Sat'] = df_val['HR'] * df_val['O2Sat']
  df_val['Temp_MAP'] = df_val['Temp'] * df_val['MAP']
  
  df_test['HR_O2Sat'] = df_test['HR'] * df_test['O2Sat']
  df_test['Temp_MAP'] = df_test['Temp'] * df_test['MAP']
  ```

- **Performance Metrics**

  - **Accuracy:** [Update after evaluation]
  - **Precision:** [Update after evaluation]
  - **Recall:** [Update after evaluation]
  - **F1 Score:** [Update after evaluation]
  - **AUC-ROC:** [Update after evaluation]
  - **Mean Absolute Error:** [Update after evaluation]
  - **Root Mean Squared Error:** [Update after evaluation]

#### **2.3. Dimensionality Reduction**

- **Objective:** Reduce feature space to eliminate multicollinearity and improve model efficiency.

- **Experiment 6: Principal Component Analysis (PCA)**

  ```python
  from sklearn.decomposition import PCA
  
  pca = PCA(n_components=20, random_state=42)
  X_train_pca = pca.fit_transform(X_train_encoded)
  X_val_pca = pca.transform(X_val_encoded)
  X_test_pca = pca.transform(X_test_encoded)
  ```

- **Performance Metrics**

  - **Accuracy:** [Update after evaluation]
  - **Precision:** [Update after evaluation]
  - **Recall:** [Update after evaluation]
  - **F1 Score:** [Update after evaluation]
  - **AUC-ROC:** [Update after evaluation]
  - **Mean Absolute Error:** [Update after evaluation]
  - **Root Mean Squared Error:** [Update after evaluation]

---

### **3. Class Imbalance Handling Refinements**

You've already experimented with various oversampling techniques. Continue refining to optimize balance and model performance.

#### **3.1. Experiment 2: SMOTEENN**

- **Objective:** Combine SMOTE oversampling with Edited Nearest Neighbors (ENN) to clean the dataset.

- **Code**

  ```python
  from imblearn.combine import SMOTEENN
  
  smote_enn = SMOTEENN(sampling_strategy=0.5, random_state=42)
  X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)
  ```

- **Performance Metrics**

  - **Accuracy:** [Update after evaluation]
  - **Precision:** [Update after evaluation]
  - **Recall:** [Update after evaluation]
  - **F1 Score:** [Update after evaluation]
  - **AUC-ROC:** [Update after evaluation]
  - **Mean Absolute Error:** [Update after evaluation]
  - **Root Mean Squared Error:** [Update after evaluation]

#### **3.2. Experiment 3: ADASYN**

- **Objective:** Generate synthetic samples focusing on difficult-to-learn instances.

- **Code**

  ```python
  from imblearn.over_sampling import ADASYN
  
  adasyn = ADASYN(sampling_strategy=0.5, random_state=42)
  X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)
  ```

- **Performance Metrics**

  - **Accuracy:** [Update after evaluation]
  - **Precision:** [Update after evaluation]
  - **Recall:** [Update after evaluation]
  - **F1 Score:** [Update after evaluation]
  - **AUC-ROC:** [Update after evaluation]
  - **Mean Absolute Error:** [Update after evaluation]
  - **Root Mean Squared Error:** [Update after evaluation]

#### **3.3. Experiment 4: Adjusting Class Weights in Random Forest**

- **Objective:** Incorporate class imbalance handling directly within the model.

- **Code**

  ```python
  from sklearn.ensemble import RandomForestClassifier
  
  rf = RandomForestClassifier(
      class_weight='balanced',
      n_estimators=300,
      max_depth=20,
      random_state=42,
      n_jobs=-1,
      max_features='sqrt',
      min_samples_split=5,
      min_samples_leaf=2
  )
  rf.fit(X_train_resampled, y_train_resampled)
  ```

- **Performance Metrics**

  - **Accuracy:** [Update after evaluation]
  - **Precision:** [Update after evaluation]
  - **Recall:** [Update after evaluation]
  - **F1 Score:** [Update after evaluation]
  - **AUC-ROC:** [Update after evaluation]
  - **Mean Absolute Error:** [Update after evaluation]
  - **Root Mean Squared Error:** [Update after evaluation]

---

### **4. Model Selection and Training Enhancements**

Explore different models and training strategies to identify the best-performing algorithm.

#### **4.1. Experiment 5: Gradient Boosting Machines (LightGBM)**

- **Objective:** Utilize LightGBM for potentially better performance and faster training.

- **Code**

  ```python
  import lightgbm as lgb
  
  lgb_model = lgb.LGBMClassifier(
      n_estimators=1000,
      learning_rate=0.05,
      num_leaves=31,
      max_depth=-1,
      objective='binary',
      class_weight='balanced',
      random_state=42,
      n_jobs=-1
  )
  lgb_model.fit(
      X_train_resampled, y_train_resampled,
      eval_set=[(X_val, y_val)],
      early_stopping_rounds=50,
      verbose=100
  )
  ```

- **Performance Metrics**

  - **Accuracy:** [Update after evaluation]
  - **Precision:** [Update after evaluation]
  - **Recall:** [Update after evaluation]
  - **F1 Score:** [Update after evaluation]
  - **AUC-ROC:** [Update after evaluation]
  - **Mean Absolute Error:** [Update after evaluation]
  - **Root Mean Squared Error:** [Update after evaluation]

#### **4.2. Experiment 6: CatBoost Classifier**

- **Objective:** Leverage CatBoost for handling categorical variables more effectively.

- **Code**

  ```python
  from catboost import CatBoostClassifier
  
  cat_model = CatBoostClassifier(
      iterations=1000,
      learning_rate=0.05,
      depth=6,
      eval_metric='AUC',
      random_seed=42,
      class_weights=[1, 10],  # Adjust based on class imbalance
      early_stopping_rounds=50,
      verbose=100
  )
  cat_model.fit(
      X_train_resampled, y_train_resampled,
      eval_set=(X_val, y_val)
  )
  ```

- **Performance Metrics**

  - **Accuracy:** [Update after evaluation]
  - **Precision:** [Update after evaluation]
  - **Recall:** [Update after evaluation]
  - **F1 Score:** [Update after evaluation]
  - **AUC-ROC:** [Update after evaluation]
  - **Mean Absolute Error:** [Update after evaluation]
  - **Root Mean Squared Error:** [Update after evaluation]

#### **4.3. Experiment 7: Support Vector Machines (SVM)**

- **Objective:** Explore SVMs for classification performance.

- **Code**

  ```python
  from sklearn.svm import SVC
  
  svm = SVC(
      probability=True,
      class_weight='balanced',
      kernel='rbf',
      C=1.0,
      gamma='scale',
      random_state=42,
      n_jobs=-1
  )
  svm.fit(X_train_resampled, y_train_resampled)
  ```

- **Performance Metrics**

  - **Accuracy:** [Update after evaluation]
  - **Precision:** [Update after evaluation]
  - **Recall:** [Update after evaluation]
  - **F1 Score:** [Update after evaluation]
  - **AUC-ROC:** [Update after evaluation]
  - **Mean Absolute Error:** [Update after evaluation]
  - **Root Mean Squared Error:** [Update after evaluation]

---

### **5. Hyperparameter Tuning**

Fine-tune model hyperparameters to optimize performance.

#### **5.1. Hyperparameter Tuning with GridSearchCV**

- **Objective:** Identify the optimal set of hyperparameters for the Random Forest model.

- **Code**

  ```python
  from sklearn.model_selection import GridSearchCV
  from sklearn.ensemble import RandomForestClassifier
  
  param_grid = {
      'n_estimators': [100, 200, 300],
      'max_depth': [10, 20, 30],
      'min_samples_split': [2, 5, 10],
      'min_samples_leaf': [1, 2, 4],
      'bootstrap': [True, False]
  }
  
  rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
  grid_search = GridSearchCV(
      estimator=rf,
      param_grid=param_grid,
      cv=3,
      n_jobs=-1,
      verbose=2,
      scoring='f1'
  )
  grid_search.fit(X_train_resampled, y_train_resampled)
  
  print("Best Parameters:", grid_search.best_params_)
  print("Best F1 Score:", grid_search.best_score_)
  
  best_rf = grid_search.best_estimator_
  ```

- **Performance Metrics**

  - **Accuracy:** [Update after evaluation]
  - **Precision:** [Update after evaluation]
  - **Recall:** [Update after evaluation]
  - **F1 Score:** [Update after evaluation]
  - **AUC-ROC:** [Update after evaluation]
  - **Mean Absolute Error:** [Update after evaluation]
  - **Root Mean Squared Error:** [Update after evaluation]

#### **5.2. Hyperparameter Tuning with Optuna**

- **Objective:** Utilize Optuna for efficient hyperparameter optimization.

- **Code**

  ```python
  import optuna
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import cross_val_score, StratifiedKFold
  
  def objective(trial):
      n_estimators = trial.suggest_int('n_estimators', 100, 1000)
      max_depth = trial.suggest_int('max_depth', 10, 50)
      min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
      min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
      bootstrap = trial.suggest_categorical('bootstrap', [True, False])
      
      clf = RandomForestClassifier(
          n_estimators=n_estimators,
          max_depth=max_depth,
          min_samples_split=min_samples_split,
          min_samples_leaf=min_samples_leaf,
          bootstrap=bootstrap,
          class_weight='balanced',
          random_state=42,
          n_jobs=-1
      )
      
      skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
      scores = cross_val_score(clf, X_train_resampled, y_train_resampled, cv=skf, scoring='f1')
      return scores.mean()
  
  study = optuna.create_study(direction='maximize')
  study.optimize(objective, n_trials=50)
  
  print("Best Parameters:", study.best_params)
  print("Best F1 Score:", study.best_value)
  
  best_rf_optuna = RandomForestClassifier(
      **study.best_params,
      class_weight='balanced',
      random_state=42,
      n_jobs=-1
  )
  best_rf_optuna.fit(X_train_resampled, y_train_resampled)
  ```

- **Performance Metrics**

  - **Accuracy:** [Update after evaluation]
  - **Precision:** [Update after evaluation]
  - **Recall:** [Update after evaluation]
  - **F1 Score:** [Update after evaluation]
  - **AUC-ROC:** [Update after evaluation]
  - **Mean Absolute Error:** [Update after evaluation]
  - **Root Mean Squared Error:** [Update after evaluation]

---

### **6. Ensembling Techniques**

Combine multiple models to leverage their strengths and mitigate individual weaknesses.

#### **6.1. Experiment 8: Voting Classifier**

- **Objective:** Combine different classifiers to improve overall performance.

- **Code**

  ```python
  from sklearn.ensemble import VotingClassifier
  from sklearn.linear_model import LogisticRegression
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.svm import SVC
  from xgboost import XGBClassifier
  
  # Initialize base models
  clf1 = RandomForestClassifier(n_estimators=300, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)
  clf2 = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42, n_jobs=-1)
  clf3 = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
  
  # Initialize Voting Classifier
  ensemble_clf = VotingClassifier(
      estimators=[
          ('rf', clf1),
          ('lr', clf2),
          ('xgb', clf3)
      ],
      voting='soft'
  )
  
  # Fit the ensemble model
  ensemble_clf.fit(X_train_resampled, y_train_resampled)
  ```

- **Performance Metrics**

  - **Accuracy:** [Update after evaluation]
  - **Precision:** [Update after evaluation]
  - **Recall:** [Update after evaluation]
  - **F1 Score:** [Update after evaluation]
  - **AUC-ROC:** [Update after evaluation]
  - **Mean Absolute Error:** [Update after evaluation]
  - **Root Mean Squared Error:** [Update after evaluation]

#### **6.2. Experiment 9: Stacking Classifier**

- **Objective:** Use a meta-classifier to combine predictions from base models.

- **Code**

  ```python
  from sklearn.ensemble import StackingClassifier
  from sklearn.linear_model import LogisticRegression
  from sklearn.ensemble import RandomForestClassifier
  from xgboost import XGBClassifier
  
  # Define base learners
  estimators = [
      ('rf', RandomForestClassifier(n_estimators=300, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)),
      ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
  ]
  
  # Define meta-learner
  meta_learner = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
  
  # Initialize Stacking Classifier
  stacking_clf = StackingClassifier(
      estimators=estimators,
      final_estimator=meta_learner,
      cv=5,
      n_jobs=-1
  )
  
  # Fit the stacking model
  stacking_clf.fit(X_train_resampled, y_train_resampled)
  ```

- **Performance Metrics**

  - **Accuracy:** [Update after evaluation]
  - **Precision:** [Update after evaluation]
  - **Recall:** [Update after evaluation]
  - **F1 Score:** [Update after evaluation]
  - **AUC-ROC:** [Update after evaluation]
  - **Mean Absolute Error:** [Update after evaluation]
  - **Root Mean Squared Error:** [Update after evaluation]

#### **6.3. Experiment 10: Blending Different Models**

- **Objective:** Blend predictions from multiple models using weighted averages.

- **Code**

  ```python
  import numpy as np
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.linear_model import LogisticRegression
  from xgboost import XGBClassifier
  
  # Initialize base models
  model1 = RandomForestClassifier(n_estimators=300, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)
  model2 = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42, n_jobs=-1)
  model3 = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
  
  # Fit base models
  model1.fit(X_train_resampled, y_train_resampled)
  model2.fit(X_train_resampled, y_train_resampled)
  model3.fit(X_train_resampled, y_train_resampled)
  
  # Get predicted probabilities
  prob1 = model1.predict_proba(X_val)[:, 1]
  prob2 = model2.predict_proba(X_val)[:, 1]
  prob3 = model3.predict_proba(X_val)[:, 1]
  
  # Weighted average of probabilities
  blended_prob = (prob1 * 0.4) + (prob2 * 0.3) + (prob3 * 0.3)
  
  # Convert to binary predictions
  blended_pred = (blended_prob >= 0.5).astype(int)
  ```

- **Performance Metrics**

  - **Accuracy:** [Update after evaluation]
  - **Precision:** [Update after evaluation]
  - **Recall:** [Update after evaluation]
  - **F1 Score:** [Update after evaluation]
  - **AUC-ROC:** [Update after evaluation]
  - **Mean Absolute Error:** [Update after evaluation]
  - **Root Mean Squared Error:** [Update after evaluation]

---

### **7. Hyperparameter Tuning for Advanced Models**

Optimize hyperparameters for models like LightGBM and CatBoost.

#### **7.1. Hyperparameter Tuning for LightGBM with Optuna**

- **Objective:** Optimize LightGBM hyperparameters for better performance.

- **Code**

  ```python
  import optuna
  import lightgbm as lgb
  from sklearn.model_selection import cross_val_score, StratifiedKFold
  
  def objective_lgb(trial):
      param = {
          'objective': 'binary',
          'metric': 'auc',
          'boosting_type': 'gbdt',
          'num_leaves': trial.suggest_int('num_leaves', 20, 50),
          'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
          'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
          'max_depth': trial.suggest_int('max_depth', 5, 50),
          'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
          'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
          'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
          'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
          'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
          'class_weight': 'balanced'
      }
      
      clf = lgb.LGBMClassifier(**param, random_state=42, n_jobs=-1)
      
      skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
      auc = cross_val_score(clf, X_train_resampled, y_train_resampled, cv=skf, scoring='roc_auc').mean()
      return auc
  
  study_lgb = optuna.create_study(direction='maximize')
  study_lgb.optimize(objective_lgb, n_trials=50)
  
  print("Best Parameters for LightGBM:", study_lgb.best_params)
  print("Best AUC-ROC Score:", study_lgb.best_value)
  
  best_lgb = lgb.LGBMClassifier(**study_lgb.best_params, random_state=42, n_jobs=-1)
  best_lgb.fit(X_train_resampled, y_train_resampled)
  ```

- **Performance Metrics**

  - **Accuracy:** [Update after evaluation]
  - **Precision:** [Update after evaluation]
  - **Recall:** [Update after evaluation]
  - **F1 Score:** [Update after evaluation]
  - **AUC-ROC:** [Update after evaluation]
  - **Mean Absolute Error:** [Update after evaluation]
  - **Root Mean Squared Error:** [Update after evaluation]

#### **7.2. Hyperparameter Tuning for CatBoost with GridSearchCV**

- **Objective:** Optimize CatBoost hyperparameters for enhanced performance.

- **Code**

  ```python
  from sklearn.model_selection import GridSearchCV
  from catboost import CatBoostClassifier
  
  param_grid_cat = {
      'depth': [6, 8, 10],
      'learning_rate': [0.01, 0.05, 0.1],
      'iterations': [500, 1000, 1500],
      'l2_leaf_reg': [1, 3, 5],
      'border_count': [32, 64, 128]
  }
  
  cat = CatBoostClassifier(
      eval_metric='AUC',
      random_seed=42,
      class_weights=[1, 10],
      verbose=0
  )
  
  grid_search_cat = GridSearchCV(
      estimator=cat,
      param_grid=param_grid_cat,
      cv=3,
      scoring='roc_auc',
      n_jobs=-1,
      verbose=2
  )
  
  grid_search_cat.fit(X_train_resampled, y_train_resampled)
  
  print("Best Parameters for CatBoost:", grid_search_cat.best_params_)
  print("Best AUC-ROC Score:", grid_search_cat.best_score_)
  
  best_cat = grid_search_cat.best_estimator_
  ```

- **Performance Metrics**

  - **Accuracy:** [Update after evaluation]
  - **Precision:** [Update after evaluation]
  - **Recall:** [Update after evaluation]
  - **F1 Score:** [Update after evaluation]
  - **AUC-ROC:** [Update after evaluation]
  - **Mean Absolute Error:** [Update after evaluation]
  - **Root Mean Squared Error:** [Update after evaluation]

---

### **8. Evaluation Enhancements**

Ensure that your evaluation strategy aligns with the problem's requirements and leverages appropriate metrics.

#### **8.1. Focus on Precision-Recall AUC**

- **Objective:** Utilize Precision-Recall AUC for a more informative evaluation on imbalanced datasets.

- **Code**

  ```python
  from sklearn.metrics import precision_recall_curve, auc
  
  y_pred_proba = best_rf.predict_proba(X_val)[:, 1]
  precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
  pr_auc = auc(recall, precision)
  print(f'Precision-Recall AUC: {pr_auc:.4f}')
  ```

- **Performance Metrics**

  - **Precision-Recall AUC:** [Update after evaluation]

#### **8.2. Implement Stratified Cross-Validation**

- **Objective:** Ensure each fold maintains the original class distribution to get a reliable estimate of model performance.

- **Code**

  ```python
  from sklearn.model_selection import StratifiedKFold, cross_val_score
  
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  cv_scores = cross_val_score(best_rf, X_train_resampled, y_train_resampled, cv=skf, scoring='f1')
  print(f'Cross-Validation F1 Scores: {cv_scores}')
  print(f'Mean CV F1 Score: {cv_scores.mean():.4f}')
  ```

- **Performance Metrics**

  - **Cross-Validation F1 Scores:** [Update after evaluation]
  - **Mean CV F1 Score:** [Update after evaluation]

#### **8.3. Threshold Optimization**

- **Objective:** Adjust the decision threshold to balance precision and recall effectively.

- **Code**

  ```python
  from sklearn.metrics import precision_recall_curve
  
  y_scores = best_rf.predict_proba(X_val)[:, 1]
  precision, recall, thresholds = precision_recall_curve(y_val, y_scores)
  f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
  best_threshold = thresholds[np.argmax(f1_scores)]
  print(f'Best Threshold: {best_threshold:.4f}')
  
  y_pred_opt = (y_scores >= best_threshold).astype(int)
  
  # Evaluate with optimized threshold
  from sklearn.metrics import classification_report, roc_auc_score
  
  print(classification_report(y_val, y_pred_opt))
  print(f'AUC-ROC: {roc_auc_score(y_val, y_scores):.4f}')
  ```

- **Performance Metrics**

  - **Optimized Precision:** [Update after evaluation]
  - **Optimized Recall:** [Update after evaluation]
  - **Optimized F1 Score:** [Update after evaluation]
  - **AUC-ROC:** [Update after evaluation]

#### **8.4. Analyze Confusion Matrix**

- **Objective:** Gain insights into the types of errors your model is making.

- **Code**

  ```python
  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
  import matplotlib.pyplot as plt
  
  cm = confusion_matrix(y_val, y_pred_opt)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm)
  disp.plot(cmap=plt.cm.Blues)
  plt.title('Confusion Matrix with Optimized Threshold')
  plt.show()
  ```

- **Performance Metrics**

  - **Confusion Matrix Analysis:** [Interpret results after evaluation]

---

### **9. Advanced Modeling Techniques**

Incorporate deep learning and transformer-based models to capture complex patterns.

#### **9.1. Recurrent Neural Networks (RNNs) with LSTM**

- **Objective:** Capture temporal dependencies in patient data.

- **Code**

  ```python
  import numpy as np
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import LSTM, Dense, Dropout
  from tensorflow.keras.callbacks import EarlyStopping
  
  # Assuming data is reshaped into sequences (patients x timesteps x features)
  # Placeholder for actual sequence data preparation
  X_train_seq = ...  # Replace with your sequence data
  y_train_seq = ...  # Replace with your sequence labels
  X_val_seq = ...    # Replace with your sequence data
  y_val_seq = ...    # Replace with your sequence labels
  
  model = Sequential()
  model.add(LSTM(64, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
  model.add(LSTM(32))
  model.add(Dense(1, activation='sigmoid'))
  
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
  
  early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
  
  model.fit(
      X_train_seq, y_train_seq,
      epochs=100,
      batch_size=64,
      validation_data=(X_val_seq, y_val_seq),
      callbacks=[early_stop],
      verbose=1
  )
  ```

- **Performance Metrics**

  - **Accuracy:** [Update after evaluation]
  - **Precision:** [Update after evaluation]
  - **Recall:** [Update after evaluation]
  - **F1 Score:** [Update after evaluation]
  - **AUC-ROC:** [Update after evaluation]

#### **9.2. Transformer Models**

- **Objective:** Utilize transformer architectures to capture complex temporal relationships.

- **Note:** Implementing transformer models requires substantial computational resources and expertise. Consider starting with pre-trained models or simpler architectures before moving to transformers.

- **Code Skeleton**

  ```python
  from transformers import TimeSeriesTransformerModel, TimeSeriesTransformerConfig
  from tensorflow.keras.models import Model
  from tensorflow.keras.layers import Input, Dense
  
  # Define transformer configuration
  config = TimeSeriesTransformerConfig(
      input_dim=X_train_seq.shape[2],
      num_heads=4,
      num_layers=2,
      d_model=64,
      dropout=0.1
  )
  
  # Initialize transformer model
  transformer = TimeSeriesTransformerModel(config)
  
  # Define custom model
  inputs = Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
  x = transformer(inputs)
  outputs = Dense(1, activation='sigmoid')(x)
  
  model = Model(inputs, outputs)
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
  
  # Train the model
  model.fit(
      X_train_seq, y_train_seq,
      epochs=100,
      batch_size=64,
      validation_data=(X_val_seq, y_val_seq),
      callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
      verbose=1
  )
  ```

- **Performance Metrics**

  - **Accuracy:** [Update after evaluation]
  - **Precision:** [Update after evaluation]
  - **Recall:** [Update after evaluation]
  - **F1 Score:** [Update after evaluation]
  - **AUC-ROC:** [Update after evaluation]

---

### **10. Pipeline Optimization and Automation**

Ensure that your machine learning pipeline is efficient, reproducible, and free from data leakage.

#### **10.1. Pipeline Integration with Scikit-Learn**

- **Objective:** Encapsulate preprocessing and modeling steps within a Scikit-Learn pipeline to prevent data leakage.

- **Code**

  ```python
  from sklearn.pipeline import Pipeline
  from sklearn.impute import IterativeImputer
  from sklearn.preprocessing import StandardScaler
  from sklearn.ensemble import RandomForestClassifier
  
  pipeline = Pipeline([
      ('imputer', IterativeImputer(random_state=42)),
      ('scaler', RobustScaler()),
      ('classifier', RandomForestClassifier(
          n_estimators=300,
          max_depth=20,
          class_weight='balanced',
          random_state=42,
          n_jobs=-1,
          max_features='sqrt',
          min_samples_split=5,
          min_samples_leaf=2
      ))
  ])
  
  pipeline.fit(X_train, y_train)
  y_pred = pipeline.predict(X_val)
  ```

- **Performance Metrics**

  - **Accuracy:** [Update after evaluation]
  - **Precision:** [Update after evaluation]
  - **Recall:** [Update after evaluation]
  - **F1 Score:** [Update after evaluation]
  - **AUC-ROC:** [Update after evaluation]
  - **Mean Absolute Error:** [Update after evaluation]
  - **Root Mean Squared Error:** [Update after evaluation]

#### **10.2. Automated Hyperparameter Tuning with Optuna**

- **Objective:** Automate and streamline hyperparameter tuning processes.

- **Code**

  ```python
  import optuna
  from sklearn.model_selection import cross_val_score, StratifiedKFold
  
  def objective_pipeline(trial):
      # Define hyperparameters to tune
      n_estimators = trial.suggest_int('classifier__n_estimators', 100, 500)
      max_depth = trial.suggest_int('classifier__max_depth', 10, 50)
      min_samples_split = trial.suggest_int('classifier__min_samples_split', 2, 10)
      min_samples_leaf = trial.suggest_int('classifier__min_samples_leaf', 1, 4)
      max_features = trial.suggest_categorical('classifier__max_features', ['sqrt', 'log2'])
      
      pipeline.set_params(
          classifier__n_estimators=n_estimators,
          classifier__max_depth=max_depth,
          classifier__min_samples_split=min_samples_split,
          classifier__min_samples_leaf=min_samples_leaf,
          classifier__max_features=max_features
      )
      
      skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
      scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='f1')
      return scores.mean()
  
  study_pipeline = optuna.create_study(direction='maximize')
  study_pipeline.optimize(objective_pipeline, n_trials=50)
  
  print("Best Parameters for Pipeline:", study_pipeline.best_params)
  print("Best F1 Score:", study_pipeline.best_value)
  
  # Update pipeline with best parameters
  pipeline.set_params(**study_pipeline.best_params)
  pipeline.fit(X_train, y_train)
  y_pred_opt = pipeline.predict(X_val)
  ```

- **Performance Metrics**

  - **Accuracy:** [Update after evaluation]
  - **Precision:** [Update after evaluation]
  - **Recall:** [Update after evaluation]
  - **F1 Score:** [Update after evaluation]
  - **AUC-ROC:** [Update after evaluation]
  - **Mean Absolute Error:** [Update after evaluation]
  - **Root Mean Squared Error:** [Update after evaluation]

#### **10.3. Code Modularization**

- **Objective:** Enhance code readability, maintainability, and scalability by organizing it into modules and functions.

- **Action Steps:**
  
  1. **Create Separate Modules:**
     - **Data Preprocessing Module:** Encapsulate all preprocessing steps.
     - **Feature Engineering Module:** Handle all feature creation and transformation.
     - **Model Training Module:** Manage model initialization, training, and prediction.
     - **Evaluation Module:** Centralize evaluation metrics and visualization.
  
  2. **Implement Reusable Functions:**
     - Define functions for repetitive tasks like scaling, encoding, and imputing.
  
  3. **Example Structure:**

     ```
     project/
     ├── data/
     ├── src/
     │   ├── __init__.py
     │   ├── preprocessing.py
     │   ├── feature_engineering.py
     │   ├── models.py
     │   ├── evaluation.py
     │   └── utils.py
     ├── notebooks/
     └── main.py
     ```

- **Benefits:**
  
  - **Reusability:** Easily reuse code across different experiments.
  - **Maintainability:** Simplify debugging and updates.
  - **Scalability:** Facilitate collaboration and expansion of the project.

---

### **11. Additional Considerations**

Incorporate domain knowledge and other advanced techniques to further enhance model performance.

#### **11.1. Encode Missingness as Features**

- **Objective:** Utilize the pattern of missing data as additional features that may carry predictive information.

- **Code**

  ```python
  # Example for Heart Rate (HR)
  X_train['HR_missing'] = X_train['HR'].isnull().astype(int)
  X_val['HR_missing'] = X_val['HR'].isnull().astype(int)
  X_test['HR_missing'] = X_test['HR'].isnull().astype(int)
  
  # Impute missing values after encoding
  X_train['HR'] = X_train['HR'].fillna(X_train['HR'].median())
  X_val['HR'] = X_val['HR'].fillna(X_val['HR'].median())
  X_test['HR'] = X_test['HR'].fillna(X_test['HR'].median())
  ```

- **Performance Metrics**

  - **Accuracy:** [Update after evaluation]
  - **Precision:** [Update after evaluation]
  - **Recall:** [Update after evaluation]
  - **F1 Score:** [Update after evaluation]
  - **AUC-ROC:** [Update after evaluation]
  - **Mean Absolute Error:** [Update after evaluation]
  - **Root Mean Squared Error:** [Update after evaluation]

#### **11.2. Evaluate Feature Importance**

- **Objective:** Identify and retain the most impactful features to enhance model performance and reduce complexity.

- **Code**

  ```python
  import matplotlib.pyplot as plt
  import pandas as pd
  
  # For Random Forest
  feature_importances = best_rf.feature_importances_
  features = X_train.columns
  importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
  importance_df = importance_df.sort_values(by='Importance', ascending=False)
  
  # Plot Feature Importances
  plt.figure(figsize=(12, 8))
  sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
  plt.title('Top 20 Feature Importances')
  plt.tight_layout()
  plt.show()
  ```

- **Action Steps:**
  
  1. **Identify Low-Importance Features:** Drop features with negligible importance.
  
  2. **Re-train Models:** After feature reduction, retrain your models and evaluate performance.

- **Performance Metrics**

  - **Accuracy:** [Update after evaluation]
  - **Precision:** [Update after evaluation]
  - **Recall:** [Update after evaluation]
  - **F1 Score:** [Update after evaluation]
  - **AUC-ROC:** [Update after evaluation]
  - **Mean Absolute Error:** [Update after evaluation]
  - **Root Mean Squared Error:** [Update after evaluation]

#### **11.3. Regularization Techniques**

- **Objective:** Prevent overfitting and improve model generalization.

- **Code Examples:**
  
  - **For Logistic Regression:**
  
    ```python
    from sklearn.linear_model import LogisticRegression
    
    lr = LogisticRegression(
        penalty='l2',
        C=0.5,  # Inverse of regularization strength
        max_iter=2000,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    lr.fit(X_train_resampled, y_train_resampled)
    ```
  
  - **For Neural Networks:**
  
    ```python
    from tensorflow.keras.layers import Dropout
    
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(Dropout(0.5))
    model.add(LSTM(32))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    ```

- **Performance Metrics**

  - **Accuracy:** [Update after evaluation]
  - **Precision:** [Update after evaluation]
  - **Recall:** [Update after evaluation]
  - **F1 Score:** [Update after evaluation]
  - **AUC-ROC:** [Update after evaluation]
  - **Mean Absolute Error:** [Update after evaluation]
  - **Root Mean Squared Error:** [Update after evaluation]

#### **11.4. Data Augmentation with Synthetic Data Generation**

- **Objective:** Enrich the dataset with clinically plausible synthetic samples to improve model robustness.

- **Code (Using GANs for Synthetic Data Generation):**

  ```python
  # Example using a simple GAN framework for tabular data
  # Note: Implementing GANs for tabular data is complex and may require specialized libraries like CTGAN or TVAE
  
  from ctgan import CTGANSynthesizer
  
  # Initialize CTGAN
  ctgan = CTGANSynthesizer(epochs=100)
  
  # Fit CTGAN on minority class
  ctgan.fit(X_train[y_train == 1], discrete_columns=['Gender', 'Unit1', 'Unit2'])
  
  # Generate synthetic samples
  synthetic_samples = ctgan.sample(10000)
  
  # Combine with original training data
  X_train_augmented = pd.concat([X_train_resampled, synthetic_samples])
  y_train_augmented = pd.concat([y_train_resampled, pd.Series([1]*10000)])
  ```

- **Performance Metrics**

  - **Accuracy:** [Update after evaluation]
  - **Precision:** [Update after evaluation]
  - **Recall:** [Update after evaluation]
  - **F1 Score:** [Update after evaluation]
  - **AUC-ROC:** [Update after evaluation]
  - **Mean Absolute Error:** [Update after evaluation]
  - **Root Mean Squared Error:** [Update after evaluation]

---

### **12. Final Model Selection and Deployment**

After conducting the above experiments, select the best-performing model for deployment.

#### **12.1. Selecting the Best Model**

- **Action Steps:**
  
  1. **Compare Performance Metrics:** Review all metrics across experiments to identify the top-performing model(s).
  
  2. **Consider Computational Efficiency:** Balance model performance with training and inference time, especially for real-time predictions.
  
  3. **Assess Model Interpretability:** Choose models that offer insights into feature importance and decision-making, crucial for clinical applications.

- **Final Model Metrics**

  - **Accuracy:** [Final Best Model]
  - **Precision:** [Final Best Model]
  - **Recall:** [Final Best Model]
  - **F1 Score:** [Final Best Model]
  - **AUC-ROC:** [Final Best Model]
  - **Mean Absolute Error:** [Final Best Model]
  - **Root Mean Squared Error:** [Final Best Model]

#### **12.2. Final Evaluation on Test Set**

- **Objective:** Validate the selected model's performance on unseen data.

- **Code**

  ```python
  # Assuming best_model is your selected model
  y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
  y_test_pred = (y_test_pred_proba >= best_threshold).astype(int)  # Use optimized threshold
  
  from sklearn.metrics import classification_report, roc_auc_score
  
  print(classification_report(y_test, y_test_pred))
  print(f'AUC-ROC on Test Set: {roc_auc_score(y_test, y_test_pred_proba):.4f}')
  
  # Plot Confusion Matrix
  cm = confusion_matrix(y_test, y_test_pred)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm)
  disp.plot(cmap=plt.cm.Blues)
  plt.title('Confusion Matrix on Test Set')
  plt.show()
  ```

- **Performance Metrics**

  - **Accuracy:** [Final Evaluation]
  - **Precision:** [Final Evaluation]
  - **Recall:** [Final Evaluation]
  - **F1 Score:** [Final Evaluation]
  - **AUC-ROC:** [Final Evaluation]
  - **Mean Absolute Error:** [Final Evaluation]
  - **Root Mean Squared Error:** [Final Evaluation]

#### **12.3. Save and Deploy the Best Model**

- **Objective:** Persist the best model for future predictions and potential deployment.

- **Code**

  ```python
  import joblib
  
  # Save the model
  model_dir = "models"
  os.makedirs(model_dir, exist_ok=True)
  model_path = os.path.join(model_dir, f"best_model_{best_model_name.lower().replace(' ', '_')}.pkl")
  joblib.dump(best_model, model_path)
  print(f"Best model saved to {model_path}")
  
  # To load the model later
  # best_model_loaded = joblib.load(model_path)
  ```

---

### **13. Documentation and Reporting**

Maintain thorough documentation to track experiments and findings.

#### **13.1. Maintain an Experiment Log**

- **Objective:** Keep a detailed record of all experiments, configurations, and results.

- **Action Steps:**
  
  1. **Use a Spreadsheet or Notebook:** Document each experiment's configuration, changes, and resulting metrics.
  
  2. **Automate Logging:** Incorporate logging within your code to automatically record experiment details.

- **Example:**

  ```python
  import logging
  
  # Setup logger
  logger = logging.getLogger('ExperimentLogger')
  logger.setLevel(logging.INFO)
  fh = logging.FileHandler('experiments.log')
  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
  fh.setFormatter(formatter)
  logger.addHandler(fh)
  
  # Log experiment details
  logger.info("Experiment 1: Iterative Imputer with Robust Scaler")
  logger.info(f"Performance Metrics: Accuracy={accuracy}, Precision={precision}, ...")
  ```

#### **13.2. Generate Comprehensive Reports**

- **Objective:** Summarize findings, model performances, and insights.

- **Action Steps:**
  
  1. **Use Jupyter Notebooks:** Create detailed notebooks that include code, explanations, and visualizations.
  
  2. **Create Dashboards:** Utilize tools like **Tableau** or **Power BI** for interactive visualizations.
  
  3. **Automate Report Generation:** Use libraries like **Matplotlib** and **Seaborn** to generate plots automatically and save them to reports.

- **Example Code for Automated Reports:**

  ```python
  import json
  
  # Save metrics to JSON
  metrics = {
      'accuracy': accuracy,
      'precision': precision,
      'recall': recall,
      'f1_score': f1,
      'auc_roc': auc_roc,
      'mae': mae,
      'rmse': rmse
  }
  
  with open('reports/experiment1_metrics.json', 'w') as f:
      json.dump(metrics, f, indent=4)
  
  # Save confusion matrix plot
  cm = confusion_matrix(y_val, y_pred_opt)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm)
  disp.plot(cmap=plt.cm.Blues)
  plt.title('Confusion Matrix - Experiment 1')
  plt.savefig('reports/experiment1_confusion_matrix.png')
  plt.close()
  ```

---

## **Summary of Recommendations**

1. **Incrementally Enhance Preprocessing:**
   - Implement advanced imputation and encoding techniques.
   - Experiment with different scaling methods.

2. **Expand Feature Engineering:**
   - Incorporate temporal and interaction features.
   - Apply dimensionality reduction where appropriate.

3. **Refine Class Imbalance Strategies:**
   - Use hybrid sampling methods and adjust class weights.
   - Monitor for overfitting and computational efficiency.

4. **Explore Diverse Models:**
   - Test various classifiers, including ensemble and boosting methods.
   - Implement deep learning models if feasible.

5. **Optimize Hyperparameters:**
   - Utilize GridSearchCV and Optuna for systematic tuning.
   - Focus on models that show promise in preliminary experiments.

6. **Leverage Ensembling Techniques:**
   - Combine multiple models using voting, stacking, or blending.
   - Assess the impact on performance metrics.

7. **Enhance Evaluation Strategies:**
   - Prioritize relevant metrics like Precision-Recall AUC.
   - Implement cross-validation and threshold optimization.

8. **Modularize and Automate the Pipeline:**
   - Ensure reproducibility and maintainability.
   - Prevent data leakage through proper pipeline structuring.

9. **Document and Report Findings:**
   - Maintain a detailed experiment log.
   - Generate comprehensive reports for insights and future reference.

10. **Consider Advanced Modeling Techniques:**
    - Explore RNNs and transformer models for temporal data.
    - Balance complexity with performance gains.

---

## **Next Steps**

1. **Implement Each Experiment Sequentially:**
   - Start with preprocessing enhancements, followed by feature engineering.
   - Progress to class imbalance refinements and model explorations.

2. **Evaluate and Compare Performance Metrics:**
   - After each experiment, record and analyze performance to identify improvements.

3. **Iterate Based on Findings:**
   - Focus on experiments that yield significant performance gains.
   - Revisit and adjust previous steps as necessary.

4. **Collaborate with Domain Experts:**
   - Incorporate clinical insights to ensure features and models align with real-world scenarios.

5. **Prepare for Deployment:**
   - Once satisfied with the model's performance, focus on deployment strategies, monitoring, and maintenance.
