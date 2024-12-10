**Model Card: Tuned Random Forest Classifier**

### **Model Overview**

- **Model Name:** Tuned Random Forest Classifier
- **Model Type:** Random Forest (Ensemble Learning)
- **Framework Used:** Scikit-learn
- **Purpose:** Binary classification to predict target classes (Class 0 and Class 1).

### **Model Objective**

The primary objective of this model is to accurately classify data into two categories, focusing on maintaining high precision and recall for the minority class (Class 1). This is particularly important for applications where false negatives could be costly or critical.

---

### **Performance Summary**

#### **Key Metrics:**

| Metric                      | Value  |
| --------------------------- | ------ |
| **Accuracy**                | 98.79% |
| **Precision (Class 1)**     | 67.88% |
| **Recall (Class 1)**        | 47.54% |
| **F1 Score (Class 1)**      | 55.91% |
| **AUC-ROC**                 | 0.98   |
| **Mean Absolute Error**     | 0.0121 |
| **Root Mean Squared Error** | 0.11   |

#### **Class-Level Performance:**

- **Class 0 (Negative):** Excellent performance with 100% recall and minimal false positives.
- **Class 1 (Positive):** Moderate performance, with a recall of 47.54% indicating that the model misses over half of actual positive cases.

---

### **Evaluation Metrics Details**

1. **Confusion Matrix (Normalized):**

   - **Class 0 (Negative):** 100% correctly classified (True Negatives).
   - **Class 1 (Positive):** 48% correctly classified (True Positives), with significant false negatives (52%).

2. **Precision-Recall Curve:**

   - **AP (Average Precision):** 0.63, reflecting moderate balance between precision and recall for the positive class.
   - **Trend:** Precision decreases as recall increases, indicating difficulty in consistently identifying all positive cases.

3. **ROC Curve and AUC:**

   - **AUC-ROC = 0.98:** Excellent ability to separate positive and negative classes, although the imbalance between classes impacts recall for the minority class.

---

### **Model Details**

- **Features:** The dataset comprises X features (details not provided but assumed numerical and categorical).
- **Hyperparameters:**
  - Number of Trees (n\_estimators): Tuned
  - Maximum Depth (max\_depth): Tuned
  - Class Weight: None (imbalance handled via post-evaluation strategies)

### **Strengths**

- **Class 0:** Exceptional accuracy and minimal false positives.
- **Overall:** High AUC-ROC (0.98), indicating strong separability between the two classes.

### **Weaknesses**

- **Class 1:** Low recall for the minority class (47.54%), meaning a significant number of positive cases are misclassified.
- **Imbalanced Dataset:** Class imbalance leads to biased predictions toward the majority class (Class 0).

---

### **Next Steps for Improvement**

1. **Handle Class Imbalance:**

   - Use oversampling techniques like SMOTE or undersampling to balance the dataset.
   - Apply class weighting in the random forest (`class_weight='balanced'`).

2. **Threshold Tuning:**

   - Adjust the decision threshold to improve recall for Class 1 at the cost of precision.

3. **Hyperparameter Optimization:**

   - Fine-tune parameters like `min_samples_split`, `min_samples_leaf`, and `max_features`.

4. **Model Alternatives:**

   - Explore other ensemble methods (e.g., Gradient Boosting, XGBoost) for better minority-class performance.

5. **Use Custom Metrics:**

   - Optimize for F2 Score (weighs recall more heavily) or other metrics that prioritize minority-class performance.

---

#### **Model Parameters: Before Tuning**

The hyperparameters used for the random forest before tuning were:

- Max Depth: 30
- Max Features: Logarithmically scaled features (log2)
- in Samples Leaf: 1 (allows for highly granular splits, increasing overfitting risks)
- Min Samples Split: 2
- Number of Estimators: 484 (reasonable but possibly suboptimal for convergence)

### **Conclusion**

The tuned random forest model demonstrates strong overall performance, particularly for the majority class (Class 0). However, its ability to identify the minority class (Class 1) needs significant improvement. By addressing class imbalance and tuning thresholds, the model can achieve a better balance between precision and recall, making it more suitable for real-world applications where both metrics are critical.
