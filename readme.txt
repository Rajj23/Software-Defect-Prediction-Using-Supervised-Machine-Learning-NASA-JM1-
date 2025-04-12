# Software Defect Prediction using Supervised Machine Learning

This project applies supervised machine learning techniques to predict software defects using the JM1 dataset. 
The focus is on robust preprocessing, handling class imbalance, and comparing multiple classification algorithms to find the most effective model.

---

## Dataset

- Name: JM1 Software Defect Dataset
- Source: NASA Metrics Data Program (MDP)
- Target Variable: defects (binary classification - 1 for defect, 0 for no defect)

---

## Workflow Overview

1. Data Cleaning
   - Remove duplicates
   - Handle missing values using SimpleImputer

2. Feature Engineering
   - Log transformation on skewed features (loc, v, e)
   - Feature scaling with MinMaxScaler

3. Handling Class Imbalance
   - Applied SMOTE (Synthetic Minority Oversampling Technique)

4. Feature Selection
   - Recursive Feature Elimination (RFE)

5. Model Training & Tuning
   - Trained multiple models:
     - Logistic Regression
     - Random Forest
     - Gradient Boosting
     - Support Vector Classifier
     - XGBoost Classifier
   - Hyperparameter tuning via GridSearchCV

6. Evaluation
   - Metrics: Accuracy, Precision, Recall, F1 Score, ROC-AUC
   - Confusion Matrix Visualization

---

## Model Performance

Each model was evaluated using multiple metrics. XGBoost and Random Forest performed best on key metrics including ROC-AUC and F1-score.

## Libraries Used

- pandas, numpy
- scikit-learn
- xgboost
- imblearn
- matplotlib, seaborn

---