# Customer-Transaction-Prediction
Customer Transaction Prediction using Machine Learning with automated hyperparameter tuning (HistGradientBoost, RandomizedSearchCV) and ROC-AUC based evaluation.

## 📌 Project Overview
This project focuses on predicting whether a customer will make a specific transaction in the future, irrespective of the transaction amount.
The objective is to build a robust machine learning model that helps financial institutions identify potential customers likely to perform transactions, enabling better targeting strategies and improved operational efficiency.

## Business Problem
Banks and financial institutions need to proactively identify customers who are likely to make transactions. 

Accurate prediction helps:
- Improve customer engagement strategies
- Optimize marketing campaigns
- Reduce operational costs
- Enhance revenue forecasting


## Dataset Description

- 200 anonymized numerical features
- one `ID_code` column (identifier)
- `target` column (binary classification)
- - `0` → Customer will NOT make a transaction
- `1` → Customer WILL make a transaction
- 200000 rows

Since features are anonymized,EDA is skipped and feature-level interpretation is limited, and the focus is placed on predictive performance and model optimization.

## Project Workflow

### Data Preprocessing
- Removed identifier column
- Train-test split
- Feature scaling using StandardScaler
- Class imbalance handling using class weighting

### Model Building
Multiple models were implemented and compared:

- Logistic Regression (Baseline)
- Decision Tree
- Random Forest
- Gradient Boosting
- HistGradientBoosting
- XGBoost
- additional model are build for comparison purpose(KNN,Naive bayes)

### Automated Hyperparameter and threshold tuning
To improve model performance and ensure systematic optimization:
- RandomizedSearchCV was applied
  
- Cross-validation used for reliable evaluation
- ROC-AUC used as scoring metric

### Model Evaluation
Primary evaluation metric:
- **ROC-AUC Score**

Additional metrics:
- Confusion Matrix
- Precision
- Recall
- F1-score

ROC-AUC was prioritized due to class imbalance.

## Feature Imporatnce
- used permutation Feature importance
- For identifing most influencing features

## Handling Class Imbalance
Since the dataset is imbalanced:
- Class weights were used where appropriate
- Threshold tuning was explored for precision-recall tradeoff

##  Final Model Selection
The best performing model was selected based on:

- Cross-validated ROC-AUC
- Generalization performance
- Computational efficiency
- HistGradientBoost is the final selected model
