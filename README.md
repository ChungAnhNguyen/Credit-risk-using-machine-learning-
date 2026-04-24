# CreditGuard ML — Credit Card Default Prediction

---

## Overview

**CreditGuard ML** is a supervised machine learning project designed to predict the probability of credit card payment default using client demographic data and historical transaction records. Early and accurate default detection enables financial institutions to take preventive measures — such as payment deferrals or debt consolidation — thereby reducing credit losses.

The dataset used is the **Default of Credit Card Clients** (UCI Machine Learning Repository), comprising 30,000 observations of Taiwanese credit card holders collected between April and September 2005.

---

## Project Structure

```
CreditGuard-ML/
├── data/
│   └── credit_card_default.csv        # UCI dataset
├── src/
│   ├── 01_eda.R                       # Exploratory data analysis
│   ├── 02_preprocessing.R             # Data cleaning & preprocessing
│   ├── 03_decision_tree.R             # Decision Tree model
│   └── 04_random_forest.R             # Random Forest model
├── outputs/
│   ├── figures/                       # Generated plots
│   └── results/                       # Metrics & confusion matrices
└── README.md
```

---

## Dataset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

**Target variable:** `DEFAULT` — whether a client defaults next month (0 = No, 1 = Yes)

**23 predictor variables:**

| Variable | Description |
|---|---|
| `LIMIT_BAL` | Total credit limit granted (TWD) |
| `SEX` | 1 = Male, 2 = Female |
| `EDUCATION` | 1 = Graduate school, 2 = University, 3 = High school, 4 = Other |
| `MARRIAGE` | 1 = Married, 2 = Single, 3 = Other |
| `AGE` | Age in years |
| `PAY_1` – `PAY_6` | Repayment status (April–September 2005) |
| `BILL_AMT1` – `BILL_AMT6` | Monthly bill statement amounts (TWD) |
| `PAY_AMT1` – `PAY_AMT6` | Previous payment amounts (TWD) |

**Class distribution:** strongly imbalanced — 77.88% non-default (class 0) vs 22.12% default (class 1).

---

## Preprocessing

- Correction of out-of-range values in `MARRIAGE`, `EDUCATION`, and `PAY_N` (unknown values reassigned to the *Other* category or remapped to 0)
- Train/test split: **75% / 25%**, stratified on the target variable
- **5-fold cross-validation** for hyperparameter tuning

---

## Models

### Decision Tree
- Recursive binary splitting to maximize information gain at each node
- Split criteria: Gini impurity or entropy
- Pruning applied to control tree depth and prevent overfitting
- Advantage: high interpretability

### Random Forest
- Ensemble method based on **bagging** (bootstrap aggregating)
- Random feature subset selection at each split to decorrelate trees
- More robust to noise and outliers than a single decision tree
- Provides feature importance estimates

---

## Results

| Model | Accuracy | Precision | Recall | AUC |
|---|---|---|---|---|
| Decision Tree | 0.8201 | 0.6836 | 0.3214 | 0.7325 |
| **Random Forest** | **0.8179** | **0.6392** | **0.3730** | **0.7609** |

The **Random Forest** achieves superior discriminative power (AUC = 0.761) and better default detection (Recall = 37.3%), making it the recommended model for credit risk assessment.

**Top predictive features (Random Forest):** repayment status (`PAY_1`), age (`AGE`), and bill statement amounts (`BILL_AMT`).

---

## Evaluation Metrics

| Metric | Formula |
|---|---|
| Accuracy | (TP + TN) / (TP + FP + TN + FN) |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| AUC | Area under the ROC curve — robust metric for imbalanced classes |

---

## Dependencies (R)

```r
install.packages(c(
  "ggplot2",
  "dplyr",
  "caret",
  "rpart",
  "rpart.plot",
  "randomForest",
  "pROC",
  "corrplot"
))
```

---

## Key Findings

- Recent financial behaviour (payment delays, billing amounts) are the strongest predictors of default; demographic variables play a secondary role.
- Class imbalance (~78/22 split) must be carefully handled to avoid biased classifiers.
- Random Forest outperforms the Decision Tree on recall and AUC, at the cost of interpretability.

---

## References

- Yeh, I.-C., & Lien, C.-H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* Expert Systems with Applications, 36(2), 2473–2480.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning with Applications in R.* Springer.
- Fischer, A. (2024). *Cours d'Apprentissage Statistique.* Master 2 ISIFAR, Université Paris Cité.
- Dua, D., & Graff, C. (2019). *UCI Machine Learning Repository.* University of California, Irvine.
