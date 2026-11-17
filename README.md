# Bank Customer Churn Prediction

## Project Overview

A bank wants to predict which customers are about to leave so that retention efforts can be focused on them. This project builds a binary classifier that predicts whether a customer is about to leave (`Exited = 1`) from demographic, financial and relationship attributes. Because only about 20% of customers churn, the work focuses on handling class imbalance and on optimizing the decision threshold for the F1-score.

**Key outcome:** a class-balanced Random Forest trained on upsampled data reached an **F1-score of 0.62** (threshold ≈ 0.45) and a **ROC-AUC of 0.85** on the validation set, above the required F1 of 0.59.

## Business Problem

Identifying customers who are likely to leave allows the bank to target retention offers before the customer closes the account. A useful model has to find most of the churners (recall) without flagging too many loyal customers (precision); the F1-score balances these two goals, which is why it is the main success criterion.

## Objectives

- Predict whether a customer will leave the bank.
- Reach an **F1-score of at least 0.59**.
- Report **ROC-AUC** to evaluate how well the model ranks customers by churn risk.

## Dataset Description

`Churn.csv`, provided for educational purposes as part of the TripleTen Data Science program, contains 10,000 customers and 14 columns.

| Group | Columns |
|---|---|
| Identifiers (removed) | `RowNumber`, `CustomerId`, `Surname` |
| Demographics | `Geography`, `Gender`, `Age` |
| Financial | `CreditScore`, `Balance`, `EstimatedSalary` |
| Relationship | `Tenure`, `NumOfProducts`, `HasCrCard`, `IsActiveMember` |
| **Target** | `Exited` (1 = the customer left the bank) |

**Data quality**

- `Tenure` has 909 missing values (9,091 non-null).
- The target is imbalanced: 20.4% of the customers churned (20.0% in the training split and 21.4% in the validation split).

## Technologies and Libraries

- Python
- pandas, NumPy
- scikit-learn (`RandomForestClassifier`, `train_test_split`, `f1_score`, `roc_auc_score`, `shuffle`)
- Jupyter Notebook

## Methodology

1. **Feature selection:** `RowNumber`, `CustomerId` and `Surname` are dropped, since identifiers carry no predictive information and could add noise.
2. **Missing values:** a binary indicator `has_deposit` (1 when `Tenure` is present) is created, and missing `Tenure` values are filled with 0 as a sentinel. The indicator keeps the information about whether the value was missing, which a plain imputation would lose. The notebook interprets a missing `Tenure` as the customer not having a fixed deposit.
3. **Encoding:** `Geography` and `Gender` are one-hot encoded with `pd.get_dummies(drop_first=True)` to avoid redundant columns.
4. **Split:** 75% training and 25% validation (`random_state=12345`).
5. **Class imbalance:** two techniques are combined:
   - **upsampling** of the minority class in the training set (each churner replicated 10 times, then shuffled);
   - `class_weight='balanced'` in the Random Forest.
6. **Model:** `RandomForestClassifier` with default hyperparameters (`random_state=12345`).
7. **Threshold tuning:** decision thresholds from 0.10 to 0.89 (step 0.01) are applied to the predicted probabilities, and the threshold with the best F1-score is selected.

## Results and Model Performance

All metrics below are computed on the **validation set** (2,500 customers).

| Stage | Decision threshold | F1-score | ROC-AUC |
|---|---|---|---|
| Balanced Random Forest | 0.50 (default) | 0.608 | 0.846 |
| **After threshold tuning** | **≈ 0.45** | **0.621** | 0.846 |

Requirement: F1 ≥ 0.59. ROC-AUC does not depend on the threshold, so it does not change after tuning.

## Key Findings

- Even with the default threshold, the balanced Random Forest already exceeds the F1 requirement (0.608 > 0.59).
- Lowering the threshold to about 0.45 increases the F1-score to 0.621, a better trade-off between catching churners and avoiding false alarms.
- A ROC-AUC of about 0.85 shows that the model ranks customers by churn risk well, which also allows the bank to adjust the threshold to its retention budget.

## Conclusions

**Achievement of objectives:** the model exceeds the required F1-score of 0.59 and achieves a good ROC-AUC, so it can identify customers with a higher probability of churn.

**Limitations**

- **No independent test set:** the decision threshold is selected and evaluated on the same validation set. In the notebook, the "test" metrics are computed on `features_valid`, so the final F1 of 0.621 is optimistic.
- Upsampling and `class_weight='balanced'` are applied at the same time, which corrects the imbalance twice. Their individual effects were not compared.
- The Random Forest uses default hyperparameters; no hyperparameter search was performed.
- No baseline model (for example, a majority-class or logistic regression model) was trained for comparison.
- Interpreting a missing `Tenure` as "no fixed deposit" is an assumption that is not documented in the data.

**Potential improvements:** use a three-way split (training, validation and test) or cross-validation, tune hyperparameters, compare each balancing strategy separately, add a baseline, and analyze feature importance to explain the drivers of churn.

## Repository Structure

```text
bank-customer-churn-prediction/
├── README.md
├── main.ipynb
└── datasets/
    └── Churn.csv
```

## How to Run

1. Create and activate a Python 3 environment (the notebook was last run with Python 3.13):

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

2. Install the dependencies:

   ```bash
   pip install pandas numpy scikit-learn jupyter
   ```

3. Start Jupyter **from the project root**, so that the relative path `./datasets/Churn.csv` resolves correctly, and open `main.ipynb`:

   ```bash
   jupyter notebook main.ipynb
   ```

Some printed labels are in Portuguese, as in the original run.

## Author

**Luiz Otávio Carneiro Filho**
Geospatial Engineer | GIS & Data Science
[LinkedIn](https://www.linkedin.com/in/luizot/)
