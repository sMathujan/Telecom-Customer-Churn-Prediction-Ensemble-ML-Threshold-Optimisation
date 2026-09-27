# Telecom Customer Churn: Is Threshold Optimisation Worth It?

Predicting telecom customer churn with Random Forest, XGBoost and CatBoost, and asking a question most churn papers skip: **does tuning the decision threshold actually help, and what is it worth in money?**

The short answer, from 30 repeated splits: tuning the threshold for F1 does **not** beat the default 0.50. But the threshold still matters, because it decides how many churners you catch and how many retention offers you waste. That is a business decision, not a modelling one.

---

## Why this project exists

A churn model outputs a probability. Something has to turn that into a decision: contact this customer, or don't. Almost every implementation uses 0.50 by default, which treats two very different mistakes as equal.

At UK prices they are not equal:

| Mistake | Cost | Source |
|---|---|---|
| Missing a customer who leaves | ~£1,656 | £69/month bundle × 24-month contract (Ofcom, Pricing Trends 2025) |
| Flagging a customer who'd have stayed | ~£99 | One year of a typical retention discount (Which?) |

One mistake costs roughly 17 times the other. A 0.50 cut-off ignores that.

---

## Headline results

Trained on the IBM Telco dataset (7,043 customers, 26.6% churn). All figures are from a held-out test set of 1,407 customers that was used **once**.

| Model | Threshold | Recall | Precision | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Random Forest | 0.39 | 88.5% | 48.3% | 0.625 | 0.847 | 0.646 |
| XGBoost | 0.45 | 86.4% | 50.3% | 0.636 | 0.852 | 0.651 |
| CatBoost | 0.44 | 81.8% | 51.3% | 0.630 | 0.845 | 0.648 |

Thresholds were chosen on validation data, never on test.

**These models are not meaningfully different.** Bootstrap 95% confidence intervals on test F1 are about 0.068 wide and every pairwise difference spans zero. Over 30 repeated splits, XGBoost and Random Forest are indistinguishable (mean difference 0.0007); only CatBoost is consistently behind.

---

## The business case: what the threshold is worth

The same Random Forest, scored on the same 1,407 customers, at two thresholds:

| | at 0.50 | at 0.37 | Change |
|---|---|---|---|
| Churners caught (of 374) | 247 | 300 | +53 |
| Offers sent to stayers | 192 | 310 | +118 |
| Accuracy | 77.3% | 72.7% | −4.6 pts |
| F1 | 0.608 | 0.610 | flat |

By the usual metrics, nothing improved. Now with UK prices:

```
Value of catching one churner = 30% × £1,656 = £496.80

at 0.50:  247 × £496.80  −  439 offers × £99  =  £79,249
at 0.37:  300 × £496.80  −  610 offers × £99  =  £88,650
                                    Difference:  +£9,401
```

**It stops paying** if an offer costs more than £154, or if fewer than 19% of contacted churners stay. If the £99 discount runs the full 24-month contract (£198), over 39% must stay.

So the right threshold depends on retention economics, not on the model. The 30% success rate is an assumption; there is no public UK figure. Figures are revenue, not profit.

---

## Method

```
1. Split first:  60% train / 20% validation / 20% test, stratified (26.6% churn in each)
2. Balance after: SMOTE on the training split only (3,098 : 1,121 → 3,098 : 3,098),
                  re-applied inside each CV fold during tuning
3. Tune:          Optuna (Bayesian) hyperparameter search, 5-fold stratified CV
4. Choose:        threshold AND model selected on validation
5. Judge:         test set opened once, with everything frozen
```

Three splits rather than two, because the threshold is itself a parameter chosen from data. If the test set helps choose it, the test set can no longer judge it.

Tuning improved ranking ability for all three models (test ROC-AUC): Random Forest 0.820 → 0.847, XGBoost 0.824 → 0.852, CatBoost 0.837 → 0.845.

---

## Repository structure

```
churn notebooks/
├── 01_EDA_and_Preprocessing.ipynb        Cleaning, encoding, exploratory analysis
├── 02_Handling_Imbalance.ipynb           Three-way split, SMOTE applied to training only
├── 03_Model_Building_and_Tuning.ipynb    Optuna tuning, threshold selection, evaluation
├── results_repeated_splits.csv           Raw output of the 30-split experiment
├── data/                                 Dataset and processed artefacts
└── plots/                                All figures
```

---

## Reproducing

```bash
git clone https://github.com/sMathujan/Telecom-Customer-Churn-Prediction-Ensemble-ML-Threshold-Optimisation.git
cd Telecom-Customer-Churn-Prediction-Ensemble-ML-Threshold-Optimisation
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in order. 

Main libraries: scikit-learn, XGBoost, CatBoost, imbalanced-learn, Optuna, SHAP, pandas, matplotlib, seaborn.

---

## Data

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — a fictional telecommunications company, published by IBM Sample Datasets. Not redistributed here beyond what is needed to reproduce the analysis.

## Author

**Mathujan Sivananthan** — MSc Data Science, University of Hertfordshire
[LinkedIn](https://linkedin.com/in/mathujan-sivananthan) · [GitHub](https://github.com/sMathujan)

