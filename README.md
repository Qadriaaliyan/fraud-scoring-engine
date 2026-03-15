# Fraud Scoring Engine

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)

Credit card fraud detection pipeline using XGBoost gradient boosted trees and behavioral transaction feature engineering.

This project implements a fraud detection pipeline including preprocessing, feature engineering, model training, evaluation, and inference.

The system estimates the probability that a transaction is fraudulent and classifies it using an optimized decision threshold.

---

## Overview

Financial fraud detection is a challenging machine learning problem because fraudulent transactions represent a very small minority of total activity. This creates a highly imbalanced classification problem where standard metrics like accuracy are insufficient.

This project builds a fraud scoring system that analyzes:

* transaction behavior
* merchant activity
* geographic anomalies
* historical spending patterns
* temporal transaction signals

These signals are combined into engineered features and fed into an **XGBoost classifier**, which produces a fraud probability score.

---

## Pipeline Architecture

1. Raw Transaction Data
2. Data Cleaning
3. Behavioral Feature Engineering
4. Preprocessing (StandardScaler + OneHotEncoder)
5. XGBoost Gradient Boosted Trees
6. Fraud Probability Score
7. Optimized Threshold
8. Fraud Classification

---

## Key Design Choices

### Behavioral Feature Engineering

Instead of relying only on raw transaction fields, the model derives additional behavioral signals such as:

* card spending averages
* merchant interaction statistics
* geographic transaction anomalies
* transaction timing patterns

---

### Geographic Fraud Signals

Fraud often occurs far from a cardholder’s typical location.

The model computes **Haversine distance** between:

customer coordinates and merchant coordinates

Large distance deviations may indicate suspicious transactions.

---

### Imbalanced Fraud Detection

Fraud datasets are heavily imbalanced. This pipeline addresses that using:

* `scale_pos_weight` in XGBoost
* fraud-specific F1 score optimization
* threshold tuning based on validation performance

---

## Engineered Features

Examples of engineered behavioral features used by the model:

```
avg_transaction_amount_7d
transaction_count_24h
distance_from_home
amount_to_avg_ratio
hour_of_day
merchant_visit_count
```

---

## Model

**XGBoost Gradient Boosted Decision Trees**

---

## Model Performance

Example evaluation results:

| Metric   | Score  |
| -------- | ------ |
| Fraud F1 | 0.9172 |
| ROC-AUC  | 0.9991 |
| Macro F1 | 0.9583 |
| Accuracy | 0.9991 |

Fraud F1 is prioritized because it reflects the model’s ability to correctly detect fraudulent transactions.

---

## Visualizations

The training pipeline generates several evaluation plots:

* Class Distribution
* Transaction Amount Distribution
* Transaction Amount by Fraud Label
* Confusion Matrix

---

## Project Structure

```
fraud-scoring-engine
│
├── training.py
├── inference.py
├── requirements.txt
├── README.md
│
├── data/
│   └── README.md
│
├── figures/
│   ├── class_distribution.png
│   ├── transaction_amount_distribution.png
│   ├── amount_by_fraud_label.png
│   └── confusion_matrix.png
│
└── results/
    └── model_metrics.csv
```

---

## Dataset

Datasets are not included in the repository due to size constraints.

Download them from:

[Download Dataset (Google Drive)](https://drive.google.com/drive/folders/1jvsWNirBeMli9NZxZm0x4_JIgobecuTg?usp=sharing)

After downloading, place the files in:

```
data/
├── creditcard_fraud_train.csv
└── creditcard_fraud_test.csv
```

---

## Expected Input Schema

The pipeline expects transaction data with the following columns:

```
ssn          # anonymized customer identifier
cc_num       # credit card number identifier
first        # cardholder first name
last         # cardholder last name
gender       # cardholder gender
street       # billing street address
city         # billing city
state        # billing state
zip          # postal code
lat          # customer latitude
long         # customer longitude
city_pop     # population of customer city
job          # customer occupation
dob          # date of birth
acct_num     # account identifier
profile      # customer profile category
trans_num    # transaction identifier
trans_date   # transaction date
trans_time   # transaction time
unix_time    # transaction timestamp
category     # merchant category
amt          # transaction amount
merchant     # merchant name
merch_lat    # merchant latitude
merch_long   # merchant longitude
is_fraud     # fraud label (1 = fraud, 0 = legitimate)
```

`is_fraud` is optional during inference but required for evaluation metrics.

---

## Running the Model on Your Own Data

You can use this model with your own transaction dataset if it contains the same feature columns.

### Case 1 — Dataset contains `is_fraud`

```
transactions.csv
├── features
└── is_fraud
```

The inference script will:

* generate predictions
* compare predictions with true labels
* print evaluation metrics

This is useful for benchmarking model performance on labeled datasets.

---

### Case 2 — Dataset without `is_fraud`

```
transactions.csv
└── features only
```

The model will still run inference and generate predictions internally, but evaluation metrics cannot be computed because ground truth labels are unavailable.

This reflects how fraud detection works in real-world systems.

---

## Installation

Clone the repository:

```
git clone https://github.com/Qadriaaliyan/fraud-scoring-engine.git
cd fraud-scoring-engine
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Training the Model

Run the training pipeline:

```
python training.py
```

The training script will:

1. load the dataset
2. perform preprocessing
3. engineer behavioral features
4. train the model using cross-validation
5. optimize the fraud classification threshold
6. generate evaluation visualizations
7. export the trained pipeline

Output file:

```
fraud_detection_pipeline.pkl
```

---

## Running Inference

Run:

```
python inference.py
```

The inference script will:

* load the trained model pipeline
* engineer features on new transaction data
* generate fraud predictions
* compute evaluation metrics if labels exist

---

## Example Inference Output

```
Fraud F1: 0.9171
Accuracy: 0.9991
Macro F1: 0.9583

Confusion Matrix:
[[174246     96]
 [    63    880]]
```

---

## Evaluation Strategy

Fraud detection models are evaluated using:

**Fraud F1 Score**

Measures the balance between precision and recall for the fraud class.

**Macro F1 Score**

Balances performance across both fraud and non-fraud classes.

**ROC-AUC**

Measures how well the model separates fraudulent and legitimate transactions.

**Confusion Matrix**

Provides insight into false positives and false negatives.

---

## Real-World Workflow

1. incoming transaction
2. feature engineering
3. fraud probability estimation
4. threshold decision
5. transaction flagged for review

Fraud outcomes become known later and are used to evaluate and retrain the model over time.

---

## Future Improvements

Possible extensions include:

* real-time fraud scoring API
* streaming transaction pipelines
* explainability using SHAP
* graph-based fraud detection
* anomaly detection models
* sequence modeling for temporal behavior

---
