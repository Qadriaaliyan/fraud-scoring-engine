import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.base import clone

from xgboost import XGBClassifier
import joblib

# set global font
plt.rcParams["font.family"] = "serif"


# create output folders

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)


# load dataset

df = pd.read_csv("data/creditcard_fraud_train.csv")


# ensure coordinate columns are numeric

for c in ["lat", "long", "merch_lat", "merch_long"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")


# remove rows with invalid coordinates

df = df.dropna(subset=["lat", "long", "merch_lat", "merch_long"])


df = df.drop_duplicates()
df = df.dropna(subset=["is_fraud"])


# ensure date columns are strings

for column in ["dob", "trans_date", "trans_time"]:
    df[column] = df[column].astype(str)


y = df["is_fraud"]
X = df.drop(columns=["is_fraud"])


# class distribution plot

plt.figure(figsize=(4, 4))
y.value_counts().plot(kind="bar", color=["#26547C", "#EF476F"])
plt.title("Class Distribution")
plt.xlabel("Fraud Label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("figures/class_distribution.png", dpi=300)
plt.show()


# transaction amount distribution

plt.figure(figsize=(6, 4))
sns.histplot(df["amt"], bins=100, color="#FFB703")
plt.xlim(0, 1000)
plt.yscale("log")
plt.title("Transaction Amount Distribution")
plt.tight_layout()
plt.savefig("figures/transaction_amount_distribution.png", dpi=300)
plt.show()


# amount vs fraud label

plt.figure(figsize=(6, 4))
sns.boxplot(
    x="is_fraud",
    y="amt",
    hue="is_fraud",
    data=df,
    palette=["#26547C", "#EF476F"],
    legend=False
)

plt.title("Transaction Amount by Fraud Label")
plt.tight_layout()
plt.savefig("figures/amount_by_fraud_label.png", dpi=300)
plt.show()


# feature engineering fit

def fit_feature_engineering(data):

    data_copy = data.copy()

    data_copy["birth_date"] = pd.to_datetime(data_copy["dob"], errors="coerce")

    data_copy["transaction_datetime"] = pd.to_datetime(
        data_copy["trans_date"] + " " + data_copy["trans_time"],
        errors="coerce"
    )

    data_copy["customer_age"] = (
        data_copy["transaction_datetime"].dt.year -
        data_copy["birth_date"].dt.year
    )

    card_stats = (
        data_copy.groupby("cc_num")["amt"]
        .agg(["mean", "count"])
        .rename(columns={
            "mean": "card_avg_amount",
            "count": "card_transaction_count"
        })
    )

    merchant_stats = (
        data_copy.groupby("merchant")["amt"]
        .agg(["mean", "count"])
        .rename(columns={
            "mean": "merchant_avg_amount",
            "count": "merchant_transaction_count"
        })
    )

    return {
        "card_stats": card_stats,
        "merchant_stats": merchant_stats
    }


# feature engineering transform

def transform_features(data, feature_stats):

    data_copy = data.copy()

    data_copy["birth_date"] = pd.to_datetime(data_copy["dob"], errors="coerce")

    data_copy["transaction_datetime"] = pd.to_datetime(
        data_copy["trans_date"] + " " + data_copy["trans_time"],
        errors="coerce"
    )

    data_copy["customer_age"] = (
        data_copy["transaction_datetime"].dt.year -
        data_copy["birth_date"].dt.year
    )

    data_copy["transaction_hour"] = data_copy["transaction_datetime"].dt.hour
    data_copy["day_of_week"] = data_copy["transaction_datetime"].dt.dayofweek
    data_copy["is_weekend"] = (data_copy["day_of_week"] >= 5).astype(int)


    # haversine distance

    earth_radius_km = 6371.0

    lat1 = np.radians(data_copy["lat"].values)
    lon1 = np.radians(data_copy["long"].values)

    lat2 = np.radians(data_copy["merch_lat"].values)
    lon2 = np.radians(data_copy["merch_long"].values)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        np.sin(delta_lat / 2) ** 2 +
        np.cos(lat1) * np.cos(lat2) *
        np.sin(delta_lon / 2) ** 2
    )

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    data_copy["distance_km"] = earth_radius_km * c


    # card aggregates

    data_copy["card_avg_amount"] = (
        data_copy["cc_num"]
        .map(feature_stats["card_stats"]["card_avg_amount"])
        .fillna(data_copy["amt"])
    )

    data_copy["card_transaction_count"] = (
        data_copy["cc_num"]
        .map(feature_stats["card_stats"]["card_transaction_count"])
        .fillna(1)
    )


    # merchant aggregates

    data_copy["merchant_avg_amount"] = (
        data_copy["merchant"]
        .map(feature_stats["merchant_stats"]["merchant_avg_amount"])
        .fillna(data_copy["amt"])
    )

    data_copy["merchant_transaction_count"] = (
        data_copy["merchant"]
        .map(feature_stats["merchant_stats"]["merchant_transaction_count"])
        .fillna(1)
    )


    # ratio features

    data_copy["amount_ratio"] = data_copy["amt"] / data_copy["card_avg_amount"]

    data_copy["distance_amount_ratio"] = (
        data_copy["distance_km"] / data_copy["amt"]
    )


    # drop columns

    drop_columns = [
        "ssn", "first", "last",
        "street", "city", "state",
        "dob", "cc_num"
    ]

    for column in drop_columns:
        if column in data_copy:
            data_copy = data_copy.drop(columns=column)

    data_copy = data_copy.drop(columns=["birth_date", "transaction_datetime"])

    return data_copy


# preprocessing

def fit_preprocessor(data):

    transformer = ColumnTransformer(
        [
            ("numeric", StandardScaler(),
             make_column_selector(dtype_include=np.number)),

            ("categorical", OneHotEncoder(handle_unknown="ignore"),
             make_column_selector(dtype_include=object))
        ]
    )

    transformer.fit(data)

    return transformer


def transform_preprocessor(data, transformer):

    return transformer.transform(data)


# train test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# class imbalance

num_negative = (y_train == 0).sum()
num_positive = (y_train == 1).sum()

scale_pos_weight = num_negative / num_positive * 1.2


# model

xgb_model = XGBClassifier(
    n_estimators=850,
    learning_rate=0.07,
    max_depth=7,
    subsample=0.9,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)


# cross validation

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fraud_f1_scores = []
macro_f1_scores = []
roc_auc_scores = []


for train_idx, val_idx in cv.split(X_train, y_train):

    X_train_fold = X_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]

    X_val_fold = X_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]

    feature_stats = fit_feature_engineering(X_train_fold)

    X_train_fe = transform_features(X_train_fold, feature_stats)
    X_val_fe = transform_features(X_val_fold, feature_stats)

    preprocessor = fit_preprocessor(X_train_fe)

    X_train_processed = transform_preprocessor(X_train_fe, preprocessor)
    X_val_processed = transform_preprocessor(X_val_fe, preprocessor)

    model = clone(xgb_model)

    model.fit(X_train_processed, y_train_fold)

    y_prob = model.predict_proba(X_val_processed)[:, 1]

    y_pred = (y_prob >= 0.5).astype(int)

    fraud_f1_scores.append(f1_score(y_val_fold, y_pred, pos_label=1))
    macro_f1_scores.append(f1_score(y_val_fold, y_pred, average="macro"))
    roc_auc_scores.append(roc_auc_score(y_val_fold, y_prob))


# threshold optimization

X_train_base, X_val_base, y_train_base, y_val_base = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    stratify=y_train,
    random_state=42
)

feature_stats = fit_feature_engineering(X_train_base)

X_train_fe = transform_features(X_train_base, feature_stats)
X_val_fe = transform_features(X_val_base, feature_stats)

preprocessor = fit_preprocessor(X_train_fe)

X_train_processed = transform_preprocessor(X_train_fe, preprocessor)
X_val_processed = transform_preprocessor(X_val_fe, preprocessor)

model = clone(xgb_model)

model.fit(X_train_processed, y_train_base)

y_val_prob = model.predict_proba(X_val_processed)[:, 1]

thresholds = np.linspace(0.1, 0.9, 301)

best_threshold = 0.5
best_f1 = 0.0

for threshold in thresholds:

    f1 = f1_score(
        y_val_base,
        (y_val_prob >= threshold).astype(int),
        pos_label=1
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


# final training

feature_stats = fit_feature_engineering(X_train)

X_train_fe = transform_features(X_train, feature_stats)
X_test_fe = transform_features(X_test, feature_stats)

preprocessor = fit_preprocessor(X_train_fe)

X_train_processed = transform_preprocessor(X_train_fe, preprocessor)
X_test_processed = transform_preprocessor(X_test_fe, preprocessor)

final_model = clone(xgb_model)

final_model.fit(X_train_processed, y_train)

y_test_prob = final_model.predict_proba(X_test_processed)[:, 1]

y_test_pred = (y_test_prob >= best_threshold).astype(int)


# save model

pipeline_bundle = {
    "feature_engineering": feature_stats,
    "preprocessor": preprocessor,
    "model": final_model,
    "threshold": best_threshold,
}

joblib.dump(pipeline_bundle, "fraud_detection_pipeline.pkl")


# confusion matrix

conf_matrix = confusion_matrix(y_test, y_test_pred)

annot = [
    [f"{conf_matrix[0,0]} (TN)", f"{conf_matrix[0,1]} (FP)"],
    [f"{conf_matrix[1,0]} (FN)", f"{conf_matrix[1,1]} (TP)"]
]

plt.figure(figsize=(6,4))

mask = np.array([[1,0],[0,1]])

sns.heatmap(
    mask,
    annot=annot,
    fmt="",
    cmap=sns.color_palette(["#EF476F","#26547C"], as_cmap=True),
    cbar=False
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.tight_layout()
plt.savefig("figures/confusion_matrix.png", dpi=300)

plt.show()


# save metrics

metrics = {
    "accuracy": accuracy_score(y_test, y_test_pred),
    "macro_f1": f1_score(y_test, y_test_pred, average="macro"),
    "fraud_f1": f1_score(y_test, y_test_pred, pos_label=1),
    "roc_auc_cv": np.mean(roc_auc_scores)
}

metrics_df = pd.DataFrame([metrics])

metrics_df.to_csv("results/model_metrics.csv", index=False)


print("CV Fraud F1 mean:", np.mean(fraud_f1_scores))
print("\nCV Macro F1 mean:", np.mean(macro_f1_scores))
print("\nCV ROC-AUC mean:", np.mean(roc_auc_scores))

print("\nBest threshold:", best_threshold, "F1:", best_f1)

print("\nTest Accuracy:", metrics["accuracy"])
print("\nTest Macro F1:", metrics["macro_f1"])
print("\nTest Fraud F1:", metrics["fraud_f1"])

print("\nModel saved: fraud_detection_pipeline.pkl\n")
