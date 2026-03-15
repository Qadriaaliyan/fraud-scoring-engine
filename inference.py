import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# load trained pipeline

bundle = joblib.load("fraud_detection_pipeline.pkl")

feature_stats = bundle["feature_engineering"]
preprocessor = bundle["preprocessor"]
model = bundle["model"]
threshold = bundle["threshold"]


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

    lat1 = np.radians(data_copy["lat"])
    lon1 = np.radians(data_copy["long"])
    lat2 = np.radians(data_copy["merch_lat"])
    lon2 = np.radians(data_copy["merch_long"])

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
    data_copy["distance_amount_ratio"] = data_copy["distance_km"] / data_copy["amt"]


    # drop unused columns

    drop_columns = [
        "ssn", "first", "last",
        "street", "city", "state",
        "dob", "cc_num",
        "birth_date", "transaction_datetime"
    ]

    keep_columns = [c for c in data_copy.columns if c not in drop_columns]

    return data_copy[keep_columns]


# run inference

def run_fraud_detection(test_csv_path):

    df = pd.read_csv(test_csv_path)

    has_labels = "is_fraud" in df.columns

    if has_labels:
        y_true = df["is_fraud"]
        X = df.drop(columns=["is_fraud"])
    else:
        X = df


    # feature engineering

    X_features = transform_features(X, feature_stats)


    # preprocessing

    X_processed = preprocessor.transform(X_features)


    # predictions

    y_prob = model.predict_proba(X_processed)[:, 1]

    y_pred = (y_prob >= threshold).astype(int)


    # evaluation if labels exist

    if has_labels:

        cm = confusion_matrix(y_true, y_pred)

        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("Macro F1:", f1_score(y_true, y_pred, average="macro"))
        print("Fraud F1:", f1_score(y_true, y_pred, pos_label=1))

        print("\nConfusion Matrix:")
        print(cm)


# entry point

if __name__ == "__main__":

    run_fraud_detection("data/creditcard_fraud_test.csv")
