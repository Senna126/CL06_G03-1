import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


DATA_PATH = "data/obesity_data.csv"
ARTIFACT_DATA_DIR = "artifacts/data"
ARTIFACT_PREPROCESSING_DIR = "artifacts/preprocessing"
METRICS_DIR = "artifacts/metrics"

TARGET_COLUMN = "NObeyesdad"


def create_directories():
    os.makedirs(ARTIFACT_DATA_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_PREPROCESSING_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)


def load_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Add obesity_data.csv inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. Available columns: {list(df.columns)}"
        )

    return df


def encode_features(df):
    df = df.copy()

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    feature_encoders = {}
    target_encoder = LabelEncoder()

    # Encode target column
    y_encoded = target_encoder.fit_transform(y.astype(str))

    # Encode every non-numeric feature column
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col].astype(str))
            feature_encoders[col] = encoder

    # Final safety check
    non_numeric_columns = X.select_dtypes(exclude=["number"]).columns.tolist()
    if len(non_numeric_columns) > 0:
        raise ValueError(f"These columns are still non-numeric: {non_numeric_columns}")

    return X, y_encoded, feature_encoders, target_encoder


def split_and_scale(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def save_outputs(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    scaler,
    feature_encoders,
    target_encoder,
    feature_columns,
    df
):
    joblib.dump(X_train, f"{ARTIFACT_DATA_DIR}/X_train.pkl")
    joblib.dump(X_val, f"{ARTIFACT_DATA_DIR}/X_val.pkl")
    joblib.dump(X_test, f"{ARTIFACT_DATA_DIR}/X_test.pkl")

    joblib.dump(y_train, f"{ARTIFACT_DATA_DIR}/y_train.pkl")
    joblib.dump(y_val, f"{ARTIFACT_DATA_DIR}/y_val.pkl")
    joblib.dump(y_test, f"{ARTIFACT_DATA_DIR}/y_test.pkl")

    joblib.dump(scaler, f"{ARTIFACT_PREPROCESSING_DIR}/scaler.pkl")
    joblib.dump(feature_encoders, f"{ARTIFACT_PREPROCESSING_DIR}/feature_encoders.pkl")
    joblib.dump(target_encoder, f"{ARTIFACT_PREPROCESSING_DIR}/target_encoder.pkl")

    with open(f"{ARTIFACT_PREPROCESSING_DIR}/feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=4)

    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_column": TARGET_COLUMN,
        "feature_count": len(feature_columns),
        "train_size": int(len(y_train)),
        "validation_size": int(len(y_val)),
        "test_size": int(len(y_test)),
        "class_distribution": df[TARGET_COLUMN].value_counts().to_dict()
    }

    with open(f"{METRICS_DIR}/preprocessing_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    with open(f"{METRICS_DIR}/preprocessing_summary.txt", "w") as f:
        f.write("Preprocessing Summary\n")
        f.write("=====================\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")


def main():
    print("Starting preprocessing...")

    create_directories()
    df = load_dataset()

    X, y, feature_encoders, target_encoder = encode_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(X, y)

    save_outputs(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler,
        feature_encoders,
        target_encoder,
        list(X.columns),
        df
    )

    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()