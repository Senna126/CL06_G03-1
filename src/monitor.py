import os
import json
import pandas as pd
from scipy.stats import ks_2samp


TRAIN_DATA_PATH = "data/obesity_data.csv"
NEW_DATA_PATH = "data/new_data.csv"

REPORTS_DIR = "reports"
METRICS_DIR = "artifacts/metrics"

NUMERIC_COLUMNS_TO_MONITOR = [
    "Age",
    "Height",
    "Weight",
    "BMI"
]


def create_directories():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)


def add_bmi_if_missing(df):
    df = df.copy()

    if "BMI" not in df.columns:
        if "Height" in df.columns and "Weight" in df.columns:
            df["BMI"] = df["Weight"] / (df["Height"] ** 2)

    return df


def load_data():
    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Training data not found: {TRAIN_DATA_PATH}")

    if not os.path.exists(NEW_DATA_PATH):
        raise FileNotFoundError(f"New data not found: {NEW_DATA_PATH}")

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    new_df = pd.read_csv(NEW_DATA_PATH)

    train_df = add_bmi_if_missing(train_df)
    new_df = add_bmi_if_missing(new_df)

    return train_df, new_df


def run_drift_checks(train_df, new_df):
    drift_results = {}
    drift_detected = False

    for column in NUMERIC_COLUMNS_TO_MONITOR:
        if column in train_df.columns and column in new_df.columns:
            train_values = train_df[column].dropna()
            new_values = new_df[column].dropna()

            statistic, p_value = ks_2samp(train_values, new_values)

            column_drift_detected = p_value < 0.05

            drift_results[column] = {
                "ks_statistic": float(statistic),
                "p_value": float(p_value),
                "drift_detected": bool(column_drift_detected)
            }

            if column_drift_detected:
                drift_detected = True

    return drift_results, drift_detected


def save_monitoring_report(drift_results, drift_detected):
    monitoring_summary = {
        "method": "Kolmogorov-Smirnov test",
        "drift_threshold": "p_value < 0.05",
        "drift_detected": drift_detected,
        "features_checked": list(drift_results.keys()),
        "results": drift_results
    }

    with open(f"{METRICS_DIR}/monitoring_metrics.json", "w") as f:
        json.dump(monitoring_summary, f, indent=4)

    with open(f"{REPORTS_DIR}/monitoring_report.txt", "w") as f:
        f.write("Monitoring and Drift Detection Report\n")
        f.write("====================================\n\n")
        f.write("Method: Kolmogorov-Smirnov test\n")
        f.write("Threshold: p_value < 0.05\n")
        f.write(f"Overall drift detected: {drift_detected}\n\n")

        for column, result in drift_results.items():
            f.write(f"Feature: {column}\n")
            f.write(f"  KS statistic: {result['ks_statistic']:.4f}\n")
            f.write(f"  p-value: {result['p_value']:.4f}\n")
            f.write(f"  Drift detected: {result['drift_detected']}\n\n")

        if drift_detected:
            f.write("Action: Drift detected. The model should be retrained using updated data.\n")
        else:
            f.write("Action: No significant drift detected. The current model remains acceptable.\n")


def main():
    print("Starting monitoring and drift detection...")

    create_directories()

    train_df, new_df = load_data()
    drift_results, drift_detected = run_drift_checks(train_df, new_df)

    save_monitoring_report(drift_results, drift_detected)

    print("Monitoring completed successfully.")
    print(f"Drift detected: {drift_detected}")
    print(drift_results)


if __name__ == "__main__":
    main()