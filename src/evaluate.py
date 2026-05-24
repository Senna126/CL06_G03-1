import os
import json
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


DATA_DIR = "artifacts/data"
MODEL_PATH = "artifacts/models/obesity_model.pkl"
PREPROCESSING_DIR = "artifacts/preprocessing"
METRICS_DIR = "artifacts/metrics"
PLOTS_DIR = "artifacts/plots"
REPORTS_DIR = "reports"


def create_directories():
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)


def load_assets():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Run python src/model.py first."
        )

    model = joblib.load(MODEL_PATH)
    X_test = joblib.load(f"{DATA_DIR}/X_test.pkl")
    y_test = joblib.load(f"{DATA_DIR}/y_test.pkl")
    target_encoder = joblib.load(f"{PREPROCESSING_DIR}/target_encoder.pkl")

    return model, X_test, y_test, target_encoder


def evaluate_model(model, X_test, y_test, target_encoder):
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_test, predictions, average="macro", zero_division=0
        )
    }

    class_names = list(target_encoder.classes_)

    report = classification_report(
        y_test,
        predictions,
        target_names=class_names,
        zero_division=0
    )

    cm = confusion_matrix(y_test, predictions)

    return metrics, report, cm, class_names


def save_outputs(metrics, report, cm, class_names):
    with open(f"{METRICS_DIR}/evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    with open(f"{REPORTS_DIR}/evaluation_report.txt", "w") as f:
        f.write("Model Evaluation Report\n")
        f.write("=======================\n\n")

        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")

        f.write("\nClassification Report\n")
        f.write("=====================\n")
        f.write(report)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    display.plot(ax=ax, xticks_rotation=45)
    plt.title("Confusion Matrix - Obesity Classification Model")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/confusion_matrix.png")
    plt.close()


def main():
    print("Starting evaluation...")

    create_directories()

    model, X_test, y_test, target_encoder = load_assets()

    metrics, report, cm, class_names = evaluate_model(
        model,
        X_test,
        y_test,
        target_encoder
    )

    save_outputs(metrics, report, cm, class_names)

    print("Evaluation completed successfully.")
    print(metrics)


if __name__ == "__main__":
    main()