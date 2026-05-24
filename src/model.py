import os
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


DATA_DIR = "artifacts/data"
MODEL_DIR = "artifacts/models"
METRICS_DIR = "artifacts/metrics"


def create_directories():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)


def load_preprocessed_data():
    required_files = [
        "X_train.pkl",
        "X_val.pkl",
        "y_train.pkl",
        "y_val.pkl"
    ]

    for file in required_files:
        file_path = f"{DATA_DIR}/{file}"
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Missing file: {file_path}. Run python src/preprocess.py first."
            )

    X_train = joblib.load(f"{DATA_DIR}/X_train.pkl")
    X_val = joblib.load(f"{DATA_DIR}/X_val.pkl")
    y_train = joblib.load(f"{DATA_DIR}/y_train.pkl")
    y_val = joblib.load(f"{DATA_DIR}/y_val.pkl")

    return X_train, X_val, y_train, y_val


def train_models(X_train, y_train):
    random_forest = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )

    decision_tree = DecisionTreeClassifier(
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )

    print("Training Random Forest model...")
    random_forest.fit(X_train, y_train)

    print("Training Decision Tree model...")
    decision_tree.fit(X_train, y_train)

    return {
        "RandomForest": random_forest,
        "DecisionTree": decision_tree
    }


def evaluate_models(models, X_val, y_val):
    results = {}

    for model_name, model in models.items():
        predictions = model.predict(X_val)

        results[model_name] = {
            "accuracy": accuracy_score(y_val, predictions),
            "precision_macro": precision_score(y_val, predictions, average="macro", zero_division=0),
            "recall_macro": recall_score(y_val, predictions, average="macro", zero_division=0),
            "f1_macro": f1_score(y_val, predictions, average="macro", zero_division=0)
        }

    return results


def select_best_model(models, results):
    best_model_name = max(results, key=lambda name: results[name]["f1_macro"])
    best_model = models[best_model_name]

    return best_model_name, best_model


def save_model_and_metrics(best_model_name, best_model, results):
    model_path = f"{MODEL_DIR}/obesity_model.pkl"
    joblib.dump(best_model, model_path)

    training_metrics = {
        "best_model": best_model_name,
        "model_path": model_path,
        "validation_results": results
    }

    with open(f"{METRICS_DIR}/training_metrics.json", "w") as f:
        json.dump(training_metrics, f, indent=4)

    with open(f"{METRICS_DIR}/training_summary.txt", "w") as f:
        f.write("Training Summary\n")
        f.write("================\n")
        f.write(f"Best model: {best_model_name}\n\n")

        for model_name, metrics in results.items():
            f.write(f"{model_name}\n")
            for metric_name, value in metrics.items():
                f.write(f"  {metric_name}: {value:.4f}\n")
            f.write("\n")


def main():
    print("Starting model training...")

    create_directories()

    X_train, X_val, y_train, y_val = load_preprocessed_data()

    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_val, y_val)

    best_model_name, best_model = select_best_model(models, results)
    save_model_and_metrics(best_model_name, best_model, results)

    print("Training completed successfully.")
    print(f"Best model: {best_model_name}")
    print(results)


if __name__ == "__main__":
    main()