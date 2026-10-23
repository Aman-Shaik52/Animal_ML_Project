from prefect import flow, task

import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "zoo3.csv"

MODEL_PATH = BASE_DIR / "artifacts" / "prefect_animal_model.joblib"


# ============================================================
# TASK 1: LOAD ANIMAL DATASET
# ============================================================

@task
def load_data():

    print("\n" + "=" * 60)
    print("TASK 1: LOADING ANIMAL DATASET")
    print("=" * 60)

    print("Dataset:", DATA_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print("Dataset shape:", data.shape)

    print("\nColumns:")
    print(list(data.columns))

    return data


# ============================================================
# TASK 2: PREPROCESS DATA
# ============================================================

@task
def preprocess_data(data):

    print("\n" + "=" * 60)
    print("TASK 2: PREPROCESSING ANIMAL DATA")
    print("=" * 60)

    # Target
    target_column = "class_type"

    # Remove animal name because it is an identifier,
    # not a useful numerical prediction feature.
    X = data.drop(
        columns=[target_column, "animal_name"]
    )

    y = data[target_column]

    print("Target column:", target_column)

    print("Features:", X.shape[1])

    print("\nFeature columns:")
    print(list(X.columns))

    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    return X, y


# ============================================================
# TASK 3: TRAIN MODEL
# ============================================================

@task
def train_model(X, y):

    print("\n" + "=" * 60)
    print("TASK 3: TRAINING ANIMAL CLASSIFICATION MODEL")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Logistic Regression
    model = LogisticRegression(
        C=0.5,
        max_iter=2000,
        random_state=42
    )

    print("\nTraining Logistic Regression...")

    model.fit(
        X_train,
        y_train
    )

    print("Model training completed.")

    # Save model
    MODEL_PATH.parent.mkdir(
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print("\nModel saved to:")
    print(MODEL_PATH)

    return model, X_test, y_test


# ============================================================
# TASK 4: EVALUATE MODEL
# ============================================================

@task
def evaluate_model(model, X_test, y_test):

    print("\n" + "=" * 60)
    print("TASK 4: EVALUATING ANIMAL CLASSIFICATION MODEL")
    print("=" * 60)

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\nModel Evaluation")
    print("-" * 40)

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Accuracy Percentage: {accuracy * 100:.2f}%"
    )

    return accuracy


# ============================================================
# PREFECT FLOW
# ============================================================

@flow(name="Animal Classification ML Workflow")
def animal_ml_workflow():

    print("\n")
    print("*" * 60)
    print(" PREFECT - ANIMAL CLASSIFICATION ML PIPELINE")
    print("*" * 60)

    # STEP 1
    data = load_data()

    # STEP 2
    X, y = preprocess_data(data)

    # STEP 3
    model, X_test, y_test = train_model(
        X,
        y
    )

    # STEP 4
    accuracy = evaluate_model(
        model,
        X_test,
        y_test
    )

    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETED")
    print("=" * 60)

    print(
        f"Final Accuracy: {accuracy:.4f}"
    )

    print(
        f"Final Accuracy Percentage: {accuracy * 100:.2f}%"
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    animal_ml_workflow()