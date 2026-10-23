from prefect import flow, task

import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "zoo3.csv"

MODEL_PATH = BASE_DIR / "artifacts" / "validated_animal_model.joblib"


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

    print("\nDataset loaded successfully!")

    print("Dataset Shape:", data.shape)

    return data


@task
def preprocess_data(data):

    print("\n" + "=" * 60)
    print("TASK 2: PREPROCESSING ANIMAL DATA")
    print("=" * 60)

    target_column = "class_type"

    X = data.drop(
        columns=[target_column, "animal_name"]
    )

    y = data[target_column]

    print("Target:", target_column)

    print("Features:", X.shape[1])

    print("Samples:", X.shape[0])

    return X, y


@task
def train_model(X, y):

    print("\n" + "=" * 60)
    print("TASK 3: TRAINING ANIMAL MODEL")
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

    model = LogisticRegression(
        C=0.5,
        max_iter=2000,
        random_state=42
    )

    print("\nTraining Logistic Regression...")

    model.fit(X_train, y_train)

    print("Model training completed.")

    return model, X_test, y_test


@task
def evaluate_model(model, X_test, y_test):

    print("\n" + "=" * 60)
    print("TASK 4: EVALUATING ANIMAL MODEL")
    print("=" * 60)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\nModel Accuracy:")
    print(f"{accuracy:.4f}")

    print(
        f"Accuracy Percentage: {accuracy * 100:.2f}%"
    )

    return accuracy


@task
def save_model(model):

    print("\n" + "=" * 60)
    print("TASK 5: SAVING ANIMAL MODEL")
    print("=" * 60)

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print("Model saved successfully!")

    print("Model Path:")
    print(MODEL_PATH)


@flow(name="Animal Model Validation Workflow")
def animal_model_workflow():

    print("\n")
    print("*" * 60)
    print(" PREFECT - ANIMAL MODEL VALIDATION")
    print("*" * 60)

    # Load dataset
    data = load_data()

    # Preprocess dataset
    X, y = preprocess_data(data)

    # Train model
    model, X_test, y_test = train_model(
        X,
        y
    )

    # Evaluate model
    accuracy = evaluate_model(
        model,
        X_test,
        y_test
    )

    # Conditional model acceptance
    if accuracy >= 0.85:

        print("\n" + "=" * 60)
        print("MODEL ACCEPTED")
        print("=" * 60)

        save_model(model)

        print(
            f"Accuracy {accuracy * 100:.2f}% "
            "is above the required threshold."
        )

    else:

        print("\n" + "=" * 60)
        print("MODEL REJECTED")
        print("=" * 60)

        print(
            f"Accuracy {accuracy * 100:.2f}% "
            "is below the required threshold."
        )

    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    animal_model_workflow()