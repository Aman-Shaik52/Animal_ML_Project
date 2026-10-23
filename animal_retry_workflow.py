from prefect import flow, task

import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "zoo3.csv"


@task(
    retries=2,
    retry_delay_seconds=2
)
def load_animal_dataset():

    print("\n" + "=" * 60)
    print("TASK: LOADING ANIMAL DATASET")
    print("=" * 60)

    print("Dataset Path:")
    print(DATA_PATH)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Animal dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    print("\nAnimal dataset loaded successfully!")

    print("Dataset Shape:", data.shape)

    print("\nColumns:")
    print(list(data.columns))

    return data


@task
def preprocess_data(data):

    print("\n" + "=" * 60)
    print("TASK: PREPROCESSING ANIMAL DATA")
    print("=" * 60)

    print("Preprocessing:", data.shape)

    target_column = "class_type"

    X = data.drop(
        columns=[target_column, "animal_name"]
    )

    y = data[target_column]

    print("\nTarget Column:", target_column)

    print("Number of Features:", X.shape[1])

    print("Number of Samples:", X.shape[0])

    print("\nFeature Columns:")
    print(list(X.columns))

    print("\nTarget Distribution:")
    print(y.value_counts().sort_index())

    return X, y


@flow(name="Animal Dataset Retry Workflow")
def animal_retry_workflow():

    print("\n")
    print("*" * 60)
    print(" PREFECT - ANIMAL DATASET RETRY WORKFLOW")
    print("*" * 60)

    data = load_animal_dataset()

    X, y = preprocess_data(data)

    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("Features:", X.shape)
    print("Target:", y.shape)


if __name__ == "__main__":
    animal_retry_workflow()