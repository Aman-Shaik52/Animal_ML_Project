from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "zoo3.csv"

ARTIFACT_DIR = BASE_DIR / "artifacts"
OUTPUT_DIR = BASE_DIR / "outputs"

ARTIFACT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "animal_pipeline.joblib"

METRICS_PATH = OUTPUT_DIR / "metrics.json"
REPORT_PATH = OUTPUT_DIR / "pipeline_report.txt"
PREDICTION_PATH = OUTPUT_DIR / "sample_prediction.csv"


# ============================================================
# 2. CONFIGURATION
# ============================================================

TARGET_COLUMN = "class_type"
ID_COLUMN = "animal_name"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# 3. LOAD DATA
# ============================================================

def load_data():

    print("\n" + "=" * 60)
    print("STEP 1: LOAD DATA")
    print("=" * 60)

    print(f"Dataset path: {DATA_PATH}")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset not found!\n"
            f"Expected file:\n{DATA_PATH}\n\n"
            f"Make sure zoo3.csv is inside:\n{BASE_DIR}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Original dataset shape: {df.shape}")

    # Remove unwanted index columns
    unwanted_columns = [
        col for col in df.columns
        if col.lower().startswith("unnamed")
    ]

    if unwanted_columns:
        df = df.drop(columns=unwanted_columns)

    # Remove completely empty columns
    empty_columns = [
        col for col in df.columns
        if df[col].isna().all()
    ]

    if empty_columns:
        df = df.drop(columns=empty_columns)

    # Remove duplicate rows
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        df = df.drop_duplicates()

    print(f"Cleaned dataset shape: {df.shape}")
    print(f"Duplicate rows removed: {duplicate_count}")

    print("\nDataset columns:")
    print(list(df.columns))

    print("\nFirst 5 rows:")
    print(df.head())

    return df


# ============================================================
# 4. PREPARE X AND y
# ============================================================

def prepare_data(df):

    print("\n" + "=" * 60)
    print("STEP 2: PREPARE FEATURES AND TARGET")
    print("=" * 60)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    y = df[TARGET_COLUMN]

    X = df.drop(columns=[TARGET_COLUMN])

    # Animal name is an identifier and should not be used
    # as a predictive feature.
    if ID_COLUMN in X.columns:
        X = X.drop(columns=[ID_COLUMN])

        print(f"Removed identifier column: {ID_COLUMN}")

    # Detect numerical and categorical columns
    numerical_columns = X.select_dtypes(
        include=["int64", "int32", "float64", "float32"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print(f"\nTarget column: {TARGET_COLUMN}")

    print(f"Number of input features: {X.shape[1]}")

    print("\nNumerical columns:")
    print(numerical_columns)

    print("\nCategorical columns:")
    print(categorical_columns)

    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    return X, y, numerical_columns, categorical_columns


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y):

    print("\n" + "=" * 60)
    print("STEP 3: TRAIN / TEST SPLIT")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test


# ============================================================
# 6. BUILD PREPROCESSING + MODEL PIPELINE
# ============================================================

def build_pipeline(numerical_columns, categorical_columns):

    print("\n" + "=" * 60)
    print("STEP 4: BUILD ML PIPELINE")
    print("=" * 60)

    # Numerical preprocessing
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical_columns
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    # Logistic Regression model
    model = LogisticRegression(
        C=0.5,
        max_iter=2000,
        random_state=RANDOM_STATE
    )

    # Complete pipeline
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    print("Preprocessing pipeline created.")
    print("Model: Logistic Regression")

    return pipeline


# ============================================================
# 7. TRAIN MODEL
# ============================================================

def train_model(pipeline, X_train, y_train):

    print("\n" + "=" * 60)
    print("STEP 5: TRAIN MODEL")
    print("=" * 60)

    pipeline.fit(X_train, y_train)

    print("Model training completed successfully.")

    return pipeline


# ============================================================
# 8. EVALUATE MODEL
# ============================================================

def evaluate_model(pipeline, X_test, y_test):

    print("\n" + "=" * 60)
    print("STEP 6: EVALUATE MODEL")
    print("=" * 60)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "f1_weighted": round(float(f1), 4)
    }

    print("\nModel Evaluation:")
    print("-" * 40)
    print(f"Accuracy           : {metrics['accuracy']}")
    print(f"Weighted Precision : {metrics['precision_weighted']}")
    print(f"Weighted Recall    : {metrics['recall_weighted']}")
    print(f"Weighted F1 Score  : {metrics['f1_weighted']}")

    return metrics, y_pred


# ============================================================
# 9. SAVE MODEL
# ============================================================

def save_model(pipeline):

    print("\n" + "=" * 60)
    print("STEP 7: SAVE MODEL")
    print("=" * 60)

    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    print(f"Model saved successfully:")
    print(MODEL_PATH)


# ============================================================
# 10. RELOAD MODEL AND PREDICT
# ============================================================

def reload_and_predict(X_test):

    print("\n" + "=" * 60)
    print("STEP 8: RELOAD MODEL + SAMPLE PREDICTION")
    print("=" * 60)

    loaded_pipeline = joblib.load(
        MODEL_PATH
    )

    sample = X_test.iloc[[0]]

    prediction = loaded_pipeline.predict(
        sample
    )

    print("\nSample input:")
    print(sample)

    print("\nPredicted class_type:")
    print(prediction[0])

    prediction_df = pd.DataFrame({
        "predicted_class_type": [prediction[0]]
    })

    prediction_df.to_csv(
        PREDICTION_PATH,
        index=False
    )

    print(f"\nSample prediction saved:")
    print(PREDICTION_PATH)

    return prediction[0]


# ============================================================
# 11. SAVE REPORT
# ============================================================

def save_outputs(
    df,
    X,
    X_train,
    X_test,
    metrics
):

    print("\n" + "=" * 60)
    print("STEP 9: SAVE OUTPUT REPORTS")
    print("=" * 60)

    # Save metrics
    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # Save text report
    report = f"""
END-TO-END MACHINE LEARNING PIPELINE
====================================

Dataset
-------
File: {DATA_PATH.name}

Original/Cleaned Dataset Shape
------------------------------
Rows: {df.shape[0]}
Columns: {df.shape[1]}

Target
------
Target column: {TARGET_COLUMN}

Features
--------
Number of input features: {X.shape[1]}

Train/Test Split
----------------
Training samples: {X_train.shape[0]}
Testing samples: {X_test.shape[0]}

Model
-----
Logistic Regression

Model Parameters
----------------
C = 0.5
max_iter = 2000
random_state = {RANDOM_STATE}

Preprocessing
-------------
Numerical:
- Median imputation
- StandardScaler

Categorical:
- Most-frequent imputation
- OneHotEncoder(handle_unknown='ignore')

Evaluation
----------
Accuracy: {metrics['accuracy']}
Weighted Precision: {metrics['precision_weighted']}
Weighted Recall: {metrics['recall_weighted']}
Weighted F1 Score: {metrics['f1_weighted']}

Saved Model
-----------
{MODEL_PATH}

Saved Metrics
-------------
{METRICS_PATH}

Saved Sample Prediction
-----------------------
{PREDICTION_PATH}
"""

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report.strip())

    print(f"Metrics saved to:")
    print(METRICS_PATH)

    print(f"\nPipeline report saved to:")
    print(REPORT_PATH)


# ============================================================
# 12. MAIN PIPELINE
# ============================================================

def main():

    print("\n")
    print("*" * 60)
    print(" END-TO-END ML PIPELINE - ANIMAL CLASSIFICATION")
    print("*" * 60)

    # STEP 1
    df = load_data()

    # STEP 2
    (
        X,
        y,
        numerical_columns,
        categorical_columns
    ) = prepare_data(df)

    # STEP 3
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(X, y)

    # STEP 4
    pipeline = build_pipeline(
        numerical_columns,
        categorical_columns
    )

    # STEP 5
    pipeline = train_model(
        pipeline,
        X_train,
        y_train
    )

    # STEP 6
    metrics, y_pred = evaluate_model(
        pipeline,
        X_test,
        y_test
    )

    # STEP 7
    save_model(pipeline)

    # STEP 8
    reload_and_predict(
        X_test
    )

    # STEP 9
    save_outputs(
        df,
        X,
        X_train,
        X_test,
        metrics
    )

    print("\n" + "*" * 60)
    print(" PIPELINE COMPLETED SUCCESSFULLY")
    print("*" * 60)

    print("\nGenerated folders/files:")
    print(f"  {ARTIFACT_DIR}")
    print(f"  {OUTPUT_DIR}")

    print("\nDone!")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
    # ============================================================
# 10. PREDICT MULTIPLE ANIMALS
# ============================================================

def predict_multiple_animals(X_test):

    print("\n" + "=" * 60)
    print("STEP 10: MULTIPLE ANIMAL PREDICTIONS")
    print("=" * 60)

    # Load the saved complete pipeline
    loaded_pipeline = joblib.load(MODEL_PATH)

    # Select first 5 test samples
    samples = X_test.head(5)

    # Predict class_type
    predictions = loaded_pipeline.predict(samples)

    # Create prediction table
    prediction_results = samples.copy()

    prediction_results["predicted_class_type"] = predictions

    print("\nAnimal Predictions:")
    print("-" * 60)
    print(prediction_results)

    # Save predictions
    multiple_prediction_path = (
        OUTPUT_DIR / "multiple_animal_predictions.csv"
    )

    prediction_results.to_csv(
        multiple_prediction_path,
        index=False
    )

    print("\nMultiple predictions saved to:")
    print(multiple_prediction_path)

    return prediction_results