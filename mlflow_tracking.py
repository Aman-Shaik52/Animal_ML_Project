
import mlflow
import mlflow.sklearn

from ml_pipeline import (
    load_data,
    prepare_data,
    split_data,
    build_pipeline,
    train_model,
    evaluate_model
)

mlflow.set_experiment("Animal-Classification")

with mlflow.start_run():
    df = load_data()
    X, y, num_cols, cat_cols = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = build_pipeline(num_cols, cat_cols)
    model = train_model(model, X_train, y_train)

    metrics, _ = evaluate_model(model, X_test, y_test)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("dataset", "zoo3.csv")
    mlflow.log_metric("accuracy", metrics["accuracy"])
    mlflow.log_metric("f1_weighted", metrics["f1_weighted"])

    mlflow.sklearn.log_model(
    model,
    name="animal_model",
    serialization_format="pickle"
)

    print("MLflow tracking completed!")
