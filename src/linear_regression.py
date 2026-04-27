"""Linear Regression model for Student Performance Prediction.

This script loads the two provided datasets, combines them, preprocesses
categorical/numerical features, trains a Linear Regression model, evaluates it,
and saves the model, metrics, and visualizations.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
FIG_DIR = ROOT_DIR / "outputs" / "figures"
MODEL_DIR = ROOT_DIR / "outputs" / "models"
RESULT_DIR = ROOT_DIR / "outputs" / "results"

for folder in [FIG_DIR, MODEL_DIR, RESULT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load Math and Portuguese datasets and combine them."""
    math_df = pd.read_csv(DATA_DIR / "student-mat.csv", sep=";")
    por_df = pd.read_csv(DATA_DIR / "student-por.csv", sep=";")

    math_df["subject"] = "math"
    por_df["subject"] = "portuguese"

    data = pd.concat([math_df, por_df], ignore_index=True)
    data = data.drop_duplicates()
    return data


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing pipeline for numeric and categorical columns."""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def evaluate_model(y_true, y_pred) -> dict:
    """Calculate regression evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"Model": "Linear Regression", "MAE": mae, "RMSE": rmse, "R2 Score": r2}


def save_plots(y_true, y_pred) -> None:
    """Save actual-vs-predicted, residual, and error distribution plots."""
    residuals = y_true - y_pred

    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.xlabel("Actual Final Grade (G3)")
    plt.ylabel("Predicted Final Grade (G3)")
    plt.title("Linear Regression: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "linear_regression_actual_vs_predicted.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Final Grade (G3)")
    plt.ylabel("Residuals")
    plt.title("Linear Regression: Residual Plot")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "linear_regression_residual_plot.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, kde=True)
    plt.xlabel("Prediction Error")
    plt.title("Linear Regression: Error Distribution")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "linear_regression_error_distribution.png", dpi=300)
    plt.close()


def main() -> None:
    data = load_data()

    X = data.drop(columns=["G3"])
    y = data["G3"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("regressor", LinearRegression()),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = evaluate_model(y_test, predictions)
    pd.DataFrame([metrics]).to_csv(RESULT_DIR / "linear_regression_metrics.csv", index=False)

    joblib.dump(model, MODEL_DIR / "linear_regression_model.pkl")
    save_plots(y_test, predictions)

    print("Linear Regression completed.")
    print(metrics)


if __name__ == "__main__":
    main()
