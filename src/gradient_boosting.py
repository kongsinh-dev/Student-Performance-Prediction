"""Gradient Boosting Regressor model for Student Performance Prediction.

This script trains a Gradient Boosting model with hyperparameter tuning,
evaluates it, and saves model artifacts, metrics, plots, and feature importance.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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
    """Create preprocessing pipeline."""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
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


def get_feature_names(model: Pipeline, X: pd.DataFrame) -> list:
    """Get transformed feature names."""
    preprocessor = model.named_steps["preprocessor"]
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    categorical_names = onehot.get_feature_names_out(categorical_features).tolist()
    return numeric_features + categorical_names


def evaluate_model(y_true, y_pred) -> dict:
    """Calculate regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"Model": "Gradient Boosting Regressor", "MAE": mae, "RMSE": rmse, "R2 Score": r2}


def save_plots(y_true, y_pred, model: Pipeline, X: pd.DataFrame) -> None:
    """Save evaluation plots and feature importance chart."""
    residuals = y_true - y_pred

    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.xlabel("Actual Final Grade (G3)")
    plt.ylabel("Predicted Final Grade (G3)")
    plt.title("Gradient Boosting: Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gradient_boosting_regressor_actual_vs_predicted.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Final Grade (G3)")
    plt.ylabel("Residuals")
    plt.title("Gradient Boosting: Residual Plot")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gradient_boosting_regressor_residual_plot.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, kde=True)
    plt.xlabel("Prediction Error")
    plt.title("Gradient Boosting: Error Distribution")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gradient_boosting_regressor_error_distribution.png", dpi=300)
    plt.close()

    feature_names = get_feature_names(model, X)
    importances = model.named_steps["regressor"].feature_importances_
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=False)
    importance_df.to_csv(RESULT_DIR / "gradient_boosting_feature_importance.csv", index=False)

    top_features = importance_df.head(15)
    plt.figure(figsize=(8, 6))
    sns.barplot(data=top_features, x="Importance", y="Feature")
    plt.title("Gradient Boosting: Top Feature Importance")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "gradient_boosting_feature_importance.png", dpi=300)
    plt.close()


def main() -> None:
    data = load_data()

    X = data.drop(columns=["G3"])
    y = data["G3"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("regressor", GradientBoostingRegressor(random_state=42)),
        ]
    )

    param_grid = {
        "regressor__n_estimators": [100, 200],
        "regressor__learning_rate": [0.05, 0.1],
        "regressor__max_depth": [2, 3],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)

    metrics = evaluate_model(y_test, predictions)
    metrics["Best Parameters"] = str(search.best_params_)
    pd.DataFrame([metrics]).to_csv(
        RESULT_DIR / "gradient_boosting_regressor_metrics.csv", index=False
    )

    joblib.dump(best_model, MODEL_DIR / "gradient_boosting_regressor_model.pkl")
    save_plots(y_test, predictions, best_model, X)

    print("Gradient Boosting Regressor completed.")
    print(metrics)


if __name__ == "__main__":
    main()
