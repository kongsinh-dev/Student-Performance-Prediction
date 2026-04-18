import json
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from preprocessing import build_preprocessor


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def run_experiment(df, drop_previous_grades=False, output_dir="../outputs/metrics"):
    """
    Train and evaluate three regression models on student performance data.
    """
    os.makedirs(output_dir, exist_ok=True)

    target = "G3"
    drop_cols = [target]

    if drop_previous_grades:
        for col in ["G1", "G2"]:
            if col in df.columns:
                drop_cols.append(col)

    X = df.drop(columns=drop_cols)
    y = df[target]

    preprocessor, _, _ = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }

    param_grids = {
        "Ridge": {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 50.0]
        },
        "RandomForest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4]
        },
        "GradientBoosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.8, 1.0]
        }
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    best_models = {}

    for name, estimator in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[name],
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        preds = best_model.predict(X_test)

        metrics = {
            "Model": name,
            "Best Params": str(grid.best_params_),
            "MAE": float(mean_absolute_error(y_test, preds)),
            "RMSE": rmse(y_test, preds),
            "R2": float(r2_score(y_test, preds))
        }

        results.append(metrics)
        best_models[name] = best_model

    results_df = pd.DataFrame(results).sort_values(by="RMSE")
    suffix = "without_G1_G2" if drop_previous_grades else "with_G1_G2"

    results_df.to_csv(os.path.join(output_dir, f"regression_results_{suffix}.csv"), index=False)

    with open(os.path.join(output_dir, f"best_params_{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results_df, best_models, X_train, X_test, y_train, y_test
