import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_df(model, X_test, y_test, n_repeats=10):
    """
    Compute permutation importance on the original feature columns.
    """
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=42,
        scoring="neg_root_mean_squared_error"
    )

    importance_df = pd.DataFrame({
        "feature": X_test.columns.tolist(),
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False)

    return importance_df
