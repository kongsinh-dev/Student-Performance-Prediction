from ucimlrepo import fetch_ucirepo
import pandas as pd


def load_student_data():
    """
    Load the UCI Student Performance dataset.
    Returns a pandas DataFrame with features and target G3.
    """
    dataset = fetch_ucirepo(id=320)

    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    if isinstance(y, pd.DataFrame):
        if "G3" in y.columns:
            y = y["G3"]
        else:
            y = y.iloc[:, 0]

    df = X.copy()
    df["G3"] = y
    return df
