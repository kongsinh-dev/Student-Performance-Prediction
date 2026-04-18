import os
import pandas as pd


def load_student_data(subject="mat", data_dir="../data"):
    """
    Load Student Performance dataset from local CSV files.

    Parameters
    ----------
    subject : str
        'mat' for Mathematics dataset
        'por' for Portuguese dataset
    data_dir : str
        Path to the local data folder

    Returns
    -------
    pd.DataFrame
        Dataset including features and target column G3
    """
    file_map = {
        "mat": "student-mat.csv",
        "por": "student-por.csv"
    }

    if subject not in file_map:
        raise ValueError("subject must be either 'mat' or 'por'")

    file_path = os.path.join(data_dir, file_map[subject])

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}\n"
            f"Please place {file_map[subject]} inside the data/ folder."
        )

    # UCI student dataset uses semicolon separator
    df = pd.read_csv(file_path, sep=";")

    if "G3" not in df.columns:
        raise ValueError("Target column 'G3' not found in dataset.")

    return df