"""Train all main models and save comparison results."""

import pandas as pd
from pathlib import Path
import subprocess
import sys

scripts = [
    "linear_regression.py",
    "random_forest.py",
    "gradient_boosting.py",
]

for script in scripts:
    print(f"Running {script}...")
    subprocess.run([sys.executable, script], cwd=Path(__file__).parent, check=True)

print("All models trained. Check outputs/results for metric files.")
