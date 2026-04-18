# Student Performance Prediction

## Project Overview
This project builds a machine learning pipeline to predict student final performance and analyze the effect of study time, family background, and social activities on education outcomes.

## Main Goal
The goal is not only to predict grades, but also to explain how environmental and social factors may influence academic performance.

## Problem Type
**Regression**

The target variable is `G3`, the final grade, which is a numeric score from 0 to 20.

## Dataset
This project uses the **UCI Student Performance** dataset through the `ucimlrepo` package.

## Why This Dataset
The dataset is a strong fit for this topic because it includes:
- Study-related factors such as `studytime`, `failures`, and `absences`
- Family background variables such as `Medu`, `Fedu`, `guardian`, and `famrel`
- Social and environmental variables such as `goout`, `freetime`, `activities`, and `internet`

## Project Design
This repository includes two experiments:

1. **Full Prediction Model**
   - Includes previous grades `G1` and `G2`
   - Usually gives better prediction accuracy

2. **Policy-Focused Model**
   - Removes `G1` and `G2`
   - Gives a more meaningful analysis of environmental and social factors
   - Better for early intervention discussion

## Models Used
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

## Evaluation Metrics
- MAE
- RMSE
- R²

## Repository Structure
```text
student-performance-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README_data.md
├── notebooks/
│   └── student_performance_prediction.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── outputs/
│   ├── figures/
│   └── metrics/
└── docs/
    └── team_contribution.md
```

## How to Run
1. Create a virtual environment if needed
2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the notebook:

```bash
jupyter notebook notebooks/student_performance_prediction.ipynb
```
