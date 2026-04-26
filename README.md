# Student Performance Prediction

## Project Overview
This project predicts student academic performance using machine learning regression models. The target variable is `G3`, which represents the final student grade. The project uses two provided datasets: `student-mat.csv` and `student-por.csv`.

Predicting student performance is useful because it can help educators understand which factors are related to academic results, such as study time, family background, previous grades, absences, and school-related support.

## Dataset Description
The dataset contains student demographic, social, school, family, and academic information. The project combines:

- `student-mat.csv`: Math course student data
- `student-por.csv`: Portuguese course student data

A new column named `subject` is added before combining the datasets so that the model can learn subject differences.

## Machine Learning Type
This is a **Regression** problem because the target variable `G3` is numeric.

## Target Variable
- `G3`: Final grade

## Main Features
Examples of features used by the model:

- `school`
- `sex`
- `age`
- `address`
- `famsize`
- `Medu`
- `Fedu`
- `studytime`
- `failures`
- `absences`
- `G1`
- `G2`
- `subject`

## Algorithms Used
The project implements and compares:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

## Evaluation Metrics
The models are compared using:

- **RMSE**: Measures average prediction error with stronger penalty for large errors.
- **MAE**: Measures average absolute prediction error.
- **R² Score**: Measures how much variation in final grades is explained by the model.

## Results Summary
The best model from the current run is:

**Gradient Boosting Regressor**

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Gradient Boosting Regressor | 1.5410 | 0.8851 | 0.8464 |
| Random Forest Regressor | 1.6650 | 0.9386 | 0.8207 |
| Ridge Regression | 1.7584 | 1.0122 | 0.8000 |
| Linear Regression | 1.7599 | 1.0169 | 0.7997 |


## Best Model Conclusion
The best model is **Gradient Boosting Regressor** because it achieved the lowest RMSE (1.5410) and a strong R² score (0.8464). This means it made more accurate predictions compared with the other tested models.

The most important predictive features are usually `G1` and `G2`, because previous grades are strongly related to the final grade `G3`. Other factors such as failures, absences, study time, and subject also provide useful information.

## Project Structure
```text
Student-Performance-Prediction/
│
├── data/
│   ├── student-mat.csv
│   └── student-por.csv
│
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_linear_regression.ipynb
│   ├── 03_random_forest.ipynb
│   └── 04_gradient_boosting.ipynb
│
├── src/
│   ├── linear_regression.py
│   ├── random_forest.py
│   ├── gradient_boosting.py
│   ├── train_all_models.py
│   └── utils.py
│
├── outputs/
│   ├── figures/
│   ├── models/
│   └── results/
│
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run the Project

### 1. Install required libraries
```bash
pip install -r requirements.txt
```

### 2. Run notebooks
Open Jupyter Notebook or JupyterLab and run the notebooks in order:

```text
01_eda_preprocessing.ipynb
02_linear_regression.ipynb
03_random_forest.ipynb
04_gradient_boosting.ipynb
```

### 3. Run scripts
You can also run the Python scripts:

```bash
cd src
python linear_regression.py
python random_forest.py
python gradient_boosting.py
```

Or train all main models:

```bash
python train_all_models.py
```

## Team Member Algorithm Contribution
Use this section to show each member's contribution.

| Team Member | Algorithm / Responsibility |
|---|---|
| Hoeurng Phally | Linear Regression and EDA |
| Som Soknouch | Random Forest Regressor |
| Kong Sinh | Gradient Boosting Regressor and Model Comparison |

## Challenges
Some challenges in this project include:

- Combining two datasets correctly
- Encoding many categorical variables
- Choosing a suitable target variable
- Comparing simple and complex models fairly
- Avoiding overfitting during model training

## Future Improvements
Future work could include:

- Testing more algorithms such as XGBoost or Support Vector Regression
- Performing deeper feature engineering
- Comparing models with and without `G1` and `G2`
- Using cross-validation more extensively
- Building a simple web app for prediction

## Presentation Summary
- **Problem Statement:** Predict student final grade using demographic, social, family, and academic features.
- **Dataset:** Student performance datasets for Math and Portuguese courses.
- **ML Type:** Regression.
- **Target:** `G3` final grade.
- **EDA Findings:** Previous grades `G1` and `G2` are strongly related to final grade. Study time, absences, and failures also show useful patterns.
- **Methodology:** Load data, combine datasets, clean data, encode categorical features, train regression models, tune hyperparameters, and compare results.
- **Models:** Linear Regression, Ridge Regression, Random Forest, Gradient Boosting.
- **Best Model:** Gradient Boosting Regressor.
- **Challenges:** Categorical encoding, model comparison, and overfitting control.
- **Future Work:** Try more models, improve feature engineering, and create a prediction interface.

## Reference Dataset
https://archive.ics.uci.edu/dataset/320/student+performance