import os
import matplotlib.pyplot as plt
import seaborn as sns


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_grade_distribution(df, save_path):
    plt.figure(figsize=(8, 5))
    sns.histplot(df["G3"], bins=15, kde=True)
    plt.title("Distribution of Final Grade (G3)")
    plt.xlabel("G3")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_correlation_heatmap(df, save_path):
    numeric_df = df.select_dtypes(include=["int64", "float64", "int32", "float32"])
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def boxplot_feature_vs_grade(df, feature, save_path):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=feature, y="G3", data=df)
    plt.title(f"{feature} vs G3")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_model_comparison(results_df, save_path):
    plt.figure(figsize=(8, 5))
    sns.barplot(data=results_df, x="Model", y="RMSE")
    plt.title("Model Comparison by RMSE")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_residuals(y_true, y_pred, save_path):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Grade")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_top_features(importance_df, save_path, top_n=15):
    top_df = importance_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(top_df["feature"], top_df["importance_mean"])
    plt.title(f"Top {top_n} Features by Permutation Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
