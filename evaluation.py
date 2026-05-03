"""
evaluation.py
=============
Compute regression metrics, display a comparison table, and generate
model-comparison visualisation plots.

Metrics used
------------
• MAE   – Mean Absolute Error      (lower is better)
• RMSE  – Root Mean Squared Error  (lower is better)
• R²    – Coefficient of Determination (higher is better)
• Train R² vs Test R²  → detect over/underfitting
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 13})


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(
    trained_models: dict,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
) -> pd.DataFrame:
    """
    Evaluate every trained model and return a summary DataFrame.

    Parameters
    ----------
    trained_models : dict {name: fitted_estimator}
    X_train, y_train, X_test, y_test : numpy arrays

    Returns
    -------
    pd.DataFrame  with columns [Model, MAE, RMSE, R², Train R², Test R², Overfit?]
    """
    print("[evaluation] Computing metrics …")
    records = []

    for name, model in trained_models.items():
        preds_test  = np.clip(model.predict(X_test),  1.0, 5.0)
        preds_train = np.clip(model.predict(X_train), 1.0, 5.0)

        mae      = mean_absolute_error(y_test, preds_test)
        rmse     = np.sqrt(mean_squared_error(y_test, preds_test))
        r2_test  = r2_score(y_test, preds_test)
        r2_train = r2_score(y_train, preds_train)

        overfit = "⚠ Yes" if (r2_train - r2_test) > 0.10 else "✓ No"

        records.append({
            "Model":    name,
            "MAE":      round(mae,      4),
            "RMSE":     round(rmse,     4),
            "R²":       round(r2_test,  4),
            "Train R²": round(r2_train, 4),
            "Test R²":  round(r2_test,  4),
            "Overfit?": overfit,
        })

    results_df = pd.DataFrame(records).sort_values("R²", ascending=False)
    results_df.reset_index(drop=True, inplace=True)

    _print_table(results_df)
    _plot_metric_comparison(results_df)
    _plot_overfitting_chart(results_df)
    _plot_best_model_predictions(trained_models, results_df, X_test, y_test)
    _plot_feature_importance_top(trained_models, None)  # feature names optional

    print("[evaluation] Done.\n")
    return results_df


def get_best_model(results_df: pd.DataFrame, trained_models: dict):
    """Return (name, estimator) of the model with highest Test R²."""
    best_name = results_df.iloc[0]["Model"]
    return best_name, trained_models[best_name]


# ─────────────────────────────────────────────────────────────────────────────
#  PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _print_table(df: pd.DataFrame) -> None:
    print("\n" + "═" * 75)
    print("  MODEL COMPARISON TABLE")
    print("═" * 75)
    print(df.to_string(index=False))
    print("═" * 75 + "\n")


def _plot_metric_comparison(df: pd.DataFrame) -> None:
    """Bar chart comparing MAE, RMSE, R² across models."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    palette = sns.color_palette("Set2", len(df))

    for ax, metric in zip(axes, ["MAE", "RMSE", "R²"]):
        vals   = df[metric]
        models = df["Model"]
        bars = ax.barh(models[::-1], vals[::-1], color=palette)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_title(metric)
        ax.set_xlabel(metric)
        if metric == "R²":
            ax.set_xlim(0, 1.0)

    fig.suptitle("Model Performance Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "11_model_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_overfitting_chart(df: pd.DataFrame) -> None:
    """Side-by-side Train R² vs Test R² to detect overfitting."""
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, df["Train R²"], width, label="Train R²", color="#42A5F5")
    ax.bar(x + width/2, df["Test R²"],  width, label="Test R²",  color="#66BB6A")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model"], rotation=20, ha="right")
    ax.set_ylabel("R²")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.set_title("Train R² vs Test R² (Overfitting Analysis)", pad=12)
    ax.axhline(0.8, color="grey", linestyle="--", linewidth=0.8, label="0.8 threshold")
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, "12_overfitting_analysis.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_best_model_predictions(trained_models, results_df, X_test, y_test) -> None:
    """Actual vs Predicted scatter for the best model."""
    best_name = results_df.iloc[0]["Model"]
    model     = trained_models[best_name]
    preds     = np.clip(model.predict(X_test), 1.0, 5.0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, preds, alpha=0.35, color="#7E57C2", edgecolors="none", s=20)
    lo, hi = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlabel("Actual Rating")
    ax.set_ylabel("Predicted Rating")
    ax.set_title(f"Actual vs Predicted – {best_name}", pad=12)
    ax.legend()

    path = os.path.join(PLOT_DIR, "13_actual_vs_predicted.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_feature_importance_top(trained_models, feature_names) -> None:
    """Top-20 feature importances for the best tree-based model available."""
    for name in ["Random Forest", "Gradient Boosting", "Decision Tree"]:
        if name not in trained_models:
            continue
        model = trained_models[name]
        if not hasattr(model, "feature_importances_"):
            continue
        imp = model.feature_importances_

        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(imp))]

        pairs = sorted(zip(feature_names, imp), key=lambda x: x[1], reverse=True)[:20]
        feats, vals = zip(*pairs)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(feats[::-1], vals[::-1], color="#FF7043")
        ax.set_title(f"Top-20 Feature Importances ({name})", pad=12)
        ax.set_xlabel("Importance")
        plt.tight_layout()

        path = os.path.join(PLOT_DIR, "14_feature_importance.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
        break


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loading import load_dataset
    from preprocessing import preprocess
    from models import train_all

    df = load_dataset()
    X_train, X_test, y_train, y_test, feat_names, _, _ = preprocess(df)
    trained = train_all(X_train, y_train)
    results = evaluate_all(trained, X_train, y_train, X_test, y_test)
    best_name, best_model = get_best_model(results, trained)
    print(f"Best model: {best_name}")
