"""
optimization.py
===============
Hyperparameter tuning for the Random Forest Regressor using
GridSearchCV with 5-fold cross-validation.

Steps
-----
1. Define parameter grid.
2. Run GridSearchCV.
3. Compare before/after metrics.
4. Save the optimised model.
5. Plot performance comparison.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

PLOT_DIR  = "plots"
MODEL_DIR = "models"
os.makedirs(PLOT_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PARAMETER GRID
# ─────────────────────────────────────────────────────────────────────────────
PARAM_GRID = {
    "n_estimators":     [100, 200],
    "max_depth":        [8, 12, None],
    "min_samples_leaf": [3, 5, 10],
    "max_features":     ["sqrt", 0.5],
}


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    baseline_model=None,
) -> tuple:
    """
    Run GridSearchCV on Random Forest and compare with the baseline model.

    Parameters
    ----------
    X_train, y_train : Training data.
    X_test,  y_test  : Test data.
    baseline_model   : Optional pre-trained baseline RF (from models.py).

    Returns
    -------
    (best_estimator, comparison_df)
    """
    print("[optimization] Starting GridSearchCV on Random Forest …")
    print(f"  Parameter combinations to test : "
          f"{_count_combinations(PARAM_GRID)}")

    base_rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    start = time.time()
    gs = GridSearchCV(
        estimator  = base_rf,
        param_grid = PARAM_GRID,
        cv         = 5,
        scoring    = "r2",
        n_jobs     = -1,
        verbose    = 1,
        refit      = True,
    )
    gs.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"\n  GridSearchCV completed in {elapsed:.1f}s")
    print(f"  Best parameters : {gs.best_params_}")
    print(f"  Best CV R²      : {gs.best_score_:.4f}")

    best_model = gs.best_estimator_
    joblib.dump(best_model, os.path.join(MODEL_DIR, "random_forest_tuned.pkl"))
    print(f"  Tuned model saved → models/random_forest_tuned.pkl")

    # ── Compare metrics ───────────────────────────────────────────────────────
    comparison_df = _compare(baseline_model, best_model,
                              X_train, y_train, X_test, y_test)
    _print_comparison(comparison_df)
    _plot_comparison(comparison_df)
    _plot_cv_results(gs)

    print("[optimization] Done.\n")
    return best_model, comparison_df


# ─────────────────────────────────────────────────────────────────────────────
#  PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _count_combinations(grid: dict) -> int:
    count = 1
    for v in grid.values():
        count *= len(v)
    return count * 5   # × cv folds


def _metrics(model, X, y) -> dict:
    preds = np.clip(model.predict(X), 1.0, 5.0)
    return {
        "MAE":  round(mean_absolute_error(y, preds), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y, preds)), 4),
        "R²":   round(r2_score(y, preds), 4),
    }


def _compare(baseline, tuned, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    rows = []
    for label, model in [("Baseline RF", baseline), ("Tuned RF", tuned)]:
        if model is None:
            continue
        m = _metrics(model, X_test, y_test)
        m["Model"]    = label
        m["Train R²"] = round(r2_score(y_train,
                                        np.clip(model.predict(X_train), 1, 5)), 4)
        rows.append(m)
    return pd.DataFrame(rows)[["Model", "MAE", "RMSE", "R²", "Train R²"]]


def _print_comparison(df: pd.DataFrame) -> None:
    print("\n" + "─" * 55)
    print("  OPTIMIZATION COMPARISON")
    print("─" * 55)
    print(df.to_string(index=False))
    print("─" * 55 + "\n")


def _plot_comparison(df: pd.DataFrame) -> None:
    """Side-by-side bars: Baseline RF vs Tuned RF."""
    metrics = ["MAE", "RMSE", "R²"]
    x = np.arange(len(metrics))
    width = 0.3

    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = ["#42A5F5", "#EF5350"]

    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width,
                      label=row["Model"], color=colors[i % 2])
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Metric Value")
    ax.set_title("Baseline RF vs Tuned RF (GridSearchCV)", pad=12)
    ax.legend()
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, "18_optimization_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_cv_results(gs: GridSearchCV) -> None:
    """
    Plot mean CV test scores for the top-20 parameter combinations.
    """
    res = pd.DataFrame(gs.cv_results_)
    res.sort_values("mean_test_score", ascending=False, inplace=True)
    top = res.head(20).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(
        top.index,
        top["mean_test_score"],
        yerr=top["std_test_score"],
        fmt="o-", color="#7E57C2", linewidth=1.5, markersize=5,
        ecolor="#BDBDBD", capsize=3,
    )
    ax.set_title("GridSearchCV: Mean CV R² for Top-20 Param Combinations", pad=12)
    ax.set_xlabel("Parameter Combination Rank")
    ax.set_ylabel("Mean CV R²")
    ax.set_ylim(max(0, top["mean_test_score"].min() - 0.05),
                min(1.0, top["mean_test_score"].max() + 0.05))
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, "19_cv_results.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loading import load_dataset
    from preprocessing import preprocess
    from models import train_all

    df = load_dataset()
    X_train, X_test, y_train, y_test, _, _, _ = preprocess(df)
    trained = train_all(X_train, y_train)
    baseline_rf = trained.get("Random Forest")
    best, cmp = tune_random_forest(X_train, y_train, X_test, y_test, baseline_rf)
    print(cmp)
