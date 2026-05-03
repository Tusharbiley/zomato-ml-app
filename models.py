"""
models.py
=========
Train multiple regression models to predict restaurant rating.

Models
------
1. Linear Regression       – interpretable baseline
2. Decision Tree Regressor – non-linear, fast
3. Random Forest Regressor – ensemble, primary model
4. Gradient Boosting       – boosted ensemble, advanced model
"""

import os
import numpy as np
import joblib
from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(alpha=1.0),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=8,
        min_samples_leaf=10,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=42
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def train_all(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    Train every model in MODELS and persist each to disk.

    Parameters
    ----------
    X_train : Scaled training feature matrix.
    y_train : Target values (rating).

    Returns
    -------
    dict  {model_name: fitted_estimator}
    """
    print("[models] Training models …")
    trained = {}

    for name, model in MODELS.items():
        print(f"  ▸ {name} …", end=" ", flush=True)
        model.fit(X_train, y_train)
        trained[name] = model

        # persist
        safe_name = name.lower().replace(" ", "_")
        path = os.path.join(MODEL_DIR, f"{safe_name}.pkl")
        joblib.dump(model, path)
        print(f"done  (saved → {path})")

    print("[models] All models trained.\n")
    return trained


def load_model(name: str):
    """Load a previously saved model by its display name."""
    safe_name = name.lower().replace(" ", "_")
    path = os.path.join(MODEL_DIR, f"{safe_name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model at '{path}'. Run train_all() first.")
    return joblib.load(path)


def predict(model, X: np.ndarray) -> np.ndarray:
    """Return raw predictions, clipped to [1.0, 5.0] rating range."""
    preds = model.predict(X)
    return np.clip(preds, 1.0, 5.0)


def get_feature_importance(model, feature_names: list) -> dict | None:
    """
    Return {feature: importance} for tree-based models.
    Returns None for linear models.
    """
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        return dict(sorted(zip(feature_names, imp),
                            key=lambda x: x[1], reverse=True))
    if hasattr(model, "coef_"):
        coef = np.abs(model.coef_)
        return dict(sorted(zip(feature_names, coef),
                            key=lambda x: x[1], reverse=True))
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loading import load_dataset
    from preprocessing import preprocess

    df = load_dataset()
    X_train, X_test, y_train, y_test, feat_names, _, _ = preprocess(df)
    trained = train_all(X_train, y_train)

    # Quick sanity check on Random Forest
    rf = trained["Random Forest"]
    preds = predict(rf, X_test[:5])
    print("Sample predictions:", preds)
    print("Sample actuals    :", y_test[:5])
