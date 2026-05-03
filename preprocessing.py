"""
preprocessing.py
================
Data cleaning, encoding, scaling, and train-test split for the Zomato dataset.

Pipeline
--------
1.  Drop exact duplicate rows.
2.  Impute missing 'rating' with the median.
3.  Impute missing 'cuisines' / 'restaurant_type' with their mode.
4.  Impute missing numeric columns (cost_for_two, votes) with the median.
5.  Binary-encode Yes/No columns (online_order, book_table).
6.  Limit high-cardinality categoricals to top-N values before OHE
    (prevents 2000+ dummy columns from the real Zomato 'location' column).
7.  One-hot encode: city, cuisines, restaurant_type  (top-15 each max).
8.  Drop non-feature columns (name, location, any leftover object cols).
9.  Final NaN sweep on X with column-wise median imputation.
10. Scale with StandardScaler.
11. Train / test split 80 / 20.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

RANDOM_SEED = 42
MODEL_DIR   = "models"
MAX_DUMMIES = 15   # max unique values kept per categorical column before OHE
os.makedirs(MODEL_DIR, exist_ok=True)


def preprocess(df: pd.DataFrame, target_col: str = "rating"):
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names, scaler, df_clean
    """
    print("[preprocessing] Starting pipeline ...")
    df = df.copy()

    # 1. Remove duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"  Duplicates removed      : {before - len(df)}")

    # 2. Impute missing target (rating)
    rating_median  = df[target_col].median()
    missing_rating = df[target_col].isna().sum()
    df[target_col] = df[target_col].fillna(rating_median)
    print(f"  Rating NaNs filled      : {missing_rating}  (median={rating_median:.2f})")

    # 3. Impute categorical columns
    for col in ["cuisines", "restaurant_type"]:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # 4. Impute numeric columns
    for col in ["cost_for_two", "votes"]:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Keep clean copy for EDA / clustering (before encoding)
    df_clean = df.copy()

    # 5. Binary-encode Yes/No columns
    for col in ["online_order", "book_table"]:
        if col in df.columns:
            df[col] = (df[col].astype(str).str.strip().str.lower() == "yes").astype(int)

    # 6. Limit cardinality: keep only top-MAX_DUMMIES per categorical column
    for col in ["city", "cuisines", "restaurant_type"]:
        if col not in df.columns:
            continue
        top_vals = df[col].value_counts().head(MAX_DUMMIES).index.tolist()
        df[col]  = df[col].where(df[col].isin(top_vals), other="Other")
        print(f"  {col} kept top-{MAX_DUMMIES} values (of {df[col].nunique()} unique)")

    # 7. One-hot encode categorical columns
    cat_cols = [c for c in ["city", "cuisines", "restaurant_type"] if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    # 8. Drop non-feature / high-cardinality columns
    object_leftovers = [c for c in df.columns
                        if df[c].dtype == object and c != target_col]
    drop_cols = list(set(["name", "location"] + object_leftovers))
    if object_leftovers:
        print(f"  Dropping leftover object cols: {object_leftovers}")
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Safety: drop rows where target is still NaN
    n_before = len(df)
    df.dropna(subset=[target_col], inplace=True)
    if len(df) < n_before:
        print(f"  Rows dropped (NaN y)    : {n_before - len(df)}")

    # 9. Separate features / target
    X = df.drop(columns=[target_col])
    y = df[target_col].values.astype(float)

    # 9b. Final NaN sweep on X (fill remaining NaNs with column median)
    X = X.astype(float)
    nan_count = int(np.isnan(X.values).sum())
    if nan_count > 0:
        print(f"  Remaining NaNs in X     : {nan_count}  -> filling with column median")
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())

    # Convert any bool columns produced by get_dummies to int
    bool_cols = X.select_dtypes(include=bool).columns
    if len(bool_cols):
        X[bool_cols] = X[bool_cols].astype(int)

    feature_names = list(X.columns)

    # 10. Scale
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    # 11. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=RANDOM_SEED
    )

    # Hard assertion -- will crash loudly if anything slipped through
    assert not np.isnan(X_train).any(), "BUG: NaN still present in X_train!"
    assert not np.isnan(y_train).any(), "BUG: NaN still present in y_train!"

    print(f"  Feature count           : {X_train.shape[1]}")
    print(f"  Train samples           : {X_train.shape[0]:,}")
    print(f"  Test  samples           : {X_test.shape[0]:,}")
    print("[preprocessing] Done.\n")

    return X_train, X_test, y_train, y_test, feature_names, scaler, df_clean


def preprocess_single(input_dict: dict, feature_names: list,
                       scaler, df_clean: pd.DataFrame) -> np.ndarray:
    """Preprocess a single user-supplied restaurant record for prediction."""
    row = pd.DataFrame([input_dict])

    for col in ["online_order", "book_table"]:
        if col in row.columns:
            row[col] = (row[col].astype(str).str.strip().str.lower() == "yes").astype(int)

    for col in ["city", "cuisines", "restaurant_type"]:
        if col in row.columns and df_clean is not None and col in df_clean.columns:
            top_vals = df_clean[col].value_counts().head(MAX_DUMMIES).index.tolist()
            row[col] = row[col].where(row[col].isin(top_vals), other="Other")

    cat_cols = [c for c in ["city", "cuisines", "restaurant_type"] if c in row.columns]
    row = pd.get_dummies(row, columns=cat_cols)

    bool_cols = row.select_dtypes(include=bool).columns
    if len(bool_cols):
        row[bool_cols] = row[bool_cols].astype(int)

    row = row.reindex(columns=feature_names, fill_value=0)
    return scaler.transform(row)


if __name__ == "__main__":
    from data_loading import load_dataset
    df = load_dataset()
    X_train, X_test, y_train, y_test, feat_names, scaler, df_c = preprocess(df)
    print("X_train shape :", X_train.shape)
    print("Any NaN        :", np.isnan(X_train).any())
    print("Sample features:", feat_names[:10])
