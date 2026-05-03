"""
main.py
=======
Full ML pipeline orchestrator for:
  "Restaurant Profitability Prediction & Customer Segmentation (Zomato Dataset)"

Usage
-----
  python main.py            # runs complete pipeline + interactive predictor
  python main.py --predict  # interactive predictor only (requires saved models)
  python main.py --skip-tune # skip slow GridSearchCV step

Pipeline
--------
  Data Loading → Preprocessing → EDA → Model Training
    → Evaluation → Clustering → Optimization → Prediction
"""

import argparse
import sys
import os
import numpy as np

# ── project modules ───────────────────────────────────────────────────────────
from data_loading  import load_dataset
from preprocessing import preprocess, preprocess_single
from eda           import run_eda
from models        import train_all, predict
from evaluation    import evaluate_all, get_best_model
from clustering    import run_clustering
from optimization  import tune_random_forest


# ─────────────────────────────────────────────────────────────────────────────
#  BANNER
# ─────────────────────────────────────────────────────────────────────────────
BANNER = r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   🍽  Restaurant Profitability Prediction & Customer Segmentation           ║
║       Machine Learning System — Zomato Dataset                              ║
║                                                                              ║
║   Modules: Data Prep · EDA · Supervised Learning · Clustering               ║
║            Model Optimization · Advanced Models · Prediction                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(skip_tune: bool = False, dataset_path: str | None = None):
    """Execute the end-to-end ML pipeline."""
    print(BANNER)

    # ── Phase 1 : Data Understanding & Preprocessing ─────────────────────────
    _section("PHASE 1 — Data Understanding & Preprocessing")
    df_raw = load_dataset(dataset_path)

    X_train, X_test, y_train, y_test, feat_names, scaler, df_clean = \
        preprocess(df_raw, target_col="rating")

    # ── Phase 1b : EDA ────────────────────────────────────────────────────────
    _section("PHASE 1b — Exploratory Data Analysis")
    eda_paths = run_eda(df_clean)
    print(f"  EDA complete – {len(eda_paths)} plots saved to plots/")

    # ── Phase 2 : Baseline & Advanced Model Training ─────────────────────────
    _section("PHASE 2 — Supervised Model Training")
    trained_models = train_all(X_train, y_train)

    # ── Phase 2b : Evaluation ─────────────────────────────────────────────────
    _section("PHASE 2b — Model Evaluation & Comparison")
    results_df = evaluate_all(trained_models, X_train, y_train, X_test, y_test)
    best_name, best_model = get_best_model(results_df, trained_models)
    print(f"  ✅ Best model (pre-tuning): {best_name}")

    # ── Phase 3 : Unsupervised Clustering ─────────────────────────────────────
    _section("PHASE 3 — Customer / Restaurant Segmentation (K-Means)")
    cluster_result = run_clustering(df_clean)
    profiles = cluster_result["cluster_profiles"]
    print(f"  Optimal clusters: {cluster_result['optimal_k']}")
    for _, row in profiles.iterrows():
        print(f"    Cluster {int(row['cluster'])}: {row['Business Label']} "
              f"(n={row['size']})")

    # ── Phase 4 : Hyperparameter Optimization ─────────────────────────────────
    tuned_model = best_model
    if not skip_tune:
        _section("PHASE 4 — Model Optimization (GridSearchCV)")
        baseline_rf = trained_models.get("Random Forest")
        tuned_model, comparison_df = tune_random_forest(
            X_train, y_train, X_test, y_test, baseline_rf
        )
        # Update best model
        trained_models["Random Forest (Tuned)"] = tuned_model
        # Final evaluation with tuned model included
        print("[main] Re-evaluating with tuned model …")
        final_results = evaluate_all(
            trained_models, X_train, y_train, X_test, y_test
        )
        best_name, best_model = get_best_model(final_results, trained_models)
        print(f"  ✅ Final best model: {best_name}")
    else:
        print("[main] Skipping GridSearchCV (--skip-tune flag set).")

    # ── Summary ───────────────────────────────────────────────────────────────
    _section("PIPELINE COMPLETE — Summary")
    print(f"  Dataset rows        : {len(df_raw):,}")
    print(f"  Features used       : {len(feat_names)}")
    print(f"  Models trained      : {len(trained_models)}")
    print(f"  Clusters found      : {cluster_result['optimal_k']}")
    print(f"  Plots saved to      : plots/")
    print(f"  Models saved to     : models/")
    print(f"\n  🏆 Best model       : {best_name}")
    print(f"  Best model Test R²  : {results_df.iloc[0]['R²']}")

    # ── Interactive prediction ────────────────────────────────────────────────
    _section("PREDICTION — Try It Yourself")
    interactive_predict(best_model, feat_names, scaler, df_clean)

    return best_model, feat_names, scaler, df_clean


# ─────────────────────────────────────────────────────────────────────────────
#  INTERACTIVE PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────

def interactive_predict(model, feat_names, scaler, df_clean):
    """
    CLI-based interactive predictor.
    The user enters restaurant features and receives a predicted rating
    with a confidence band and cluster assignment.
    """
    CITIES     = ["Bangalore", "Mumbai", "Delhi", "Hyderabad",
                  "Chennai",   "Pune",   "Kolkata"]
    CUISINES   = ["North Indian", "South Indian", "Chinese", "Continental",
                  "Italian",     "Mughlai",       "Fast Food","Biryani",
                  "Café",        "Pizza"]
    REST_TYPES = ["Quick Bites",  "Casual Dining", "Fine Dining",
                  "Café",         "Buffet",         "Delivery"]

    print("""
  This predictor estimates a restaurant's rating given its features.
  You will be prompted for each input. Type the number shown or press
  Enter to use the default value.
""")

    def _menu(prompt, options, default_idx=0):
        print(f"\n  {prompt}")
        for i, opt in enumerate(options, 1):
            marker = " ◀ default" if i - 1 == default_idx else ""
            print(f"    {i}. {opt}{marker}")
        while True:
            raw = input("  Enter number: ").strip()
            if raw == "":
                return options[default_idx]
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(options):
                    return options[idx]
            except ValueError:
                pass
            print("  ❌ Invalid. Try again.")

    def _num_input(prompt, lo, hi, default):
        print(f"\n  {prompt} [{lo}–{hi}] (default: {default})")
        while True:
            raw = input("  Enter value: ").strip()
            if raw == "":
                return default
            try:
                val = float(raw)
                if lo <= val <= hi:
                    return val
            except ValueError:
                pass
            print(f"  ❌ Enter a number between {lo} and {hi}.")

    while True:
        print("\n" + "─" * 55)
        print("  🍽  New Restaurant Prediction")
        print("─" * 55)

        city         = _menu("Select City:", CITIES, default_idx=0)
        cuisine      = _menu("Select Primary Cuisine:", CUISINES, default_idx=0)
        rest_type    = _menu("Select Restaurant Type:", REST_TYPES, default_idx=1)
        cost         = int(_num_input("Cost for Two (₹):", 100, 5000, 600))
        votes        = int(_num_input("Expected Votes:", 1, 5000, 100))
        online_order = _menu("Online Order Available?", ["Yes", "No"], 0)
        book_table   = _menu("Table Booking Available?", ["Yes", "No"], 1)

        input_dict = {
            "city":            city,
            "cuisines":        cuisine,
            "restaurant_type": rest_type,
            "cost_for_two":    cost,
            "votes":           votes,
            "online_order":    online_order,
            "book_table":      book_table,
        }

        try:
            X_new     = preprocess_single(input_dict, feat_names, scaler, df_clean)
            pred_raw  = model.predict(X_new)[0]
            pred      = float(np.clip(pred_raw, 1.0, 5.0))
            band_lo   = max(1.0, pred - 0.25)
            band_hi   = min(5.0, pred + 0.25)
            stars     = _stars(pred)
            segment   = _segment(pred, cost)

            print(f"""
  ┌─────────────────────────────────────────────────┐
  │  🎯 PREDICTION RESULT                           │
  ├─────────────────────────────────────────────────┤
  │  Predicted Rating   : {pred:.2f} / 5.00             │
  │  Confidence Band    : {band_lo:.2f} – {band_hi:.2f}            │
  │  Stars              : {stars}               │
  │  Business Segment   : {segment:<30}│
  ├─────────────────────────────────────────────────┤
  │  Input Summary                                  │
  │    City      : {city:<34}│
  │    Cuisine   : {cuisine:<34}│
  │    Type      : {rest_type:<34}│
  │    Cost/two  : ₹{cost:<33}│
  │    Votes     : {votes:<34}│
  │    Online    : {online_order:<34}│
  │    Booking   : {book_table:<34}│
  └─────────────────────────────────────────────────┘""")

            print("\n  💡 Recommendation:")
            _print_recommendations(pred, online_order, book_table, cost)

        except Exception as e:
            print(f"  ❌ Prediction error: {e}")

        again = input("\n  Predict another? (y/n): ").strip().lower()
        if again != "y":
            print("\n  Thank you for using the Zomato ML Predictor! 👋\n")
            break


def _stars(rating: float) -> str:
    full  = int(rating)
    half  = 1 if (rating - full) >= 0.25 else 0
    empty = 5 - full - half
    return "★" * full + "½" * half + "☆" * empty


def _segment(rating: float, cost: int) -> str:
    if rating >= 4.2 and cost >= 800:
        return "Premium High-Performer"
    elif rating >= 3.8 and cost < 600:
        return "Budget Star"
    elif rating < 3.5:
        return "Needs Improvement"
    else:
        return "Mid-Tier Mainstream"


def _print_recommendations(rating, online, booking, cost):
    tips = []
    if rating < 3.5:
        tips.append("Focus on food quality and service to improve rating.")
    if online == "No":
        tips.append("Enable online ordering to reach more customers.")
    if booking == "No" and cost > 800:
        tips.append("Consider adding table booking for this price tier.")
    if not tips:
        tips.append("Restaurant looks well-positioned for the market!")
    for tip in tips:
        print(f"    • {tip}")


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    width = 76
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Zomato Restaurant ML Pipeline"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Path to Zomato CSV (optional; synthetic data used if not provided)"
    )
    parser.add_argument(
        "--skip-tune", action="store_true",
        help="Skip GridSearchCV (faster run for development)"
    )
    parser.add_argument(
        "--predict", action="store_true",
        help="Run only the interactive predictor (requires saved models)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.predict:
        # Prediction-only mode
        import joblib
        from preprocessing import preprocess

        df_raw = load_dataset(args.dataset)
        _, _, _, _, feat_names, scaler, df_clean = preprocess(df_raw)

        try:
            model = joblib.load("models/random_forest_tuned.pkl")
            print("[main] Loaded tuned Random Forest.")
        except FileNotFoundError:
            model = joblib.load("models/random_forest.pkl")
            print("[main] Loaded baseline Random Forest.")

        interactive_predict(model, feat_names, scaler, df_clean)
    else:
        run_pipeline(
            skip_tune    = args.skip_tune,
            dataset_path = args.dataset,
        )
