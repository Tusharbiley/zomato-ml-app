"""
data_loading.py
===============
Module for loading and initial inspection of the Zomato restaurant dataset.
Generates a synthetic dataset that mirrors the real Zomato dataset structure
(rating, cost, cuisines, location, online_order, book_table, votes).
"""

import pandas as pd
import numpy as np
import os

# ── reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(filepath: str | None = None) -> pd.DataFrame:
    """
    Load the Zomato dataset.

    Priority:
      1. Real CSV at *filepath* (if supplied and exists).
      2. Synthetic dataset (fallback for demo / CI environments).

    Parameters
    ----------
    filepath : str | None
        Optional path to a local Zomato CSV file.

    Returns
    -------
    pd.DataFrame
        Raw, unprocessed DataFrame ready for preprocessing.
    """
    if filepath and os.path.exists(filepath):
        print(f"[data_loading] Loading real dataset from: {filepath}")
        df = _load_real(filepath)
    else:
        print("[data_loading] No CSV found – generating synthetic Zomato dataset …")
        df = _generate_synthetic()

    _print_summary(df)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_real(filepath: str) -> pd.DataFrame:
    """Read a CSV that follows the standard Zomato Kaggle schema."""
    df = pd.read_csv(filepath, encoding="latin-1")

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Rename real Zomato column variants to standard names
    rename_map = {
        "rate":                        "rating",
        "approx_cost(for_two_people)": "cost_for_two",
        "rest_type":                   "restaurant_type",
        "listed_in(type)":             "restaurant_type_alt",
        "listed_in(city)":             "city",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns},
              inplace=True)

    # Clean rating: "4.1/5", "NEW", "-" -> numeric
    if "rating" in df.columns:
        df["rating"] = (
            df["rating"].astype(str)
            .str.replace("/5", "", regex=False)
            .str.strip()
            .replace({"NEW": None, "-": None, "nan": None, "": None})
        )
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Clean cost_for_two: "1,200" -> 1200
    if "cost_for_two" in df.columns:
        df["cost_for_two"] = (
            df["cost_for_two"].astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace({"nan": None, "": None})
        )
        df["cost_for_two"] = pd.to_numeric(df["cost_for_two"], errors="coerce")

    # Drop columns not useful for modelling
    drop_cols = ["url", "address", "phone", "dish_liked",
                 "reviews_list", "menu_item", "restaurant_type_alt"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    return df


def _generate_synthetic(n: int = 1500) -> pd.DataFrame:
    """
    Create a realistic synthetic restaurant dataset with ~1 500 rows.

    Columns
    -------
    name            : restaurant name
    city            : city (Bangalore, Mumbai, Delhi, …)
    location        : neighbourhood
    cuisines        : primary cuisine type
    cost_for_two    : average meal cost for two people (INR)
    online_order    : whether online ordering is available
    book_table      : whether table booking is available
    votes           : number of user votes
    rating          : aggregate rating (1.0 – 5.0)
    restaurant_type : QSR / Casual Dining / Fine Dining / Café / Buffet
    """
    cities = ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Pune", "Kolkata"]
    city_weights = [0.30, 0.20, 0.18, 0.12, 0.10, 0.06, 0.04]

    neighbourhoods = {
        "Bangalore": ["Koramangala", "Indiranagar", "Whitefield", "BTM Layout", "HSR Layout"],
        "Mumbai":    ["Bandra", "Andheri", "Juhu", "Powai", "Lower Parel"],
        "Delhi":     ["Connaught Place", "Hauz Khas", "Lajpat Nagar", "Saket", "Karol Bagh"],
        "Hyderabad": ["Banjara Hills", "Jubilee Hills", "Hitech City", "Gachibowli", "Madhapur"],
        "Chennai":   ["T Nagar", "Anna Nagar", "Adyar", "Velachery", "Nungambakkam"],
        "Pune":      ["Koregaon Park", "Viman Nagar", "Aundh", "Kothrud", "Baner"],
        "Kolkata":   ["Park Street", "Salt Lake", "New Town", "Gariahat", "Ballygunge"],
    }

    cuisines = [
        "North Indian", "South Indian", "Chinese", "Continental",
        "Italian", "Mughlai", "Street Food", "Fast Food",
        "Biryani", "Cafe", "Desserts", "Pizza",
    ]

    rest_types = ["Quick Bites", "Casual Dining", "Fine Dining", "Café", "Buffet", "Delivery"]

    # Sample cities
    city_col = np.random.choice(cities, size=n, p=city_weights)

    # Neighbourhood follows city
    location_col = np.array([
        np.random.choice(neighbourhoods[c]) for c in city_col
    ])

    # Cost for two: log-normal distribution centred ~₹600
    base_cost = np.random.lognormal(mean=6.4, sigma=0.6, size=n).astype(int)
    base_cost = np.clip(base_cost, 100, 5000)
    # Round to nearest 50
    cost_for_two = (np.round(base_cost / 50) * 50).astype(int)

    # Votes: log-normal
    votes = np.random.lognormal(mean=4.5, sigma=1.2, size=n).astype(int)
    votes = np.clip(votes, 1, 5000)

    # Rating: correlated weakly with cost and votes
    rating_base = (
        3.2
        + 0.0002 * (cost_for_two - 600)
        + 0.0001 * votes
        + np.random.normal(0, 0.4, n)
    )
    rating = np.clip(np.round(rating_base * 2) / 2, 1.0, 5.0)   # nearest 0.5

    # Introduce 6 % missing ratings
    missing_mask = np.random.random(n) < 0.06
    rating = rating.astype(float)
    rating[missing_mask] = np.nan

    # Online order & table booking
    online_order = np.random.choice(["Yes", "No"], size=n, p=[0.62, 0.38])
    book_table   = np.random.choice(["Yes", "No"], size=n, p=[0.28, 0.72])

    # Cuisines – introduce 4 % missing
    cuisines_col = np.random.choice(cuisines, size=n)
    cuisines_col = pd.array(cuisines_col, dtype=object)
    miss_c = np.random.random(n) < 0.04
    cuisines_col[miss_c] = np.nan

    # Restaurant type
    rest_type_col = np.random.choice(rest_types, size=n,
                                     p=[0.30, 0.25, 0.10, 0.15, 0.05, 0.15])

    # Restaurant names
    prefixes = ["The", "Café", "Hotel", "Spice", "Royal", "Urban", "Green",
                "Golden", "Star", "Classic"]
    suffixes = ["Garden", "Kitchen", "House", "Corner", "Palace", "Hub",
                "Bites", "Twist", "Junction", "Lounge"]
    names = [f"{np.random.choice(prefixes)} {np.random.choice(suffixes)} #{i+1}"
             for i in range(n)]

    df = pd.DataFrame({
        "name":            names,
        "city":            city_col,
        "location":        location_col,
        "cuisines":        cuisines_col,
        "cost_for_two":    cost_for_two,
        "online_order":    online_order,
        "book_table":      book_table,
        "votes":           votes,
        "rating":          rating,
        "restaurant_type": rest_type_col,
    })

    return df


def _print_summary(df: pd.DataFrame) -> None:
    """Print a concise data summary."""
    print(f"\n{'─'*55}")
    print(f"  Dataset shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns        : {list(df.columns)}")
    null_counts = df.isnull().sum()
    null_cols   = null_counts[null_counts > 0]
    if len(null_cols):
        print(f"  Missing values : {dict(null_cols)}")
    else:
        print("  Missing values : none")
    print(f"{'─'*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_dataset()
    print(df.head())
    print(df.dtypes)
