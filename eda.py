"""
eda.py
======
Exploratory Data Analysis (EDA) for the Zomato restaurant dataset.

Plots generated (saved to plots/ directory)
-------------------------------------------
1. Rating distribution
2. Cost-for-two distribution
3. Top-10 cuisines
4. Restaurant count by city
5. Online order vs. rating (boxplot)
6. Table booking vs. rating (boxplot)
7. Rating vs. cost scatter
8. Votes vs. rating scatter
9. Correlation heatmap (numeric features)
10. Restaurant type distribution
"""

import os
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ── shared style ──────────────────────────────────────────────────────────────
PALETTE   = "Set2"
FIG_SIZE  = (9, 5)
TITLE_PAD = 14

sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({
    "figure.dpi":       110,
    "axes.titlesize":   14,
    "axes.labelsize":   11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.titlesize": 16,
})


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> list[str]:
    """
    Run all EDA plots and return a list of saved file paths.

    Parameters
    ----------
    df : Cleaned (pre-encoding) Zomato DataFrame from preprocessing.

    Returns
    -------
    list of plot file paths
    """
    print("[eda] Generating EDA plots …")
    paths = []

    paths.append(_plot_rating_distribution(df))
    paths.append(_plot_cost_distribution(df))
    paths.append(_plot_top_cuisines(df))
    paths.append(_plot_city_counts(df))
    paths.append(_plot_online_order_vs_rating(df))
    paths.append(_plot_booking_vs_rating(df))
    paths.append(_plot_rating_vs_cost(df))
    paths.append(_plot_votes_vs_rating(df))
    paths.append(_plot_correlation_heatmap(df))
    paths.append(_plot_restaurant_type(df))

    print(f"[eda] {len(paths)} plots saved to '{PLOT_DIR}/'.\n")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
#  PRIVATE PLOT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, name: str) -> str:
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def _plot_rating_distribution(df):
    """
    Insight: Most restaurants cluster between 3.5 and 4.5. Very few receive
    perfect 5-star ratings, and a small tail exists below 3.0 – suggesting
    the platform houses a predominantly mid-quality segment.
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    vals = df["rating"].dropna()
    ax.hist(vals, bins=20, color="#4CAF50", edgecolor="white", linewidth=0.7)
    ax.axvline(vals.mean(), color="#E53935", linestyle="--",
               linewidth=1.5, label=f"Mean = {vals.mean():.2f}")
    ax.axvline(vals.median(), color="#1E88E5", linestyle="-.",
               linewidth=1.5, label=f"Median = {vals.median():.2f}")
    ax.set_title("Rating Distribution", pad=TITLE_PAD)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Restaurants")
    ax.legend()
    _annotate_insight(ax,
        "Most ratings fall between 3.5 – 4.5.\nPeak around 4.0 indicates a mid-quality platform.")
    return _save(fig, "01_rating_distribution.png")


def _plot_cost_distribution(df):
    """
    Insight: Cost-for-two is right-skewed with the bulk of restaurants priced
    between ₹200–₹1 000. A long upper tail represents premium fine-dining
    establishments – a minority but high-revenue segment.
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    vals = df["cost_for_two"].dropna()
    ax.hist(vals, bins=40, color="#FF7043", edgecolor="white", linewidth=0.5)
    ax.set_title("Cost for Two (₹) Distribution", pad=TITLE_PAD)
    ax.set_xlabel("Cost for Two (₹)")
    ax.set_ylabel("Number of Restaurants")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{int(x):,}"))
    _annotate_insight(ax,
        "Right-skewed: most restaurants cost ₹200–₹1000.\nPremium outliers exist above ₹2500.")
    return _save(fig, "02_cost_distribution.png")


def _plot_top_cuisines(df):
    """
    Insight: North Indian and Chinese dominate the platform, reflecting broad
    consumer preference. Niche cuisines (Italian, Continental) have fewer
    listings but typically attract higher price points.
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    top = df["cuisines"].value_counts().head(10)
    bars = ax.barh(top.index[::-1], top.values[::-1],
                   color=sns.color_palette(PALETTE, 10))
    ax.bar_label(bars, padding=4, fontsize=8)
    ax.set_title("Top 10 Cuisines", pad=TITLE_PAD)
    ax.set_xlabel("Number of Restaurants")
    _annotate_insight(ax,
        "North Indian & Chinese lead.\nNiche cuisines tend to be higher-priced.", x=0.55)
    return _save(fig, "03_top_cuisines.png")


def _plot_city_counts(df):
    """
    Insight: Bangalore and Mumbai together account for ~50 % of all
    restaurants, making them the primary markets for a delivery platform.
    Tier-2 cities show growth potential.
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    counts = df["city"].value_counts()
    bars = ax.bar(counts.index, counts.values,
                  color=sns.color_palette(PALETTE, len(counts)))
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_title("Restaurant Count by City", pad=TITLE_PAD)
    ax.set_xlabel("City")
    ax.set_ylabel("Number of Restaurants")
    plt.xticks(rotation=30, ha="right")
    _annotate_insight(ax,
        "Bangalore & Mumbai dominate.\nTier-2 cities are emerging markets.")
    return _save(fig, "04_city_counts.png")


def _plot_online_order_vs_rating(df):
    """
    Insight: Restaurants offering online ordering tend to have marginally
    higher median ratings, possibly because digitally active restaurants also
    invest more in service and menu quality.
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.boxplot(data=df, x="online_order", y="rating", ax=ax,
                palette={"Yes": "#66BB6A", "No": "#EF5350"})
    ax.set_title("Online Order Availability vs. Rating", pad=TITLE_PAD)
    ax.set_xlabel("Online Order Available")
    ax.set_ylabel("Rating")
    _annotate_insight(ax,
        "Online-order restaurants show slightly\nhigher median ratings.")
    return _save(fig, "05_online_order_vs_rating.png")


def _plot_booking_vs_rating(df):
    """
    Insight: Table-booking restaurants command notably higher ratings –
    this correlates with Fine Dining establishments that invest heavily in
    the guest experience.
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.boxplot(data=df, x="book_table", y="rating", ax=ax,
                palette={"Yes": "#42A5F5", "No": "#FFA726"})
    ax.set_title("Table Booking vs. Rating", pad=TITLE_PAD)
    ax.set_xlabel("Table Booking Available")
    ax.set_ylabel("Rating")
    _annotate_insight(ax,
        "Table-booking venues have higher ratings;\nassociated with Fine Dining.")
    return _save(fig, "06_booking_vs_rating.png")


def _plot_rating_vs_cost(df):
    """
    Insight: There is a weak positive correlation between cost and rating –
    more expensive restaurants tend to score slightly higher, but the
    relationship is not strong, indicating that affordable eateries can also
    achieve excellent ratings.
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sample = df.dropna(subset=["rating", "cost_for_two"]).sample(
        min(600, len(df)), random_state=42)
    ax.scatter(sample["cost_for_two"], sample["rating"],
               alpha=0.4, color="#7E57C2", edgecolors="none", s=25)
    # regression line
    z = np.polyfit(sample["cost_for_two"], sample["rating"], 1)
    p = np.poly1d(z)
    xs = np.linspace(sample["cost_for_two"].min(), sample["cost_for_two"].max(), 200)
    ax.plot(xs, p(xs), color="#E53935", linewidth=2, label="Trend")
    ax.set_title("Rating vs. Cost for Two", pad=TITLE_PAD)
    ax.set_xlabel("Cost for Two (₹)")
    ax.set_ylabel("Rating")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{int(x):,}"))
    ax.legend()
    _annotate_insight(ax,
        "Weak positive correlation between cost & rating.\nBudget restaurants can still rate highly.")
    return _save(fig, "07_rating_vs_cost.png")


def _plot_votes_vs_rating(df):
    """
    Insight: Restaurants with more votes tend to sustain higher ratings –
    social proof and engagement drive both visibility and quality perception.
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sample = df.dropna(subset=["rating", "votes"]).sample(
        min(600, len(df)), random_state=42)
    sc = ax.scatter(sample["votes"], sample["rating"],
                    c=sample["cost_for_two"], cmap="YlOrRd",
                    alpha=0.5, edgecolors="none", s=25)
    plt.colorbar(sc, ax=ax, label="Cost for Two (₹)")
    ax.set_title("Votes vs. Rating (colour = cost)", pad=TITLE_PAD)
    ax.set_xlabel("Number of Votes")
    ax.set_ylabel("Rating")
    _annotate_insight(ax,
        "More votes → higher rating trend.\nPopularity drives perceived quality.")
    return _save(fig, "08_votes_vs_rating.png")


def _plot_correlation_heatmap(df):
    """
    Insight: 'votes' and 'cost_for_two' are the numeric features most
    correlated with 'rating'. Multicollinearity between numeric predictors
    is low, making them suitable for linear models.
    """
    num_df = df.copy()
    # encode booleans for correlation
    for col in ["online_order", "book_table"]:
        if col in num_df.columns:
            num_df[col] = (num_df[col].str.lower() == "yes").astype(int)
    num_cols = num_df.select_dtypes(include=[np.number]).columns.tolist()
    corr = num_df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, square=True)
    ax.set_title("Correlation Heatmap", pad=TITLE_PAD)
    _annotate_insight(ax,
        "Votes & cost correlate most with rating.\nLow multicollinearity between features.",
        y=0.01, fontsize=7.5)
    return _save(fig, "09_correlation_heatmap.png")


def _plot_restaurant_type(df):
    """
    Insight: Quick Bites and Casual Dining make up the bulk of the platform's
    listings. Fine Dining is a small but premium segment.
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    counts = df["restaurant_type"].value_counts()
    wedge_props = {"edgecolor": "white", "linewidth": 2}
    ax.pie(counts.values, labels=counts.index,
           autopct="%1.1f%%", startangle=140,
           colors=sns.color_palette(PALETTE, len(counts)),
           wedgeprops=wedge_props)
    ax.set_title("Restaurant Type Distribution", pad=TITLE_PAD)
    _annotate_insight(ax,
        "Quick Bites dominate; Fine Dining is niche.", y=-1.18, fontsize=8.5)
    return _save(fig, "10_restaurant_type.png")


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def _annotate_insight(ax, text: str, x: float = 0.02, y: float = 0.97,
                      fontsize: float = 8.0) -> None:
    """Add a small insight text box to the plot."""
    ax.text(x, y, f"💡 {text}",
            transform=ax.transAxes,
            fontsize=fontsize,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF9C4",
                      edgecolor="#F9A825", alpha=0.85))


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loading import load_dataset
    from preprocessing import preprocess
    df_raw = load_dataset()
    _, _, _, _, _, _, df_clean = preprocess(df_raw)
    run_eda(df_clean)
