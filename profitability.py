
"""
profitability.py
================
Computes a composite Restaurant Profitability Score (0–100) from:

  1. Predicted Rating        (40% weight)  – quality signal
  2. Demand Score            (25% weight)  – votes / engagement
  3. Price Positioning Score (20% weight)  – cost vs rating efficiency
  4. Digital Presence Score  (15% weight)  – online order + table booking

The final score is bucketed into 4 tiers:
  80–100  → High Profit Potential  🟢
  60–79   → Moderate Potential     🟡
  40–59   → Low Potential          🟠
  0–39    → At Risk                🔴

Business logic is fully explainable so each sub-score can be shown
to the user with a natural-language rationale.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
W_RATING   = 0.40
W_DEMAND   = 0.25
W_PRICE    = 0.20
W_DIGITAL  = 0.15


def compute_profitability(
    predicted_rating: float,
    votes: int,
    cost_for_two: int,
    online_order: str,
    book_table: str,
    df_reference: pd.DataFrame = None,
) -> dict:
    """
    Compute a full profitability breakdown for one restaurant.

    Parameters
    ----------
    predicted_rating : float  predicted rating from ML model (1.0–5.0)
    votes            : int    number of user votes / reviews
    cost_for_two     : int    average meal cost for two (INR)
    online_order     : str    "Yes" or "No"
    book_table       : str    "Yes" or "No"
    df_reference     : pd.DataFrame  (optional) dataset used to normalise
                       votes and cost percentiles. If None, uses fixed ranges.

    Returns
    -------
    dict with keys:
        profitability_score   : 0–100 composite score
        tier                  : str  "High" / "Moderate" / "Low" / "At Risk"
        tier_color            : hex colour for UI
        tier_emoji            : emoji
        rating_score          : 0–100
        demand_score          : 0–100
        price_score           : 0–100
        digital_score         : 0–100
        rating_rationale      : str
        demand_rationale      : str
        price_rationale       : str
        digital_rationale     : str
        revenue_estimate      : str   rough monthly revenue band
        recommendations       : list[str]
    """

    # ── 1. Rating Score (0–100) ───────────────────────────────────────────────
    # 5.0 → 100,  1.0 → 0
    rating_score = round((predicted_rating - 1.0) / 4.0 * 100, 1)
    if predicted_rating >= 4.5:
        rating_rationale = "Excellent rating — top-tier restaurants retain customers and command premium."
    elif predicted_rating >= 4.0:
        rating_rationale = "Good rating — above average, drives repeat visits."
    elif predicted_rating >= 3.5:
        rating_rationale = "Average rating — acceptable but room for quality improvement."
    else:
        rating_rationale = "Below average rating — risk of customer churn and poor discoverability."

    # ── 2. Demand Score (0–100) ───────────────────────────────────────────────
    # Normalise votes against reference dataset or fixed scale
    if df_reference is not None and "votes" in df_reference.columns:
        p90 = df_reference["votes"].quantile(0.90)
        demand_raw = min(votes / max(p90, 1), 1.0)
    else:
        # Fixed: 0 → 0, 500+ → 100
        demand_raw = min(votes / 500.0, 1.0)

    demand_score = round(demand_raw * 100, 1)

    if votes >= 500:
        demand_rationale = f"High engagement ({votes:,} votes) — strong word-of-mouth, high footfall."
    elif votes >= 150:
        demand_rationale = f"Moderate engagement ({votes:,} votes) — growing customer base."
    elif votes >= 50:
        demand_rationale = f"Low engagement ({votes:,} votes) — needs more marketing & visibility."
    else:
        demand_rationale = f"Very few reviews ({votes:,}) — new or under-promoted restaurant."

    # ── 3. Price Positioning Score (0–100) ────────────────────────────────────
    # Best outcome: high rating for low/mid cost (efficiency)
    # Penalise: low rating + high cost (poor value)
    cost_norm = min(cost_for_two / 2000.0, 1.0)      # 0 (cheap) → 1 (expensive)
    value_ratio = predicted_rating / max(cost_norm * 5, 0.5)  # rating per cost-unit
    price_score = round(min(value_ratio / 5.0 * 100, 100), 1)

    if cost_for_two <= 400:
        price_rationale = f"Budget segment (₹{cost_for_two}) — high volume potential, low margins per cover."
    elif cost_for_two <= 900:
        price_rationale = f"Mid-range (₹{cost_for_two}) — optimal balance of volume and margin."
    elif cost_for_two <= 1600:
        price_rationale = f"Upper-mid (₹{cost_for_two}) — lower footfall but higher revenue per table."
    else:
        price_rationale = f"Premium segment (₹{cost_for_two}) — niche market, high margin if rating supports it."

    # ── 4. Digital Presence Score (0–100) ─────────────────────────────────────
    has_online  = str(online_order).strip().lower() == "yes"
    has_booking = str(book_table).strip().lower() == "yes"
    digital_raw = (0.65 * int(has_online)) + (0.35 * int(has_booking))
    digital_score = round(digital_raw * 100, 1)

    parts = []
    if has_online:  parts.append("online ordering ✓")
    else:           parts.append("online ordering ✗ (missing revenue channel)")
    if has_booking: parts.append("table booking ✓")
    else:           parts.append("table booking ✗")
    digital_rationale = "Digital channels: " + ", ".join(parts) + "."

    # ── Composite Score ───────────────────────────────────────────────────────
    composite = (
        W_RATING  * rating_score  +
        W_DEMAND  * demand_score  +
        W_PRICE   * price_score   +
        W_DIGITAL * digital_score
    )
    profitability_score = round(composite, 1)

    # ── Tier ──────────────────────────────────────────────────────────────────
    if profitability_score >= 80:
        tier, tier_color, tier_emoji = "High Profit Potential",  "#27ae60", "🟢"
    elif profitability_score >= 60:
        tier, tier_color, tier_emoji = "Moderate Potential",     "#f39c12", "🟡"
    elif profitability_score >= 40:
        tier, tier_color, tier_emoji = "Low Potential",          "#e67e22", "🟠"
    else:
        tier, tier_color, tier_emoji = "At Risk",                "#e74c3c", "🔴"

    # ── Revenue Estimate (rough monthly band) ─────────────────────────────────
    # Assumptions: avg covers/day based on votes proxy, avg spend = cost_for_two/2 per person
    daily_covers_est = max(5, min(votes // 10, 300))
    monthly_rev_low  = daily_covers_est * (cost_for_two * 0.8) * 22   # 22 working days
    monthly_rev_high = daily_covers_est * (cost_for_two * 1.2) * 26
    revenue_estimate = f"₹{_fmt(monthly_rev_low)} – ₹{_fmt(monthly_rev_high)} / month (estimated)"

    # ── Recommendations ───────────────────────────────────────────────────────
    recs = []
    if predicted_rating < 3.8:
        recs.append("Improve food quality & service — even +0.3 in rating can lift revenue 10–20%.")
    if not has_online:
        recs.append("Enable online ordering — adds 20–35% revenue for most restaurant types.")
    if not has_booking and cost_for_two > 700:
        recs.append("Add table booking — essential for mid/fine dining customer expectations.")
    if votes < 100:
        recs.append("Run promotions or loyalty offers to grow reviews and boost discoverability.")
    if cost_for_two > 1200 and predicted_rating < 4.0:
        recs.append("Rating is low for a premium price point — risk of poor perceived value.")
    if not recs:
        recs.append("Strong position overall — focus on consistency to maintain your advantage.")

    return {
        "profitability_score": profitability_score,
        "tier":                tier,
        "tier_color":          tier_color,
        "tier_emoji":          tier_emoji,
        "rating_score":        rating_score,
        "demand_score":        demand_score,
        "price_score":         price_score,
        "digital_score":       digital_score,
        "rating_rationale":    rating_rationale,
        "demand_rationale":    demand_rationale,
        "price_rationale":     price_rationale,
        "digital_rationale":   digital_rationale,
        "revenue_estimate":    revenue_estimate,
        "recommendations":     recs,
        "predicted_rating":    round(predicted_rating, 2),
    }


def _fmt(n: float) -> str:
    """Format large numbers: 125000 → '1.25L', 12000 → '12K'"""
    if n >= 100_000:
        return f"{n/100_000:.1f}L"
    elif n >= 1_000:
        return f"{int(n/1_000)}K"
    return str(int(n))


def profitability_label(score: float) -> str:
    if score >= 80: return "High Profit Potential"
    if score >= 60: return "Moderate Potential"
    if score >= 40: return "Low Potential"
    return "At Risk"

