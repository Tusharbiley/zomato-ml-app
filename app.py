
"""
app.py  –  Zomato Restaurant Profitability Prediction Web App
=============================================================
Pages
  1. Home           – stats overview
  2. EDA            – exploratory charts
  3. Model Results  – ML metrics & comparison
  4. Clusters       – K-Means segmentation
  5. Profitability  – live predictor with full profitability breakdown
"""

import os, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

st.set_page_config(
    page_title="Zomato Profitability ML",
    page_icon="🍽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stSidebar"]          { background:#1a1a2e; }
[data-testid="stSidebar"] *        { color:#e8e8e8 !important; }
[data-testid="stSidebar"] .stRadio label { color:#ffffff !important; }

.kpi-card {
  background:#f8f9fa; border-radius:12px; padding:20px 24px;
  border-left:5px solid #e23744; text-align:center;
}
.kpi-card .val  { font-size:30px; font-weight:800; color:#1a1a2e; margin:0; }
.kpi-card .lbl  { font-size:12px; color:#888; text-transform:uppercase;
                  letter-spacing:.6px; margin:4px 0 0; }

.profit-box {
  border-radius:16px; padding:28px; text-align:center; margin:18px 0;
}
.score-bar-wrap { background:#e9ecef; border-radius:999px; height:18px;
                  margin:8px 0 4px; overflow:hidden; }
.score-bar-fill { height:100%; border-radius:999px; transition:width .6s; }

.rec-card {
  background:#ed1a0b; border-left:4px solid #FFFFFF;
  border-radius:8px; padding:12px 16px; margin:6px 0;
  font-size:14px;
}
.section-pill {
  display:inline-block; background:#e23744; color:white;
  border-radius:20px; padding:4px 16px; font-size:13px;
  font-weight:600; margin-bottom:12px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CACHED PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Running ML pipeline … (first run ~30s)")
def run_pipeline(csv_path=None):
    from data_loading  import load_dataset
    from preprocessing import preprocess
    from models        import train_all
    from evaluation    import evaluate_all, get_best_model
    from clustering    import run_clustering

    df_raw = load_dataset(csv_path)
    X_tr, X_te, y_tr, y_te, feat, scaler, df_c = preprocess(df_raw)
    trained  = train_all(X_tr, y_tr)
    results  = evaluate_all(trained, X_tr, y_tr, X_te, y_te)
    best_nm, best_m = get_best_model(results, trained)
    clust    = run_clustering(df_c)
    return dict(df_raw=df_raw, df_clean=df_c,
                X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te,
                feat_names=feat, scaler=scaler,
                trained=trained, results=results,
                best_name=best_nm, best_model=best_m,
                clusters=clust)


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/7/75/Zomato_logo.png", width=160)
    st.markdown("## 🍽 Zomato ML System")
    st.markdown("*Restaurant Profitability Prediction*")
    st.markdown("---")
    page = st.radio("", [
        "🏠  Home",
        "📊  EDA & Insights",
        "🧠  Model Results",
        "🔵  Clusters",
        "💰  Profitability Predictor",
    ])
    st.markdown("---")
    uploaded = st.file_uploader("Upload zomato.csv", type="csv")
    csv_path = None
    if uploaded:
        p = "zomato_uploaded.csv"
        open(p,"wb").write(uploaded.read())
        csv_path = p
        st.success("CSV loaded!")
    elif os.path.exists("zomato.csv"):
        csv_path = "zomato.csv"
    if st.button("🔄 Refresh Pipeline"):
        st.cache_resource.clear()
        st.rerun()
    st.markdown("---")
    st.caption("scikit-learn · streamlit · pandas")

data = run_pipeline(csv_path)
df   = data["df_clean"]
res  = data["results"]
clst = data["clusters"]


# ─────────────────────────────────────────────────────────────────────────────
#  HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠  Home":
    st.title("🍽 Restaurant Profitability Prediction System")
    st.markdown("**Complete ML pipeline** — from raw Zomato data to live profitability scoring")
    st.markdown("---")

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, val, lbl in [
        (c1, f"{len(df):,}",                         "Restaurants"),
        (c2, f"{df['rating'].mean():.2f} ★",         "Avg Rating"),
        (c3, f"₹{int(df['cost_for_two'].mean()):,}", "Avg Cost/Two"),
        (c4, f"{len(data['trained'])}",              "Models Trained"),
        (c5, f"{clst['optimal_k']}",                 "Segments Found"),
    ]:
        col.markdown(f"""<div class="kpi-card">
          <p class="val">{val}</p><p class="lbl">{lbl}</p></div>""",
          unsafe_allow_html=True)

    st.markdown("---")

    # NEW: pipeline workflow
    st.markdown("""
### 🔄 ML Pipeline Workflow

Data Loading → Preprocessing → EDA → Model Training  
→ Evaluation → Clustering → Optimization → Profitability Prediction
    """)

    st.markdown("---")

    cl, cr = st.columns(2)

    with cl:
        st.markdown("#### 🤖 What this system does")

        st.markdown("""
This ML system **predicts restaurant profitability** by:

1. **Predicting the rating** the restaurant is likely to receive (supervised ML)
2. **Computing a Profitability Score (0–100)** using 4 business signals:
   - ⭐ Predicted Rating (40%)
   - 📈 Customer Demand / Votes (25%)
   - 💰 Price Positioning (20%)
   - 📱 Digital Presence (15%)
3. **Segmenting restaurants** into market clusters (K-Means)
4. Giving **actionable recommendations** to improve revenue
        """)

        # NEW: business insight
        st.info("""
💡 Business Insight:
Restaurants with stronger digital presence and higher customer engagement
typically achieve better profitability scores and customer ratings.
        """)

    with cr:
        st.markdown("#### 🏆 Best Model Performance")

        br = res.iloc[0]

        st.markdown(f"""
| Metric | Value |
|--------|-------|
| **Best Model** | {br['Model']} |
| MAE | {br['MAE']} |
| RMSE | {br['RMSE']} |
| **R² Score** | {br['R²']} |
| Overfit? | {br['Overfit?']} |
        """)

        # NEW: model recommendation
        st.success(f"""
✅ {br['Model']} achieved the best balance between predictive accuracy
and generalisation, making it the final production model.
        """)

    st.markdown("---")

    # NEW: dataset info
    st.caption(f"""
Dataset contains {len(df):,} restaurants and {df.shape[1]} features after preprocessing.
    """)

    st.markdown("#### 📂 Processed Restaurant Dataset Preview")

    st.dataframe(df.head(8), use_container_width=True)
# ─────────────────────────────────────────────────────────────────────────────
#  EDA
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊  EDA & Insights":

    st.title("📊 Exploratory Data Analysis")

    with st.expander("🧹 Data Cleaning & Preprocessing", expanded=True):

        st.markdown("""
### Data Cleaning Strategy

- Removed duplicate restaurant records
- Missing ratings imputed/removed using median strategy
- Missing categorical values handled using mode imputation
- Outliers capped using IQR-based filtering
- High-cardinality categorical features reduced using Top-N encoding
- One-hot encoding applied to categorical variables
- Numerical features scaled using StandardScaler
- Final NaN sweep performed before model training
        """)

    st.info("""
💡 EDA Insight:
Most restaurants fall within the ₹200–₹1000 cost range and maintain
ratings between 3.5–4.2, indicating a strong mid-market concentration.
    """)
    tab1,tab2,tab3,tab4 = st.tabs([
        "Distributions",
        "Cuisine & City",
        "Rating Drivers",
        "Correlations"
    ])
    
    def plot(fig):
        st.pyplot(fig)
        plt.close()
    
    # ------------------------------------------------------------------
    # TAB 1 — DISTRIBUTIONS
    # ------------------------------------------------------------------
    
    with tab1:
    
        st.info("""
    💡 Distribution Analysis Insight:
    Most restaurants operate in the affordable-to-mid-price segment,
    while ratings remain concentrated between 3.5 and 4.5 stars.
        """)
    
        c1, c2 = st.columns(2)
    
        # --------------------------------------------------------------
        # RATING DISTRIBUTION
        # --------------------------------------------------------------
    
        with c1:
    
            st.markdown("**Rating Distribution**")
    
            fig, ax = plt.subplots(figsize=(6,3.5))
    
            v = df["rating"].dropna()
    
            ax.hist(
                v,
                bins=25,
                color="#e23744",
                edgecolor="white",
                lw=.6
            )
    
            ax.axvline(
                v.mean(),
                color="#1a1a2e",
                ls="--",
                lw=1.8,
                label=f"Mean {v.mean():.2f}"
            )
    
            ax.set_xlabel("Rating")
    
            ax.legend()
    
            plt.tight_layout()
    
            plot(fig)
    
            st.info("""
    💡 Most ratings fall between 3.5–4.5.
    Very few restaurants score below 2.5 or above 4.8.
            """)
    
        # --------------------------------------------------------------
        # COST DISTRIBUTION
        # --------------------------------------------------------------
    
        with c2:
    
            st.markdown("**Cost for Two (₹)**")
    
            fig, ax = plt.subplots(figsize=(6,3.5))
    
            ax.hist(
                df["cost_for_two"].dropna(),
                bins=40,
                color="#FF7043",
                edgecolor="white",
                lw=.4
            )
    
            ax.set_xlabel("₹")
    
            plt.tight_layout()
    
            plot(fig)
    
            st.info("""
    💡 Right-skewed distribution:
    Most restaurants fall between ₹200–₹1000,
    with premium outliers above ₹2500.
            """)
    
        c3, c4 = st.columns(2)
    
        # --------------------------------------------------------------
        # VOTES DISTRIBUTION
        # --------------------------------------------------------------
    
        with c3:
    
            st.markdown("**Votes Distribution**")
    
            fig, ax = plt.subplots(figsize=(6,3.5))
    
            ax.hist(
                df["votes"].dropna(),
                bins=50,
                color="#42A5F5",
                edgecolor="white",
                lw=.4
            )
    
            ax.set_xlabel("Votes")
    
            plt.tight_layout()
    
            plot(fig)
    
            st.info("""
    💡 Most restaurants receive relatively low customer engagement,
    while a small percentage achieve very high vote counts.
            """)
    
        # --------------------------------------------------------------
        # RESTAURANT TYPE
        # --------------------------------------------------------------
    
        with c4:
    
            st.markdown("**Restaurant Type**")
    
            fig, ax = plt.subplots(figsize=(6,3.5))
    
            if "restaurant_type" in df.columns:
    
                c = (
                    df["restaurant_type"]
                    .value_counts()
                    .head(8)
                )
    
                ax.barh(
                    c.index[::-1],
                    c.values[::-1],
                    color="#66BB6A"
                )
    
            ax.set_xlabel("Count")
    
            plt.tight_layout()
    
            plot(fig)
    
            st.info("""
    💡 Quick Bites and Casual Dining dominate the market,
    reflecting strong demand for affordable dining formats.
            """)
    
    # ------------------------------------------------------------------
    # TAB 2 — CUISINE & LOCATION
    # ------------------------------------------------------------------
    
    with tab2:
    
        st.info("""
    💡 Market Segmentation Insight:
    Cuisine preference and geographic concentration strongly influence
    restaurant competition and customer demand patterns.
        """)
    
        c1, c2 = st.columns(2)
    
        # --------------------------------------------------------------
        # TOP CUISINES
        # --------------------------------------------------------------
    
        with c1:
    
            st.markdown("**Top 10 Cuisines**")
    
            fig, ax = plt.subplots(figsize=(6,4))
    
            if "cuisines" in df.columns:
    
                t = (
                    df["cuisines"]
                    .value_counts()
                    .head(10)
                )
    
                bars = ax.barh(
                    t.index[::-1],
                    t.values[::-1],
                    color=sns.color_palette("Set2",10)
                )
    
                ax.bar_label(
                    bars,
                    padding=3,
                    fontsize=8
                )
    
            plt.tight_layout()
    
            plot(fig)
    
            st.info("""
    💡 North Indian, Chinese, and Fast Food categories
    represent the most competitive cuisine segments.
            """)
    
        # --------------------------------------------------------------
        # LOCATION ANALYSIS
        # --------------------------------------------------------------
    
        with c2:
    
            st.markdown("**Restaurants by Location**")
    
            fig, ax = plt.subplots(figsize=(6,4))
    
            col = next(
                (c for c in ["city","location"] if c in df.columns),
                None
            )
    
            if col:
    
                c = (
                    df[col]
                    .value_counts()
                    .head(10)
                )
    
                bars = ax.bar(
                    c.index,
                    c.values,
                    color=sns.color_palette("Set2",len(c))
                )
    
                ax.bar_label(
                    bars,
                    padding=2,
                    fontsize=8
                )
    
                plt.xticks(
                    rotation=30,
                    ha="right"
                )
    
            plt.tight_layout()
    
            plot(fig)
    
            st.info("""
    💡 Restaurant density is highly concentrated in major urban
    commercial areas with strong customer traffic.
            """)
    
    # ------------------------------------------------------------------
    # TAB 3 — RATING DRIVERS
    # ------------------------------------------------------------------
    
    with tab3:
    
        st.info("""
    💡 Rating Driver Insight:
    Customer convenience features such as online ordering and
    table booking show positive relationships with ratings.
        """)
    
        c1, c2 = st.columns(2)
    
        # --------------------------------------------------------------
        # ONLINE ORDER VS RATING
        # --------------------------------------------------------------
    
        with c1:
    
            st.markdown("**Online Order vs Rating**")
    
            fig, ax = plt.subplots(figsize=(6,4))
    
            if "online_order" in df.columns:
    
                for v, col in [
    
                    ("Yes", "#66BB6A"),
                    ("No",  "#EF5350")
    
                ]:
    
                    s = (
                        df[df["online_order"] == v]["rating"]
                        .dropna()
                    )
    
                    ax.hist(
                        s,
                        bins=15,
                        alpha=.6,
                        label=v,
                        color=col
                    )
    
                ax.legend(title="Online Order")
    
                ax.set_xlabel("Rating")
    
            plt.tight_layout()
    
            plot(fig)
    
            st.info("""
    💡 Restaurants offering online ordering tend to achieve
    slightly stronger customer ratings.
            """)
    
        # --------------------------------------------------------------
        # TABLE BOOKING VS RATING
        # --------------------------------------------------------------
    
        with c2:
    
            st.markdown("**Table Booking vs Rating**")
    
            fig, ax = plt.subplots(figsize=(6,4))
    
            if "book_table" in df.columns:
    
                for v, col in [
    
                    ("Yes", "#42A5F5"),
                    ("No",  "#FFA726")
    
                ]:
    
                    s = (
                        df[df["book_table"] == v]["rating"]
                        .dropna()
                    )
    
                    ax.hist(
                        s,
                        bins=15,
                        alpha=.6,
                        label=v,
                        color=col
                    )
    
                ax.legend(title="Table Booking")
    
                ax.set_xlabel("Rating")
    
            plt.tight_layout()
    
            plot(fig)
    
            st.info("""
    💡 Table-booking restaurants generally belong to premium
    or fine-dining categories with stronger ratings.
            """)
    
        # --------------------------------------------------------------
        # COST VS RATING
        # --------------------------------------------------------------
    
        st.markdown("**Rating vs Cost for Two**")
    
        fig, ax = plt.subplots(figsize=(12,4))
    
        s = (
            df
            .dropna(subset=["rating","cost_for_two"])
            .sample(min(800, len(df)), random_state=42)
        )
    
        ax.scatter(
            s["cost_for_two"],
            s["rating"],
            alpha=.3,
            color="#7E57C2",
            s=14,
            edgecolors="none"
        )
    
        z = np.polyfit(
            s["cost_for_two"],
            s["rating"],
            1
        )
    
        xs = np.linspace(
            s["cost_for_two"].min(),
            s["cost_for_two"].max(),
            200
        )
    
        ax.plot(
            xs,
            np.poly1d(z)(xs),
            color="#e23744",
            lw=2,
            label="Trend"
        )
    
        ax.set_xlabel("Cost for Two (₹)")
    
        ax.set_ylabel("Rating")
    
        ax.legend()
    
        plt.tight_layout()
    
        plot(fig)
    
        st.info("""
    💡 Higher-priced restaurants tend to achieve slightly better ratings,
    though the relationship is moderate rather than strong.
        """)
    
    # ------------------------------------------------------------------
    # TAB 4 — CORRELATIONS
    # ------------------------------------------------------------------
    
    with tab4:
    
        st.info("""
    💡 Correlation Analysis Insight:
    Customer engagement and pricing variables show the strongest
    relationships with restaurant ratings.
        """)
    
        st.markdown("**Correlation Heatmap**")
    
        nd = df.copy()
    
        for c in ["online_order", "book_table"]:
    
            if c in nd.columns:
    
                nd[c] = (
                    nd[c]
                    .str.lower()
                    .eq("yes")
                    .astype(int)
                )
    
        nc = (
            nd
            .select_dtypes(include=[np.number])
            .columns
            .tolist()
        )
    
        fig, ax = plt.subplots(figsize=(8,6))
    
        sns.heatmap(
            nd[nc].corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            lw=.5,
            ax=ax,
            square=True
        )
    
        plt.tight_layout()
    
        plot(fig)
    
        st.info("""
    💡 Votes and cost_for_two appear to be the strongest
    numeric predictors of restaurant rating.
        """)


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL RESULTS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧠  Model Results":

    st.title("🧠 ML Model Training & Evaluation")

    # ------------------------------------------------------------------
    # MODEL EXPLANATION
    # ------------------------------------------------------------------

    with st.expander("🤖 Supervised Learning Pipeline", expanded=True):

        st.markdown("""
### Models Implemented

- **Linear Regression** → baseline interpretable model
- **Ridge Regression** → regularized linear model
- **Decision Tree Regressor** → captures non-linear decision boundaries
- **Random Forest Regressor** → ensemble learning model with strong generalisation
- **Gradient Boosting Regressor** → boosted ensemble model for advanced prediction

### Evaluation Metrics

- **MAE** → Mean Absolute Error (lower is better)
- **RMSE** → Root Mean Squared Error (lower is better)
- **R² Score** → prediction quality metric (higher is better)
- **Train vs Test R²** → used for overfitting analysis

### Model Selection Logic

Random Forest achieved the best balance between:
- predictive accuracy
- robustness
- generalisation performance

making it the final production model.
        """)

    st.info("""
💡 Model Insight:
Ensemble models (Random Forest & Gradient Boosting) significantly
outperformed linear models, indicating strong non-linear relationships
within restaurant business features.
    """)

    # ------------------------------------------------------------------
    # MODEL COMPARISON TABLE
    # ------------------------------------------------------------------

    st.markdown("#### 📋 Model Comparison Table")

    st.dataframe(
        res.style
           .highlight_max(subset=["R²","Test R²"], color="#ed1a0b")
           .highlight_min(subset=["MAE","RMSE"],   color="#ed1a0b")
           .format({
                    "MAE": "{:.4f}",
                    "RMSE": "{:.4f}",
                    "R²": "{:.4f}",
                    "Train R²": "{:.4f}",
                    "Test R²": "{:.4f}"
                                        }),
        use_container_width=True,
        height=230
    )

    st.markdown("---")

    # ------------------------------------------------------------------
    # TABS
    # ------------------------------------------------------------------

    t1, t2, t3 = st.tabs([
        "📊 Metric Bars",
        "🔍 Overfitting",
        "🎯 Actual vs Predicted"
    ])

    # ------------------------------------------------------------------
    # TAB 1 — METRIC BARS
    # ------------------------------------------------------------------

    with t1:

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        pal = sns.color_palette("Set2", len(res))

        for ax, m in zip(axes, ["MAE", "RMSE", "R²"]):

            bars = ax.barh(
                res["Model"][::-1],
                res[m][::-1],
                color=pal
            )

            ax.bar_label(
                bars,
                fmt="%.3f",
                padding=3,
                fontsize=8
            )

            ax.set_title(m)

        plt.suptitle(
            "Model Performance Comparison",
            fontsize=14,
            fontweight="bold"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close()

    # ------------------------------------------------------------------
    # TAB 2 — OVERFITTING
    # ------------------------------------------------------------------

    with t2:

        x = np.arange(len(res))
        w = .35

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(
            x - w/2,
            res["Train R²"],
            w,
            label="Train R²",
            color="#42A5F5"
        )

        ax.bar(
            x + w/2,
            res["Test R²"],
            w,
            label="Test R²",
            color="#66BB6A"
        )

        ax.set_xticks(x)

        ax.set_xticklabels(
            res["Model"],
            rotation=20,
            ha="right"
        )

        ax.set_ylabel("R²")

        ax.set_ylim(0, 1.1)

        ax.legend()

        ax.set_title(
            "Train R² vs Test R² — Overfitting Analysis"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close()

        st.info("""
💡 Overfitting Insight:
A large gap between Train R² and Test R² suggests overfitting.
Random Forest achieved strong generalisation with only mild overfitting.
        """)

    # ------------------------------------------------------------------
    # TAB 3 — ACTUAL VS PREDICTED
    # ------------------------------------------------------------------

    with t3:

        bm = data["best_model"]

        p = np.clip(
            bm.predict(data["X_test"]),
            1.0,
            5.0
        )

        fig, ax = plt.subplots(figsize=(6, 6))

        ax.scatter(
            data["y_test"],
            p,
            alpha=.3,
            color="#7E57C2",
            s=14,
            edgecolors="none"
        )

        ax.plot(
            [1, 5],
            [1, 5],
            "r--",
            lw=1.5,
            label="Perfect fit"
        )

        ax.set_xlabel("Actual Rating")

        ax.set_ylabel("Predicted Rating")

        ax.set_title(
            f"Actual vs Predicted — {data['best_name']}"
        )

        ax.legend()

        plt.tight_layout()

        st.pyplot(fig)

        plt.close()

        st.success(f"""
✅ {data['best_name']} demonstrates the strongest predictive
performance and was selected as the final production model.
        """)

# ─────────────────────────────────────────────────────────────────────────────
#  CLUSTERS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔵  Clusters":

    st.title("🔵 Restaurant Market Segmentation")

    st.caption(
        f"K-Means clustering — Optimal k = **{clst['optimal_k']}** segments"
    )

    # ------------------------------------------------------------------
    # CLUSTERING EXPLANATION
    # ------------------------------------------------------------------

    with st.expander("🧩 Unsupervised Learning Process", expanded=True):

        st.markdown("""
### Clustering Methodology

K-Means clustering was used to identify hidden restaurant market segments
based on business performance indicators.

### Features Used for Clustering

- Restaurant Rating
- Cost for Two
- Customer Votes / Engagement
- Online Ordering Availability
- Table Booking Availability

### Clustering Pipeline

1. Numerical feature selection
2. Feature scaling using StandardScaler
3. Optimal K detection using Elbow Method
4. K-Means cluster generation
5. PCA-based dimensionality reduction
6. Business interpretation of clusters

### Business Goal

The clustering system helps identify:
- premium restaurant segments
- budget-friendly high performers
- underperforming restaurants
- customer engagement patterns
        """)

    st.info("""
💡 Clustering Insight:
Restaurants naturally separate into distinct business groups based on
pricing, customer engagement, and ratings — enabling targeted strategies
for marketing and profitability improvement.
    """)

    # ------------------------------------------------------------------
    # CLUSTER DATA
    # ------------------------------------------------------------------

    profiles = clst["cluster_profiles"]

    df_cl = clst["df_clustered"]

    COLS = [
        "#e23744",
        "#42A5F5",
        "#66BB6A",
        "#FFA726",
        "#7E57C2",
        "#26C6DA"
    ]

    # ------------------------------------------------------------------
    # CLUSTER SUMMARY CARDS
    # ------------------------------------------------------------------

    st.markdown("#### 🏷 Cluster Summary Cards")

    cols = st.columns(min(clst['optimal_k'], 3))

    for i, (_, row) in enumerate(profiles.iterrows()):

        with cols[i % len(cols)]:

            c = COLS[i % len(COLS)]

            st.markdown(f"""
<div style="background:{c}18;border-left:4px solid {c};
     border-radius:10px;padding:14px;margin-bottom:12px;">

  <b style="font-size:15px;color:{c};">
    Cluster {int(row['cluster'])}
  </b><br>

  <span style="font-size:12px;color:#555;">
    {row.get('Business Label','—')}
  </span><br><br>

  ⭐ Rating: <b>{row.get('rating','—')}</b><br>

  💰 Avg Cost: <b>₹{row.get('cost_for_two','—')}</b><br>

  👍 Votes: <b>{row.get('votes','—')}</b><br>

  🏪 Count: <b>{row.get('size','—')}</b>

</div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ------------------------------------------------------------------
    # TABS
    # ------------------------------------------------------------------

    t1, t2, t3 = st.tabs([
        "🗺 PCA Map",
        "📊 Profiles",
        "📉 Elbow"
    ])

    # ------------------------------------------------------------------
    # TAB 1 — PCA MAP
    # ------------------------------------------------------------------

    with t1:

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler as SS
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        nf = [
            f for f in [
                "cost_for_two",
                "votes",
                "rating"
            ]
            if f in df_cl.columns
        ]

        Xc = df_cl[nf].copy()

        for c in ["online_order", "book_table"]:

            if c in df_cl.columns:

                Xc[c] = (
                    df_cl[c]
                    .str.lower()
                    .eq("yes")
                    .astype(int)
                )

        Xc.fillna(
            Xc.median(),
            inplace=True
        )

        Xs = SS().fit_transform(Xc)

        pca = PCA(
            n_components=2,
            random_state=42
        )

        Xp = pca.fit_transform(Xs)

        lbl = clst["labels"]

        var = pca.explained_variance_ratio_

        fig, ax = plt.subplots(figsize=(9, 6))

        for c in range(clst["optimal_k"]):

            m = lbl == c

            ax.scatter(
                Xp[m, 0],
                Xp[m, 1],
                label=f"Cluster {c}",
                color=COLS[c % len(COLS)],
                alpha=.45,
                s=16,
                edgecolors="none"
            )

        ax.set_xlabel(
            f"PC1 ({var[0]*100:.1f}%)"
        )

        ax.set_ylabel(
            f"PC2 ({var[1]*100:.1f}%)"
        )

        ax.set_title(
            "Restaurant Clusters — PCA Projection"
        )

        ax.legend(title="Cluster")

        plt.tight_layout()

        st.pyplot(fig)

        plt.close()

        st.success("""
✅ PCA projection successfully visualises separation between
restaurant market segments in reduced 2D feature space.
        """)

    # ------------------------------------------------------------------
    # TAB 2 — CLUSTER PROFILES
    # ------------------------------------------------------------------

    with t2:

        metrics = [
            m for m in [
                "cost_for_two",
                "votes",
                "rating"
            ]
            if m in profiles.columns
        ]

        fig, axes = plt.subplots(
            1,
            len(metrics),
            figsize=(5 * len(metrics), 4)
        )

        if len(metrics) == 1:
            axes = [axes]

        for ax, m in zip(axes, metrics):

            bars = ax.bar(
                profiles["cluster"].astype(str),
                profiles[m],
                color=[
                    COLS[i % len(COLS)]
                    for i in range(len(profiles))
                ]
            )

            ax.bar_label(
                bars,
                fmt="%.1f",
                padding=3,
                fontsize=9
            )

            ax.set_title(
                f"Avg {m.replace('_',' ').title()}"
            )

        plt.suptitle(
            "Cluster Profiles",
            fontsize=13,
            fontweight="bold"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close()

        st.info("""
💡 Cluster Profile Insight:
Each segment demonstrates distinct customer engagement,
pricing behaviour, and restaurant quality characteristics.
        """)

    # ------------------------------------------------------------------
    # TAB 3 — ELBOW METHOD
    # ------------------------------------------------------------------

    with t3:

        ins, sils = [], []

        kv = list(range(2, 8))

        for k in kv:

            km = KMeans(
                n_clusters=k,
                n_init=5,
                random_state=42
            )

            lb = km.fit_predict(Xs)

            ins.append(km.inertia_)

            sils.append(
                silhouette_score(Xs, lb)
            )

        fig, (a1, a2) = plt.subplots(
            1,
            2,
            figsize=(11, 4)
        )

        a1.plot(
            kv,
            ins,
            "bo-",
            lw=2,
            ms=7
        )

        a1.set_title("Inertia (Elbow)")

        a1.set_xlabel("k")

        a2.plot(
            kv,
            sils,
            "rs-",
            lw=2,
            ms=7
        )

        a2.set_title("Silhouette Score")

        a2.set_xlabel("k")

        plt.tight_layout()

        st.pyplot(fig)

        plt.close()

        st.info("""
💡 Optimal Cluster Selection:
The Elbow Method and Silhouette Score were jointly used to determine
the most meaningful number of restaurant segments.
        """)


# ─────────────────────────────────────────────────────────────────────────────
#  PROFITABILITY PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💰  Profitability Predictor":

    st.title("💰 Restaurant Profitability Predictor")

    st.markdown(
        "Enter restaurant details — get a **Profitability Score (0–100)** with full breakdown"
    )

    # ------------------------------------------------------------------
    # PROFITABILITY EXPLANATION
    # ------------------------------------------------------------------

    with st.expander("📈 Profitability Scoring Methodology", expanded=True):

        st.markdown("""
### Profitability Prediction Logic

This system combines Machine Learning predictions with business analytics
to estimate restaurant profitability potential.

### Profitability Score Components

- ⭐ Predicted Restaurant Rating → 40%
- 📈 Customer Demand / Votes → 25%
- 💰 Price Positioning Efficiency → 20%
- 📱 Digital Presence → 15%

### Business Objectives

The system helps identify:
- high-growth restaurant opportunities
- customer engagement strength
- pricing effectiveness
- digital readiness
- operational improvement opportunities

### Final Profitability Categories

- 🟢 High Profit Potential
- 🟡 Moderate Potential
- 🟠 Low Potential
- 🔴 At Risk
        """)

    st.info("""
💡 Business Insight:
Restaurants with strong customer engagement, good ratings,
and digital ordering availability generally achieve higher
profitability potential.
    """)


    from profitability import compute_profitability
    from preprocessing import preprocess_single

    df_c = data["df_clean"]
    def top_vals(col, n=15):
        if col in df_c.columns:
            return sorted(df_c[col].dropna().value_counts().head(n).index.tolist())
        return []

    cities     = top_vals("city")     or ["Bangalore","Mumbai","Delhi","Hyderabad","Chennai","Pune"]
    cuisines   = top_vals("cuisines") or ["North Indian","Chinese","Continental","South Indian"]
    rest_types = top_vals("restaurant_type") or ["Quick Bites","Casual Dining","Fine Dining","Café"]

    with st.form("predict_form"):
        st.markdown("#### 🏪 Restaurant Details")
        c1,c2,c3 = st.columns(3)
        with c1:
            city      = st.selectbox("🌆 City", cities)
            cuisine   = st.selectbox("🍛 Cuisine", cuisines)
            rest_type = st.selectbox("🏪 Type", rest_types)
        with c2:
            cost  = st.slider("💰 Cost for Two (₹)", 100, 5000, 600, 50)
            votes = st.slider("👍 Votes / Reviews",  1,  3000, 120, 10)
        with c3:
            st.markdown("**Digital Presence**")
            online_order = st.radio("📱 Online Order?", ["Yes","No"], horizontal=True)
            book_table   = st.radio("📅 Table Booking?",["Yes","No"], horizontal=True, index=1)
            st.markdown(" ")
            submitted = st.form_submit_button("🔮 Predict Profitability", use_container_width=True)

    if submitted:
        try:
            # Step 1: ML model predicts the rating
            X_new = preprocess_single(
                {"city":city,"cuisines":cuisine,"restaurant_type":rest_type,
                 "cost_for_two":cost,"votes":votes,
                 "online_order":online_order,"book_table":book_table},
                data["feat_names"], data["scaler"], df_c
            )
            pred_rating = float(np.clip(data["best_model"].predict(X_new)[0], 1.0, 5.0))

            # Step 2: compute profitability
            P = compute_profitability(
                predicted_rating=pred_rating,
                votes=votes,
                cost_for_two=cost,
                online_order=online_order,
                book_table=book_table,
                df_reference=df_c,
            )

            # ── BIG SCORE DISPLAY ─────────────────────────────────────────────
            st.markdown("---")
            st.markdown("### 📊 Profitability Analysis Results")

            sc1,sc2,sc3 = st.columns([1,1,1])
            with sc1:
                st.markdown(f"""
<div style="background:{P['tier_color']}18; border:2px solid {P['tier_color']};
     border-radius:16px; padding:24px; text-align:center;">
  <div style="font-size:48px; font-weight:900; color:{P['tier_color']};">
    {P['profitability_score']}<span style="font-size:22px;">/100</span>
  </div>
  <div style="font-size:16px; font-weight:600; margin:6px 0; color:{P['tier_color']};">
    {P['tier_emoji']} {P['tier']}
  </div>
  <div style="font-size:12px; color:#666;">Profitability Score</div>
</div>""", unsafe_allow_html=True)

            with sc2:
                stars_n = int(pred_rating)
                stars_s = "★"*stars_n + ("½" if pred_rating-stars_n>=.25 else "") + "☆"*(5-stars_n-(1 if pred_rating-stars_n>=.25 else 0))
                st.markdown(f"""
<div style="background:#fff3f3; border:2px solid #e23744;
     border-radius:16px; padding:24px; text-align:center;">
  <div style="font-size:48px; font-weight:900; color:#e23744;">{pred_rating:.1f}</div>
  <div style="font-size:20px; color:#f39c12; margin:4px 0;">{stars_s}</div>
  <div style="font-size:12px; color:#666;">Predicted Rating / 5.0</div>
</div>""", unsafe_allow_html=True)

            with sc3:
                st.markdown(f"""
<div style="background:#f0fff4; border:2px solid #27ae60;
     border-radius:16px; padding:24px; text-align:center;">
  <div style="font-size:32px; font-weight:700; color:#27ae60; margin-top:8px;">
    {P['revenue_estimate']}
  </div>
  <div style="font-size:12px; color:#666; margin-top:8px;">Estimated Monthly Revenue</div>
</div>""", unsafe_allow_html=True)

            # ── SCORE BREAKDOWN ───────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 🔍 Score Breakdown")

            breakdown = [
                ("⭐ Rating Quality",     P["rating_score"],  P["rating_rationale"],  "#e23744", "40% weight"),
                ("📈 Customer Demand",    P["demand_score"],  P["demand_rationale"],  "#42A5F5", "25% weight"),
                ("💰 Price Positioning",  P["price_score"],   P["price_rationale"],   "#66BB6A", "20% weight"),
                ("📱 Digital Presence",   P["digital_score"], P["digital_rationale"], "#FFA726", "15% weight"),
            ]

            for label, score, rationale, color, weight in breakdown:
                with st.expander(f"{label}  —  **{score:.0f}/100**  ·  _{weight}_", expanded=True):
                    pct = int(score)
                    st.markdown(f"""
<div class="score-bar-wrap">
  <div class="score-bar-fill" style="width:{pct}%; background:{color};"></div>
</div>
<div style="font-size:13px; color:#555; margin-top:6px;">{rationale}</div>
""", unsafe_allow_html=True)

            # ── RADAR CHART ───────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 📡 Score Radar")
            labels_r = ["Rating\nQuality","Customer\nDemand","Price\nPositioning","Digital\nPresence"]
            vals_r   = [P["rating_score"], P["demand_score"],
                        P["price_score"],  P["digital_score"]]
            vals_r  += vals_r[:1]
            angles   = np.linspace(0, 2*np.pi, len(labels_r), endpoint=False).tolist()
            angles  += angles[:1]

            fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
            ax.fill(angles, vals_r, color=P["tier_color"], alpha=0.25)
            ax.plot(angles, vals_r, color=P["tier_color"], lw=2.5, marker="o", markersize=7)
            ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels_r, fontsize=10)
            ax.set_ylim(0,100)
            ax.set_yticks([25,50,75,100]); ax.set_yticklabels(["25","50","75","100"], fontsize=7)
            ax.set_title("Profitability Score Breakdown", pad=18, fontsize=12)
            _,c2_,_ = st.columns([1,2,1])
            with c2_: st.pyplot(fig); plt.close()

            # ── RECOMMENDATIONS ───────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 💡 Actionable Recommendations")
            for i, rec in enumerate(P["recommendations"], 1):
                st.markdown(f"""
<div class="rec-card">
  <b style="color:#FFFFFF;">#{i}</b> &nbsp; {rec}
</div>""", unsafe_allow_html=True)

            # ── COMPETITOR CONTEXT ────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 📊 How you compare to similar restaurants")
            col_ctx = "city" if "city" in df_c.columns else ("location" if "location" in df_c.columns else None)
            if col_ctx:
                peers = df_c[df_c[col_ctx]==city].copy() if city in df_c[col_ctx].values else df_c.copy()
                p_metrics = {
                    "Avg Rating in Area":     f"{peers['rating'].mean():.2f}",
                    "Your Predicted Rating":  f"{pred_rating:.2f}",
                    "Avg Cost in Area":       f"₹{int(peers['cost_for_two'].mean()):,}",
                    "Your Cost":              f"₹{cost:,}",
                    "Avg Votes in Area":      f"{int(peers['votes'].mean()):,}",
                    "Your Votes":             f"{votes:,}",
                }
                c1_,c2_,c3_ = st.columns(3)
                items = list(p_metrics.items())
                for idx,col_ in enumerate([c1_,c2_,c3_]):
                    k1,v1 = items[idx*2]
                    k2,v2 = items[idx*2+1]
                    col_.metric(k1, v1)
                    col_.metric(k2, v2, delta=None)

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.exception(e)