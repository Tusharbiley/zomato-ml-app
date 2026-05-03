"""
clustering.py
=============
Unsupervised learning – K-Means customer / restaurant segmentation.

Pipeline
--------
1. Select relevant numeric features for clustering.
2. Scale with StandardScaler.
3. Elbow method to determine optimal k.
4. Fit K-Means with optimal k.
5. Visualise clusters with PCA 2-D projection.
6. Profile each cluster with business interpretations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 110})

CLUSTER_COLORS = ["#E53935", "#43A047", "#1E88E5", "#FB8C00", "#8E24AA"]


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def run_clustering(df_clean: pd.DataFrame, k_range: range = range(2, 10)) -> dict:
    """
    Full clustering pipeline.

    Parameters
    ----------
    df_clean : Cleaned (pre-OHE) DataFrame from preprocessing.
    k_range  : Range of k values to test with the Elbow method.

    Returns
    -------
    dict with keys:
      'labels'          – cluster label for each row
      'kmeans'          – fitted KMeans object
      'optimal_k'       – chosen number of clusters
      'cluster_profiles'– DataFrame with per-cluster statistics
      'df_clustered'    – df_clean with 'cluster' column added
    """
    print("[clustering] Preparing data …")
    X, scaler = _prepare(df_clean)

    # Elbow + silhouette
    optimal_k = _elbow_method(X, k_range)

    # Fit final model
    print(f"[clustering] Fitting K-Means with k={optimal_k} …")
    km = KMeans(n_clusters=optimal_k, init="k-means++",
                n_init=15, random_state=42)
    labels = km.fit_predict(X)

    sil = silhouette_score(X, labels)
    print(f"  Silhouette score : {sil:.4f}")

    # Attach labels
    df_c = df_clean.copy()
    df_c["cluster"] = labels

    # Profile clusters
    profiles = _profile_clusters(df_c, optimal_k)

    # Visualise
    _plot_cluster_2d(X, labels, optimal_k)
    _plot_cluster_profiles(profiles)

    print("[clustering] Done.\n")
    return {
        "labels":           labels,
        "kmeans":           km,
        "optimal_k":        optimal_k,
        "cluster_profiles": profiles,
        "df_clustered":     df_c,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _prepare(df: pd.DataFrame):
    """Extract and scale numeric clustering features."""
    num_feats = ["cost_for_two", "votes", "rating"]
    num_feats = [f for f in num_feats if f in df.columns]

    X_raw = df[num_feats].copy()

    # Encode binary booleans
    for col in ["online_order", "book_table"]:
        if col in df.columns:
            X_raw[col] = (df[col].str.lower() == "yes").astype(int)

    X_raw.fillna(X_raw.median(), inplace=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    return X, scaler


def _elbow_method(X: np.ndarray, k_range: range) -> int:
    """Compute inertia for each k, plot elbow, return optimal k."""
    inertias    = []
    sil_scores  = []

    for k in k_range:
        km  = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        lbl = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, lbl))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(list(k_range), inertias, "bo-", linewidth=2, markersize=7)
    axes[0].set_title("Elbow Method – Inertia vs k")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(list(k_range), sil_scores, "rs-", linewidth=2, markersize=7)
    axes[1].set_title("Silhouette Score vs k")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")

    plt.suptitle("Optimal k Selection", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "15_elbow_silhouette.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # Choose k with best silhouette
    optimal_k = list(k_range)[int(np.argmax(sil_scores))]
    print(f"  Optimal k (best silhouette) : {optimal_k}")
    return optimal_k


def _plot_cluster_2d(X: np.ndarray, labels: np.ndarray, k: int) -> None:
    """2-D PCA projection coloured by cluster."""
    pca  = PCA(n_components=2, random_state=42)
    Xpca = pca.fit_transform(X)
    var  = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(k):
        mask = labels == c
        ax.scatter(Xpca[mask, 0], Xpca[mask, 1],
                   label=f"Cluster {c}",
                   color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                   alpha=0.5, edgecolors="none", s=20)

    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% variance)")
    ax.set_title("Restaurant Clusters – PCA Projection", pad=12)
    ax.legend(title="Cluster", loc="upper right")

    path = os.path.join(PLOT_DIR, "16_cluster_pca.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _profile_clusters(df_c: pd.DataFrame, k: int) -> pd.DataFrame:
    """Compute per-cluster mean statistics and assign a business label."""
    agg = {"cost_for_two": "mean", "votes": "mean", "rating": "mean"}
    agg = {col: fn for col, fn in agg.items() if col in df_c.columns}

    profiles = df_c.groupby("cluster").agg(
        **{col: (col, fn) for col, fn in agg.items()},
        size=("rating", "count")
    ).round(2).reset_index()

    # Assign human-readable business label
    labels_map = {}
    for _, row in profiles.iterrows():
        c   = int(row["cluster"])
        r   = row.get("rating", 3.5)
        cft = row.get("cost_for_two", 500)
        v   = row.get("votes", 100)

        if r >= 4.2 and cft >= 800:
            label = "🌟 Premium High-Performers"
        elif r >= 3.8 and cft < 600:
            label = "💚 Budget Stars"
        elif r < 3.5 and v > 200:
            label = "⚠ Popular but Underperforming"
        elif v < 50:
            label = "🆕 New / Low Visibility"
        else:
            label = "🍽 Mid-Tier Mainstream"

        labels_map[c] = label

    profiles["Business Label"] = profiles["cluster"].map(labels_map)
    print("\n  Cluster Profiles:")
    print(profiles.to_string(index=False))

    # Print insights
    print("\n  💡 Business Insights:")
    for _, row in profiles.iterrows():
        print(f"    Cluster {int(row['cluster'])} → {row['Business Label']}")
        if "rating" in row:
            print(f"      Avg rating: {row['rating']}, "
                  f"Avg cost: ₹{row.get('cost_for_two','N/A')}, "
                  f"Avg votes: {row.get('votes','N/A')}, "
                  f"Size: {row['size']}")

    return profiles


def _plot_cluster_profiles(profiles: pd.DataFrame) -> None:
    """Grouped bar chart of cluster mean features."""
    metrics = [m for m in ["cost_for_two", "votes", "rating"] if m in profiles.columns]
    if not metrics:
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
              for i in range(len(profiles))]

    for ax, metric in zip(axes, metrics):
        bars = ax.bar(profiles["cluster"].astype(str),
                      profiles[metric], color=colors)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        ax.set_title(f"Avg {metric.replace('_',' ').title()}")
        ax.set_xlabel("Cluster")

    fig.suptitle("Cluster Profiles", fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, "17_cluster_profiles.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_loading import load_dataset
    from preprocessing import preprocess

    df = load_dataset()
    _, _, _, _, _, _, df_clean = preprocess(df)
    result = run_clustering(df_clean)
    print("Optimal k:", result["optimal_k"])
