"""
User Segmentation via K-Means Clustering.

Segments users into behavioral clusters based on interaction patterns,
category preferences, and engagement metrics. Uses silhouette score
for optimal K selection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from recommender.config import ProjectConfig, get_config

logger = logging.getLogger(__name__)

# Features used for clustering (must exist in user features DataFrame)
CLUSTERING_FEATURES = [
    "user_total_interactions",
    "user_total_clicks",
    "user_ctr",
    "user_unique_items",
    "user_interaction_frequency",
    "user_click_diversity",
    "user_category_diversity",
    "user_subcategory_diversity",
    "user_total_sessions",
    "user_avg_session_size",
    "user_avg_session_clicks",
    "user_history_length",
]


@dataclass
class ClusteringResult:
    """Results from user clustering."""

    model: KMeans
    scaler: StandardScaler
    n_clusters: int
    silhouette: float
    cluster_sizes: dict[int, int]
    cluster_stats: pd.DataFrame
    labels: np.ndarray


def find_optimal_k(
    X: np.ndarray,
    min_k: int = 3,
    max_k: int = 10,
    seed: int = 42,
) -> int:
    """
    Find optimal cluster count using silhouette score.

    Parameters
    ----------
    X : np.ndarray
        Scaled feature matrix.
    min_k, max_k : int
        Range of K values to test.
    seed : int

    Returns
    -------
    int
        Optimal K.
    """
    best_k = min_k
    best_score = -1.0
    scores = {}

    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels, sample_size=min(5000, len(X)))
        scores[k] = score
        logger.info("K=%d → silhouette=%.4f", k, score)

        if score > best_score:
            best_score = score
            best_k = k

    logger.info("Optimal K=%d (silhouette=%.4f)", best_k, best_score)
    return best_k


def run_user_clustering(
    user_features_pdf: pd.DataFrame,
    config: ProjectConfig | None = None,
) -> ClusteringResult:
    """
    Run the user clustering pipeline.

    Parameters
    ----------
    user_features_pdf : pd.DataFrame
        User features (must contain user_id and CLUSTERING_FEATURES).
    config : ProjectConfig, optional

    Returns
    -------
    ClusteringResult
    """
    cfg = config or get_config()
    seg_cfg = cfg.segmentation

    logger.info("Starting user clustering on %d users...", len(user_features_pdf))

    # Select clustering features
    available = [f for f in CLUSTERING_FEATURES if f in user_features_pdf.columns]
    if len(available) < 3:
        raise ValueError(f"Too few clustering features available: {available}")

    X = user_features_pdf[available].fillna(0).values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Find optimal K
    min_k = seg_cfg.get("min_clusters", 3)
    max_k = seg_cfg.get("max_clusters", 10)
    optimal_k = find_optimal_k(X_scaled, min_k, max_k, cfg.seed)

    # Train final model
    kmeans = KMeans(n_clusters=optimal_k, random_state=cfg.seed, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels, sample_size=min(5000, len(X_scaled)))

    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))

    # Cluster analysis
    user_features_pdf = user_features_pdf.copy()
    user_features_pdf["user_cluster"] = labels

    cluster_stats = user_features_pdf.groupby("user_cluster")[available].mean()

    for cluster_id, size in cluster_sizes.items():
        logger.info("Cluster %d: %d users (%.1f%%)", cluster_id, size, 100 * size / len(labels))

    result = ClusteringResult(
        model=kmeans,
        scaler=scaler,
        n_clusters=optimal_k,
        silhouette=sil_score,
        cluster_sizes=cluster_sizes,
        cluster_stats=cluster_stats,
        labels=labels,
    )

    logger.info(
        "Clustering complete: %d clusters, silhouette=%.4f",
        optimal_k,
        sil_score,
    )
    return result


def save_clustering_artifacts(
    result: ClusteringResult,
    config: ProjectConfig | None = None,
) -> None:
    """Save clustering model, scaler, and analysis."""
    cfg = config or get_config()
    out_dir = cfg.artifacts_dir / "clustering"
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(result.model, out_dir / "kmeans_model.joblib")
    joblib.dump(result.scaler, out_dir / "scaler.joblib")
    result.cluster_stats.to_csv(out_dir / "cluster_stats.csv")

    meta = {
        "n_clusters": result.n_clusters,
        "silhouette": result.silhouette,
        "cluster_sizes": result.cluster_sizes,
    }
    import json

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Clustering artifacts saved to %s", out_dir)


def load_clustering_artifacts(
    config: ProjectConfig | None = None,
) -> tuple[KMeans, StandardScaler]:
    """Load saved clustering model and scaler."""
    cfg = config or get_config()
    out_dir = cfg.artifacts_dir / "clustering"
    model = joblib.load(out_dir / "kmeans_model.joblib")
    scaler = joblib.load(out_dir / "scaler.joblib")
    return model, scaler
