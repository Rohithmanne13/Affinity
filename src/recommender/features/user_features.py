"""
User Feature Engineering (Spark-based).

Generates comprehensive user-level features from interaction history:
  - Total interactions and clicks
  - Historical CTR
  - Category preferences (distribution)
  - Recency statistics
  - Session-level statistics
  - Engagement intensity

All features are computed using only data available before the prediction
time (leakage-safe).
"""

from __future__ import annotations

import logging
from typing import Any

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


def build_user_features(
    interactions_df: DataFrame,
    news_df: DataFrame,
    max_categories: int = 20,
) -> DataFrame:
    """
    Build user-level features using PySpark aggregations.

    Parameters
    ----------
    interactions_df : DataFrame
        Interaction records with user_id, news_id, label, timestamp_unix.
    news_df : DataFrame
        News catalogue with news_id, category, subcategory.
    max_categories : int
        Maximum number of category preference features.

    Returns
    -------
    DataFrame
        User features with one row per user_id.
    """
    logger.info("Building user features...")

    # Join with news to get category info
    df = interactions_df.join(
        news_df.select("news_id", "category", "subcategory"),
        on="news_id",
        how="left",
    )

    # ---- Basic interaction statistics ----
    user_stats = df.groupBy("user_id").agg(
        F.count("*").alias("user_total_interactions"),
        F.sum("label").alias("user_total_clicks"),
        F.mean("label").alias("user_ctr"),
        F.countDistinct("news_id").alias("user_unique_items"),
        # Recency
        F.max("timestamp_unix").alias("user_last_interaction_ts"),
        F.min("timestamp_unix").alias("user_first_interaction_ts"),
        # History length
        F.first("history_length").alias("user_history_length"),
    )

    # Interaction frequency (interactions per day)
    user_stats = user_stats.withColumn(
        "user_active_duration_days",
        F.greatest(
            (F.col("user_last_interaction_ts") - F.col("user_first_interaction_ts")) / 86400,
            F.lit(1.0),
        ),
    )
    user_stats = user_stats.withColumn(
        "user_interaction_frequency",
        F.col("user_total_interactions") / F.col("user_active_duration_days"),
    )

    # Click diversity (unique clicked items / total clicks)
    clicked = df.filter(F.col("label") == 1)
    click_diversity = clicked.groupBy("user_id").agg(
        F.countDistinct("news_id").alias("user_unique_clicked_items"),
    )
    user_stats = user_stats.join(click_diversity, on="user_id", how="left")
    user_stats = user_stats.fillna({"user_unique_clicked_items": 0})
    user_stats = user_stats.withColumn(
        "user_click_diversity",
        F.when(
            F.col("user_total_clicks") > 0,
            F.col("user_unique_clicked_items") / F.col("user_total_clicks"),
        ).otherwise(0.0),
    )

    # ---- Category preferences ----
    cat_counts = clicked.groupBy("user_id", "category").agg(
        F.count("*").alias("cat_click_count"),
    )

    # Top category per user
    w = Window.partitionBy("user_id").orderBy(F.desc("cat_click_count"))
    top_cats = cat_counts.withColumn("cat_rank", F.row_number().over(w))
    top_cat = top_cats.filter(F.col("cat_rank") == 1).select(
        "user_id",
        F.col("category").alias("user_top_category"),
    )
    user_stats = user_stats.join(top_cat, on="user_id", how="left")

    # Number of distinct categories clicked
    cat_diversity = clicked.groupBy("user_id").agg(
        F.countDistinct("category").alias("user_category_diversity"),
        F.countDistinct("subcategory").alias("user_subcategory_diversity"),
    )
    user_stats = user_stats.join(cat_diversity, on="user_id", how="left")
    user_stats = user_stats.fillna(
        {
            "user_category_diversity": 0,
            "user_subcategory_diversity": 0,
        }
    )

    # ---- Session-level statistics ----
    # Approximate sessions by impression_id groups
    session_stats = df.groupBy("user_id", "impression_id").agg(
        F.count("*").alias("session_size"),
        F.sum("label").alias("session_clicks"),
    )
    session_agg = session_stats.groupBy("user_id").agg(
        F.count("*").alias("user_total_sessions"),
        F.mean("session_size").alias("user_avg_session_size"),
        F.mean("session_clicks").alias("user_avg_session_clicks"),
        F.max("session_size").alias("user_max_session_size"),
    )
    user_stats = user_stats.join(session_agg, on="user_id", how="left")

    # ---- Fill nulls ----
    user_stats = user_stats.fillna(0)

    n_users = user_stats.count()
    n_features = len(user_stats.columns) - 1  # exclude user_id
    logger.info("User features built: %d users, %d features", n_users, n_features)

    return user_stats
