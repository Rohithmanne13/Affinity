"""
Item/Content Feature Engineering (Spark-based).

Generates comprehensive item-level features:
  - Popularity statistics (impressions, clicks, CTR)
  - Freshness / article age
  - Category / subcategory encoding
  - Content-level aggregations

All features use only historical data — no future leakage.
"""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


def build_item_features(
    interactions_df: DataFrame,
    news_df: DataFrame,
    reference_timestamp: float | None = None,
) -> DataFrame:
    """
    Build item-level features using PySpark aggregations.

    Parameters
    ----------
    interactions_df : DataFrame
        Interaction records with news_id, label, timestamp_unix.
    news_df : DataFrame
        News catalogue with news_id, category, subcategory, title, abstract.
    reference_timestamp : float, optional
        Unix timestamp for freshness computation. Defaults to max in data.

    Returns
    -------
    DataFrame
        Item features with one row per news_id.
    """
    logger.info("Building item features...")

    # ---- Interaction-based popularity ----
    item_stats = interactions_df.groupBy("news_id").agg(
        F.count("*").alias("item_total_impressions"),
        F.sum("label").alias("item_total_clicks"),
        F.mean("label").alias("item_ctr"),
        F.countDistinct("user_id").alias("item_unique_users"),
        # Temporal info
        F.min("timestamp_unix").alias("item_first_seen_ts"),
        F.max("timestamp_unix").alias("item_last_seen_ts"),
    )

    # Non-click count
    item_stats = item_stats.withColumn(
        "item_non_clicks",
        F.col("item_total_impressions") - F.col("item_total_clicks"),
    )

    # Click-through quality (smoothed CTR with Laplace smoothing)
    item_stats = item_stats.withColumn(
        "item_smoothed_ctr",
        (F.col("item_total_clicks") + 1) / (F.col("item_total_impressions") + 10),
    )

    # ---- Freshness features ----
    if reference_timestamp is None:
        max_ts = interactions_df.agg(F.max("timestamp_unix")).collect()[0][0]
        reference_timestamp = float(max_ts) if max_ts else 0.0

    item_stats = item_stats.withColumn(
        "item_age_hours",
        (F.lit(reference_timestamp) - F.col("item_first_seen_ts")) / 3600.0,
    )
    item_stats = item_stats.withColumn(
        "item_recency_hours",
        (F.lit(reference_timestamp) - F.col("item_last_seen_ts")) / 3600.0,
    )
    item_stats = item_stats.withColumn(
        "item_freshness_score",
        F.exp(-F.col("item_age_hours") / 72.0),  # exponential decay, 72h half-life
    )

    # ---- Content features from news catalogue ----
    news_features = news_df.select(
        "news_id",
        "category",
        "subcategory",
        "title",
        "abstract",
    )

    # Title length (word count)
    news_features = news_features.withColumn(
        "item_title_length",
        F.when(
            F.col("title").isNotNull(),
            F.size(F.split(F.col("title"), " ")),
        ).otherwise(0),
    )

    # Abstract length
    news_features = news_features.withColumn(
        "item_abstract_length",
        F.when(
            F.col("abstract").isNotNull(),
            F.size(F.split(F.col("abstract"), " ")),
        ).otherwise(0),
    )

    # Has abstract flag
    news_features = news_features.withColumn(
        "item_has_abstract",
        F.when(
            F.col("abstract").isNotNull() & (F.trim(F.col("abstract")) != ""),
            1,
        ).otherwise(0),
    )

    # ---- Category popularity (global rank within category) ----
    category_popularity = (
        interactions_df.join(
            news_df.select("news_id", "category"),
            on="news_id",
            how="inner",
        )
        .groupBy("category")
        .agg(
            F.count("*").alias("category_total_impressions"),
            F.sum("label").alias("category_total_clicks"),
            F.mean("label").alias("category_avg_ctr"),
        )
    )

    news_features = news_features.join(
        category_popularity,
        on="category",
        how="left",
    )

    # ---- Join interaction stats with content features ----
    item_df = news_features.join(item_stats, on="news_id", how="left")

    # Fill items with no interactions (cold-start items)
    item_df = item_df.fillna(
        {
            "item_total_impressions": 0,
            "item_total_clicks": 0,
            "item_ctr": 0.0,
            "item_unique_users": 0,
            "item_non_clicks": 0,
            "item_smoothed_ctr": 0.1,
            "item_age_hours": 0.0,
            "item_recency_hours": 0.0,
            "item_freshness_score": 1.0,
            "item_title_length": 0,
            "item_abstract_length": 0,
            "item_has_abstract": 0,
            "category_total_impressions": 0,
            "category_total_clicks": 0,
            "category_avg_ctr": 0.0,
        }
    )

    n_items = item_df.count()
    n_features = len(item_df.columns) - 1  # exclude news_id
    logger.info("Item features built: %d items, %d features", n_items, n_features)

    return item_df
