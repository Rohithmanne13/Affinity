"""
Relational Feature Engineering (Spark-based).

Generates user-item relational features:
  - User-category affinity
  - User-subcategory affinity
"""

from __future__ import annotations

import logging
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


def build_user_category_affinity(interactions_df: DataFrame, news_df: DataFrame) -> DataFrame:
    """Compute user-category affinity = clicks_in_cat / total_clicks."""
    clicked = interactions_df.filter(F.col("label") == 1).join(
        news_df.select("news_id", "category"),
        on="news_id",
        how="inner",
    )
    user_clicks = clicked.groupBy("user_id").agg(F.count("*").alias("_total"))
    user_cat = clicked.groupBy("user_id", "category").agg(
        F.count("*").alias("user_category_clicks"),
    )
    user_cat = user_cat.join(user_clicks, on="user_id", how="left")
    user_cat = user_cat.withColumn(
        "user_category_affinity",
        F.col("user_category_clicks") / F.col("_total"),
    ).drop("_total")
    logger.info("User-category affinity: %d entries", user_cat.count())
    return user_cat


def build_user_subcategory_affinity(interactions_df: DataFrame, news_df: DataFrame) -> DataFrame:
    """Compute user-subcategory affinity."""
    clicked = interactions_df.filter(F.col("label") == 1).join(
        news_df.select("news_id", "subcategory"),
        on="news_id",
        how="inner",
    )
    user_clicks = clicked.groupBy("user_id").agg(F.count("*").alias("_total"))
    user_subcat = clicked.groupBy("user_id", "subcategory").agg(
        F.count("*").alias("user_subcategory_clicks"),
    )
    user_subcat = user_subcat.join(user_clicks, on="user_id", how="left")
    user_subcat = user_subcat.withColumn(
        "user_subcategory_affinity",
        F.col("user_subcategory_clicks") / F.col("_total"),
    ).drop("_total")
    return user_subcat


def build_relational_features(
    interactions_df: DataFrame,
    news_df: DataFrame,
) -> DataFrame:
    """Attach user-category and user-subcategory affinity to each interaction."""
    logger.info("Building relational features...")
    df = interactions_df.join(
        news_df.select("news_id", "category", "subcategory"),
        on="news_id",
        how="left",
    )
    cat_aff = build_user_category_affinity(interactions_df, news_df)
    df = df.join(
        cat_aff.select("user_id", "category", "user_category_affinity", "user_category_clicks"),
        on=["user_id", "category"],
        how="left",
    )
    subcat_aff = build_user_subcategory_affinity(interactions_df, news_df)
    df = df.join(
        subcat_aff.select("user_id", "subcategory", "user_subcategory_affinity"),
        on=["user_id", "subcategory"],
        how="left",
    )
    df = df.fillna(
        {"user_category_affinity": 0.0, "user_category_clicks": 0, "user_subcategory_affinity": 0.0}
    )
    df = df.drop("category", "subcategory")
    logger.info("Relational features added")
    return df
