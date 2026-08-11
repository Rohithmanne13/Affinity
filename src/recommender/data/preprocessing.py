"""
Data Preprocessing Pipeline.

Converts raw MIND data into training/evaluation examples using PySpark.
Handles: impression parsing, interaction extraction, time-aware splitting,
missing values, duplicates, and preprocessing statistics.

Target leakage prevention:
    - Train/val/test splits are time-aware (chronological)
    - Features are computed only from data available before the prediction time
    - No future interaction information leaks into historical features
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from recommender.config import ProjectConfig, get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preprocessing Statistics
# ---------------------------------------------------------------------------


@dataclass
class PreprocessingStats:
    """Tracks and reports preprocessing statistics."""

    raw_behaviors: int = 0
    raw_news: int = 0
    total_impressions: int = 0
    total_interactions: int = 0
    positive_interactions: int = 0
    negative_interactions: int = 0
    unique_users: int = 0
    unique_items: int = 0
    removed_malformed: int = 0
    removed_duplicates: int = 0
    removed_invalid_ids: int = 0
    train_size: int = 0
    val_size: int = 0
    test_size: int = 0
    users_with_empty_history: int = 0
    categories: int = 0

    def log_report(self) -> None:
        """Log a formatted preprocessing report."""
        logger.info("=" * 60)
        logger.info("PREPROCESSING STATISTICS")
        logger.info("=" * 60)
        logger.info("Raw behaviors:          %d", self.raw_behaviors)
        logger.info("Raw news articles:      %d", self.raw_news)
        logger.info("Total impression rows:  %d", self.total_impressions)
        logger.info("Total interactions:     %d", self.total_interactions)
        logger.info("  Positive (clicked):   %d", self.positive_interactions)
        logger.info("  Negative (not clicked):%d", self.negative_interactions)
        logger.info("Unique users:           %d", self.unique_users)
        logger.info("Unique items:           %d", self.unique_items)
        logger.info("Categories:             %d", self.categories)
        logger.info("Users w/ empty history: %d", self.users_with_empty_history)
        logger.info("Removed (malformed):    %d", self.removed_malformed)
        logger.info("Removed (duplicates):   %d", self.removed_duplicates)
        logger.info("Removed (invalid IDs):  %d", self.removed_invalid_ids)
        logger.info("Train size:             %d", self.train_size)
        logger.info("Validation size:        %d", self.val_size)
        logger.info("Test size:              %d", self.test_size)
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Impression Parsing
# ---------------------------------------------------------------------------


def parse_impressions(behaviors_df: DataFrame) -> DataFrame:
    """
    Parse the MIND impressions column into individual (news_id, label) rows.

    MIND format: "N12345-1 N67890-0 N11111-1"
    where -1 = clicked, -0 = not clicked.

    Parameters
    ----------
    behaviors_df : DataFrame
        Raw behaviors DataFrame with 'impressions' string column.

    Returns
    -------
    DataFrame
        Exploded DataFrame with columns: impression_id, user_id, timestamp,
        history, news_id, label.
    """
    # Split impressions string into array
    df = behaviors_df.withColumn(
        "impression_items",
        F.split(F.col("impressions"), " "),
    )

    # Explode into individual items
    df = df.withColumn("impression_item", F.explode("impression_items"))

    # Parse news_id and label from "NXXXXX-0/1"
    df = df.withColumn(
        "news_id",
        F.regexp_extract("impression_item", r"^(.+)-\d$", 1),
    )
    df = df.withColumn(
        "label",
        F.regexp_extract("impression_item", r"-(\d)$", 1).cast(IntegerType()),
    )

    # Drop intermediate columns
    df = df.drop("impression_items", "impression_item", "impressions")

    # Remove rows where parsing failed
    df = df.filter((F.col("news_id") != "") & F.col("label").isNotNull())

    return df


def parse_user_history(behaviors_df: DataFrame) -> DataFrame:
    """
    Parse the history column into an array of news IDs.

    Parameters
    ----------
    behaviors_df : DataFrame

    Returns
    -------
    DataFrame
        With 'history_items' column as array of news_id strings
        and 'history_length' column.
    """
    df = behaviors_df.withColumn(
        "history_items",
        F.when(
            F.col("history").isNotNull() & (F.trim(F.col("history")) != ""),
            F.split(F.col("history"), " "),
        ).otherwise(F.array()),
    )

    df = df.withColumn("history_length", F.size("history_items"))

    return df


# ---------------------------------------------------------------------------
# Time-Aware Splitting
# ---------------------------------------------------------------------------


def time_aware_split(
    interactions_df: DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """
    Split interactions chronologically to prevent data leakage.

    Uses the timestamp to assign interactions to train/val/test based
    on quantile boundaries, ensuring no future information leaks.

    Parameters
    ----------
    interactions_df : DataFrame
        Must have a 'timestamp' column.
    train_ratio, val_ratio, test_ratio : float
        Must sum to 1.0.

    Returns
    -------
    tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    # Compute timestamp quantiles for split boundaries
    train_boundary = interactions_df.approxQuantile("timestamp_unix", [train_ratio], 0.01)[0]
    val_boundary = interactions_df.approxQuantile(
        "timestamp_unix", [train_ratio + val_ratio], 0.01
    )[0]

    train_df = interactions_df.filter(F.col("timestamp_unix") <= train_boundary)
    val_df = interactions_df.filter(
        (F.col("timestamp_unix") > train_boundary) & (F.col("timestamp_unix") <= val_boundary)
    )
    test_df = interactions_df.filter(F.col("timestamp_unix") > val_boundary)

    logger.info(
        "Time-aware split: train=%d, val=%d, test=%d",
        train_df.count(),
        val_df.count(),
        test_df.count(),
    )

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Full Preprocessing Pipeline
# ---------------------------------------------------------------------------


def run_preprocessing(
    spark: SparkSession,
    config: ProjectConfig | None = None,
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame, PreprocessingStats]:
    """
    Run the complete preprocessing pipeline.

    Steps:
        1. Load raw news + behaviors from train and dev splits
        2. Parse impressions → individual interactions
        3. Parse user histories
        4. Handle missing values, duplicates, invalid IDs
        5. Time-aware train/val/test split
        6. Log statistics

    Parameters
    ----------
    spark : SparkSession
    config : ProjectConfig, optional

    Returns
    -------
    tuple
        (news_df, train_df, val_df, test_df, stats)
    """
    from recommender.data.loader import load_behaviors, load_news

    cfg = config or get_config()
    stats = PreprocessingStats()
    preproc_cfg = cfg.preprocessing
    split_cfg = cfg.splitting

    # ------------------------------------------------------------------
    # 1. Load raw data from both splits
    # ------------------------------------------------------------------
    logger.info("Loading raw data...")

    news_train = load_news(spark, "train", cfg)
    news_dev = load_news(spark, "dev", cfg)
    behaviors_train = load_behaviors(spark, "train", cfg)
    behaviors_dev = load_behaviors(spark, "dev", cfg)

    # Combine news (deduplicate across splits)
    news_df = news_train.unionByName(news_dev).dropDuplicates(["news_id"])
    stats.raw_news = news_df.count()
    stats.categories = news_df.select("category").distinct().count()

    # Combine behaviors
    behaviors_all = behaviors_train.unionByName(behaviors_dev)
    stats.raw_behaviors = behaviors_all.count()

    logger.info(
        "Combined: %d news articles, %d behavior records", stats.raw_news, stats.raw_behaviors
    )

    # ------------------------------------------------------------------
    # 2. Parse impressions into individual interactions
    # ------------------------------------------------------------------
    logger.info("Parsing impressions...")
    interactions_df = parse_impressions(behaviors_all)
    stats.total_impressions = stats.raw_behaviors

    # ------------------------------------------------------------------
    # 3. Parse user histories
    # ------------------------------------------------------------------
    logger.info("Parsing user histories...")
    interactions_df = parse_user_history(interactions_df)

    # Count users with empty history
    empty_hist = interactions_df.filter(F.col("history_length") == 0).select("user_id").distinct()
    stats.users_with_empty_history = empty_hist.count()

    # ------------------------------------------------------------------
    # 4. Handle missing values, duplicates, invalid IDs
    # ------------------------------------------------------------------
    logger.info("Cleaning data...")

    n_before = interactions_df.count()

    # Remove rows with null essential fields
    interactions_df = interactions_df.filter(
        F.col("user_id").isNotNull() & F.col("news_id").isNotNull() & F.col("label").isNotNull()
    )
    n_after_nulls = interactions_df.count()
    stats.removed_malformed = n_before - n_after_nulls

    # Remove invalid news IDs (not present in news catalogue)
    valid_news_ids = news_df.select("news_id")
    interactions_df = interactions_df.join(
        valid_news_ids,
        on="news_id",
        how="inner",
    )
    n_after_valid = interactions_df.count()
    stats.removed_invalid_ids = n_after_nulls - n_after_valid

    # Remove exact duplicates
    if preproc_cfg.get("remove_duplicates", True):
        n_before_dedup = interactions_df.count()
        interactions_df = interactions_df.dropDuplicates(["user_id", "news_id", "impression_id"])
        stats.removed_duplicates = n_before_dedup - interactions_df.count()

    # ------------------------------------------------------------------
    # 5. Compute statistics
    # ------------------------------------------------------------------
    stats.total_interactions = interactions_df.count()
    stats.positive_interactions = interactions_df.filter(F.col("label") == 1).count()
    stats.negative_interactions = interactions_df.filter(F.col("label") == 0).count()
    stats.unique_users = interactions_df.select("user_id").distinct().count()
    stats.unique_items = interactions_df.select("news_id").distinct().count()

    # ------------------------------------------------------------------
    # 6. Add timestamp_unix for splitting
    # ------------------------------------------------------------------
    interactions_df = interactions_df.withColumn(
        "timestamp_unix",
        F.unix_timestamp("timestamp"),
    )

    # Handle null timestamps (use median)
    median_ts = interactions_df.filter(F.col("timestamp_unix").isNotNull()).approxQuantile(
        "timestamp_unix", [0.5], 0.01
    )

    if median_ts:
        interactions_df = interactions_df.fillna({"timestamp_unix": median_ts[0]})

    # ------------------------------------------------------------------
    # 7. Time-aware split
    # ------------------------------------------------------------------
    logger.info("Performing time-aware split...")
    train_df, val_df, test_df = time_aware_split(
        interactions_df,
        train_ratio=split_cfg.get("train_ratio", 0.7),
        val_ratio=split_cfg.get("val_ratio", 0.15),
        test_ratio=split_cfg.get("test_ratio", 0.15),
    )

    stats.train_size = train_df.count()
    stats.val_size = val_df.count()
    stats.test_size = test_df.count()

    # ------------------------------------------------------------------
    # 8. Log statistics
    # ------------------------------------------------------------------
    stats.log_report()

    return news_df, train_df, val_df, test_df, stats


def save_processed_data(
    news_df: DataFrame,
    train_df: DataFrame,
    val_df: DataFrame,
    test_df: DataFrame,
    config: ProjectConfig | None = None,
) -> None:
    """
    Save processed DataFrames to Parquet.

    Parameters
    ----------
    news_df, train_df, val_df, test_df : DataFrame
    config : ProjectConfig, optional
    """
    cfg = config or get_config()
    output_dir = cfg.processed_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving processed data to %s", output_dir)

    news_df.write.mode("overwrite").parquet(str(output_dir / "news"))
    train_df.write.mode("overwrite").parquet(str(output_dir / "train"))
    val_df.write.mode("overwrite").parquet(str(output_dir / "val"))
    test_df.write.mode("overwrite").parquet(str(output_dir / "test"))

    logger.info("Processed data saved successfully")


def load_processed_data(
    spark: SparkSession,
    config: ProjectConfig | None = None,
) -> dict[str, DataFrame]:
    """
    Load previously saved processed data.

    Returns
    -------
    dict
        {"news": DataFrame, "train": DataFrame, "val": DataFrame, "test": DataFrame}
    """
    cfg = config or get_config()
    output_dir = cfg.processed_dir

    return {
        "news": spark.read.parquet(str(output_dir / "news")),
        "train": spark.read.parquet(str(output_dir / "train")),
        "val": spark.read.parquet(str(output_dir / "val")),
        "test": spark.read.parquet(str(output_dir / "test")),
    }
