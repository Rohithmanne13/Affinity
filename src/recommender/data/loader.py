"""
MIND Dataset Loader.

Loads MIND news.tsv and behaviors.tsv into PySpark DataFrames
with proper schemas, handling malformed records and missing values.

MIND Schema Reference:
  news.tsv:       NewsID | Category | SubCategory | Title | Abstract | URL | TitleEntities | AbstractEntities
  behaviors.tsv:  ImpressionID | UserID | Time | History | Impressions
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from recommender.config import ProjectConfig, get_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

NEWS_SCHEMA = StructType(
    [
        StructField("news_id", StringType(), nullable=False),
        StructField("category", StringType(), nullable=True),
        StructField("subcategory", StringType(), nullable=True),
        StructField("title", StringType(), nullable=True),
        StructField("abstract", StringType(), nullable=True),
        StructField("url", StringType(), nullable=True),
        StructField("title_entities", StringType(), nullable=True),
        StructField("abstract_entities", StringType(), nullable=True),
    ]
)

BEHAVIORS_SCHEMA = StructType(
    [
        StructField("impression_id", StringType(), nullable=False),
        StructField("user_id", StringType(), nullable=False),
        StructField("timestamp", StringType(), nullable=True),
        StructField("history", StringType(), nullable=True),
        StructField("impressions", StringType(), nullable=False),
    ]
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_news(
    spark: SparkSession,
    split: str = "train",
    config: ProjectConfig | None = None,
) -> DataFrame:
    """
    Load news.tsv into a Spark DataFrame.

    Parameters
    ----------
    spark : SparkSession
    split : str
        "train" or "dev".
    config : ProjectConfig, optional

    Returns
    -------
    DataFrame
        News articles with columns: news_id, category, subcategory,
        title, abstract, url, title_entities, abstract_entities.
    """
    cfg = config or get_config()
    news_path = cfg.raw_dir / split / "news.tsv"

    if not news_path.exists():
        raise FileNotFoundError(
            f"News file not found: {news_path}. " "Run `python scripts/download_data.py` first."
        )

    df = spark.read.csv(
        str(news_path),
        schema=NEWS_SCHEMA,
        sep="\t",
        header=False,
        mode="DROPMALFORMED",
    )

    n_total = df.count()
    n_null_id = df.filter(F.col("news_id").isNull()).count()
    if n_null_id > 0:
        logger.warning("Dropping %d records with null news_id", n_null_id)
        df = df.filter(F.col("news_id").isNotNull())

    # Deduplicate news articles
    n_before = df.count()
    df = df.dropDuplicates(["news_id"])
    n_after = df.count()
    if n_before != n_after:
        logger.info("Deduplicated news: %d → %d", n_before, n_after)

    logger.info(
        "Loaded news [%s]: %d articles, %d categories, %d subcategories",
        split,
        n_after,
        df.select("category").distinct().count(),
        df.select("subcategory").distinct().count(),
    )

    return df


def load_behaviors(
    spark: SparkSession,
    split: str = "train",
    config: ProjectConfig | None = None,
) -> DataFrame:
    """
    Load behaviors.tsv into a Spark DataFrame.

    Parameters
    ----------
    spark : SparkSession
    split : str
        "train" or "dev".
    config : ProjectConfig, optional

    Returns
    -------
    DataFrame
        User behaviors with columns: impression_id, user_id, timestamp,
        history, impressions.
    """
    cfg = config or get_config()
    behaviors_path = cfg.raw_dir / split / "behaviors.tsv"

    if not behaviors_path.exists():
        raise FileNotFoundError(
            f"Behaviors file not found: {behaviors_path}. "
            "Run `python scripts/download_data.py` first."
        )

    df = spark.read.csv(
        str(behaviors_path),
        schema=BEHAVIORS_SCHEMA,
        sep="\t",
        header=False,
        mode="DROPMALFORMED",
    )

    # Parse timestamp
    df = df.withColumn(
        "timestamp",
        F.to_timestamp(F.col("timestamp"), "M/d/yyyy h:mm:ss a"),
    )

    # Handle null/empty impressions
    n_null = df.filter(F.col("impressions").isNull() | (F.trim(F.col("impressions")) == "")).count()
    if n_null > 0:
        logger.warning("Dropping %d records with null/empty impressions", n_null)
        df = df.filter(F.col("impressions").isNotNull() & (F.trim(F.col("impressions")) != ""))

    n_records = df.count()
    n_users = df.select("user_id").distinct().count()

    logger.info(
        "Loaded behaviors [%s]: %d impressions, %d unique users",
        split,
        n_records,
        n_users,
    )

    return df


def load_dataset(
    spark: SparkSession,
    split: str = "train",
    config: ProjectConfig | None = None,
) -> dict[str, DataFrame]:
    """
    Load both news and behaviors for a split.

    Returns
    -------
    dict
        {"news": DataFrame, "behaviors": DataFrame}
    """
    return {
        "news": load_news(spark, split, config),
        "behaviors": load_behaviors(spark, split, config),
    }
