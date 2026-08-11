"""
Feature Pipeline Orchestrator.

Runs all feature families (user, item, context, relational),
joins them, and persists the final feature matrix to parquet.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from recommender.config import ProjectConfig, get_config
from recommender.features.context_features import build_context_features
from recommender.features.item_features import build_item_features
from recommender.features.relational_features import build_relational_features
from recommender.features.user_features import build_user_features

logger = logging.getLogger(__name__)


def run_feature_pipeline(
    spark: SparkSession,
    news_df: DataFrame,
    train_df: DataFrame,
    val_df: DataFrame | None = None,
    test_df: DataFrame | None = None,
    config: ProjectConfig | None = None,
) -> dict[str, DataFrame]:
    """
    Run the full feature engineering pipeline.

    Steps:
        1. Build user features from train interactions
        2. Build item features from train interactions
        3. Add context features to all splits
        4. Add relational features to all splits
        5. Join user + item features onto each split

    Leakage prevention: user and item features are built from
    train data only. Val/test inherit those features.

    Returns
    -------
    dict with keys: "user_features", "item_features",
                    "train", "val", "test" (feature-enriched DataFrames)
    """
    cfg = config or get_config()
    start = time.time()
    logger.info("Starting feature pipeline...")

    # 1. Build user features from TRAIN only (prevent leakage)
    user_feats = build_user_features(
        train_df,
        news_df,
        max_categories=cfg.features.get("user", {}).get("max_categories", 20),
    )

    # 2. Build item features from TRAIN only
    item_feats = build_item_features(train_df, news_df)

    # 3. Process each split
    results = {"user_features": user_feats, "item_features": item_feats}

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if split_df is None:
            continue

        logger.info("Processing %s split...", split_name)

        # Context features
        df = build_context_features(split_df)

        # Relational features
        df = build_relational_features(df, news_df)

        # Join user features
        df = df.join(user_feats, on="user_id", how="left")

        # Join item features (select subset to avoid column collision)
        item_cols = [c for c in item_feats.columns if c != "news_id" and c not in df.columns]
        df = df.join(item_feats.select("news_id", *item_cols), on="news_id", how="left")

        # Fill nulls for cold-start users/items
        df = df.fillna(0)

        results[split_name] = df
        logger.info("%s split: %d rows, %d columns", split_name, df.count(), len(df.columns))

    elapsed = time.time() - start
    logger.info("Feature pipeline complete in %.1f seconds", elapsed)
    return results


def save_features(
    feature_data: dict[str, DataFrame],
    config: ProjectConfig | None = None,
) -> None:
    """Save feature DataFrames to parquet."""
    cfg = config or get_config()
    output_dir = cfg.features_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, df in feature_data.items():
        path = output_dir / name
        df.write.mode("overwrite").parquet(str(path))
        logger.info("Saved %s features to %s", name, path)


def load_features(
    spark: SparkSession,
    config: ProjectConfig | None = None,
) -> dict[str, DataFrame]:
    """Load previously saved features."""
    cfg = config or get_config()
    d = cfg.features_dir
    result = {}
    for name in ["user_features", "item_features", "train", "val", "test"]:
        p = d / name
        if Path(str(p)).exists():
            result[name] = spark.read.parquet(str(p))
    return result
