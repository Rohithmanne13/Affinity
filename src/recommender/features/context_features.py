"""
Context Feature Engineering.

Generates contextual features from interaction metadata:
  - Hour of day (cyclic encoding)
  - Day of week (cyclic encoding)
  - Session position
  - Impression position within session
"""

from __future__ import annotations

import logging
import math

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


def build_context_features(interactions_df: DataFrame) -> DataFrame:
    """
    Build context features for each interaction.

    Parameters
    ----------
    interactions_df : DataFrame
        Must have: timestamp (or timestamp_unix), impression_id, user_id.

    Returns
    -------
    DataFrame
        Original DataFrame augmented with context feature columns.
    """
    logger.info("Building context features...")
    df = interactions_df

    # ---- Temporal features from timestamp ----
    if "timestamp" in df.columns:
        ts_col = "timestamp"
    else:
        # Reconstruct from unix timestamp
        df = df.withColumn("_ts", F.from_unixtime("timestamp_unix"))
        ts_col = "_ts"

    # Hour of day
    df = df.withColumn("ctx_hour", F.hour(ts_col))

    # Cyclic encoding for hour (sin/cos)
    df = df.withColumn(
        "ctx_hour_sin",
        F.sin(2 * math.pi * F.col("ctx_hour") / 24.0),
    )
    df = df.withColumn(
        "ctx_hour_cos",
        F.cos(2 * math.pi * F.col("ctx_hour") / 24.0),
    )

    # Day of week (0=Monday, 6=Sunday)
    df = df.withColumn("ctx_day_of_week", F.dayofweek(ts_col) - 1)

    # Cyclic encoding for day of week
    df = df.withColumn(
        "ctx_dow_sin",
        F.sin(2 * math.pi * F.col("ctx_day_of_week") / 7.0),
    )
    df = df.withColumn(
        "ctx_dow_cos",
        F.cos(2 * math.pi * F.col("ctx_day_of_week") / 7.0),
    )

    # Is weekend
    df = df.withColumn(
        "ctx_is_weekend",
        F.when(F.col("ctx_day_of_week") >= 5, 1).otherwise(0),
    )

    # ---- Session position features ----
    # Position within an impression (proxy for session)
    w_session = Window.partitionBy("impression_id").orderBy("news_id")
    df = df.withColumn(
        "ctx_position_in_impression",
        F.row_number().over(w_session),
    )

    # Impression size
    w_count = Window.partitionBy("impression_id")
    df = df.withColumn(
        "ctx_impression_size",
        F.count("*").over(w_count),
    )

    # Normalized position
    df = df.withColumn(
        "ctx_normalized_position",
        F.col("ctx_position_in_impression") / F.col("ctx_impression_size"),
    )

    # Clean up temp column
    if "_ts" in df.columns:
        df = df.drop("_ts")

    n_features = 10  # approximate
    logger.info("Context features built: %d features added", n_features)

    return df
