"""
Preprocessing CLI Entry Point.

Runs the complete Spark-based preprocessing pipeline:
  1. Load MIND raw data
  2. Parse impressions & histories
  3. Clean and validate
  4. Time-aware train/val/test split
  5. Save to Parquet

Usage:
    python scripts/preprocess.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recommender.config import get_config, set_seeds, setup_logging
from src.recommender.data.preprocessing import run_preprocessing, save_processed_data
from src.recommender.data.spark_session import create_spark_session, stop_spark

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the preprocessing pipeline."""
    setup_logging()
    cfg = get_config()
    set_seeds(cfg.seed)

    logger.info("Starting preprocessing pipeline...")
    start = time.time()

    spark = create_spark_session(cfg, app_name="MIND-Preprocessing")

    try:
        news_df, train_df, val_df, test_df, stats = run_preprocessing(spark, cfg)
        save_processed_data(news_df, train_df, val_df, test_df, cfg)
    finally:
        stop_spark()

    elapsed = time.time() - start
    logger.info("Preprocessing complete in %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
