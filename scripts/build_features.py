"""Build features CLI entry point."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recommender.config import get_config, set_seeds, setup_logging
from src.recommender.data.preprocessing import load_processed_data
from src.recommender.data.spark_session import create_spark_session, stop_spark
from src.recommender.features.feature_pipeline import run_feature_pipeline, save_features

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    cfg = get_config()
    set_seeds(cfg.seed)

    logger.info("Starting feature engineering pipeline...")
    start = time.time()
    spark = create_spark_session(cfg, app_name="MIND-FeatureEngineering")

    try:
        data = load_processed_data(spark, cfg)
        feature_data = run_feature_pipeline(
            spark, data["news"], data["train"], data["val"], data["test"], cfg,
        )
        save_features(feature_data, cfg)
    finally:
        stop_spark()

    logger.info("Feature engineering complete in %.1f seconds", time.time() - start)


if __name__ == "__main__":
    main()
