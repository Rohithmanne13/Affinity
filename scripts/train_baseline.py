"""Train the popularity baseline model."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recommender.config import get_config, set_seeds, setup_logging
from src.recommender.models.baseline import PopularityBaseline
from src.recommender.models.model_registry import log_training_run

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    cfg = get_config()
    set_seeds(cfg.seed)

    logger.info("Training popularity baseline...")
    start = time.time()

    import pandas as pd
    features_dir = cfg.features_dir

    # Load item features
    item_feats = pd.read_parquet(str(features_dir / "item_features"))
    logger.info("Loaded item features: %d items", len(item_feats))

    # Train baseline
    baseline = PopularityBaseline(
        recency_weight=cfg.baseline.get("recency_weight", 0.3),
        time_decay=cfg.baseline.get("time_decay", True),
    )
    baseline.fit(item_feats)
    baseline.save(cfg)

    elapsed = time.time() - start

    # Log to MLflow
    try:
        exp_name = cfg.mlflow_config.get("experiments", {}).get("baseline", "content-ranking-baseline")
        log_training_run(
            experiment_name=exp_name,
            model_type="popularity_baseline",
            params={"recency_weight": baseline.recency_weight, "time_decay": baseline.time_decay},
            metrics={"n_items": len(baseline._item_scores)},
            training_time=elapsed,
        )
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)

    logger.info("Baseline training complete in %.1fs", elapsed)


if __name__ == "__main__":
    main()
