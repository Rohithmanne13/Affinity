"""Train the PyTorch neural ranking model."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recommender.config import get_config, set_seeds, setup_logging
from src.recommender.models.model_registry import log_training_run
from src.recommender.models.neural_ranker import NeuralRanker

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    cfg = get_config()
    set_seeds(cfg.seed)

    logger.info("Training PyTorch neural ranker...")
    start = time.time()

    import pandas as pd
    features_dir = cfg.features_dir

    train_pdf = pd.read_parquet(str(features_dir / "train"))
    val_pdf = pd.read_parquet(str(features_dir / "val"))
    logger.info("Train: %d rows, Val: %d rows", len(train_pdf), len(val_pdf))

    # Train
    ranker = NeuralRanker(config=cfg)
    ranker.fit(train_pdf, val_pdf)
    ranker.save(cfg)

    elapsed = time.time() - start

    # Log to MLflow
    try:
        exp_name = cfg.mlflow_config.get("experiments", {}).get("neural", "content-ranking-neural")
        log_training_run(
            experiment_name=exp_name,
            model_type="neural_ranker",
            params=cfg.neural.get("architecture", {}),
            metrics={
                "n_features": len(ranker.feature_cols),
                "training_time": ranker.training_time,
                "final_train_loss": ranker.training_history[-1]["train_loss"] if ranker.training_history else 0,
            },
            training_time=elapsed,
        )
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)

    logger.info("Neural ranker training complete in %.1fs", elapsed)


if __name__ == "__main__":
    main()
