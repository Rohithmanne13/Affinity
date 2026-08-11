"""Train the XGBoost tree ranking model."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recommender.config import get_config, set_seeds, setup_logging
from src.recommender.explainability.feature_importance import plot_feature_importance
from src.recommender.models.model_registry import log_training_run
from src.recommender.models.tree_ranker import TreeRanker

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    cfg = get_config()
    set_seeds(cfg.seed)

    logger.info("Training XGBoost tree ranker...")
    start = time.time()

    import pandas as pd
    features_dir = cfg.features_dir

    train_pdf = pd.read_parquet(str(features_dir / "train"))
    val_pdf = pd.read_parquet(str(features_dir / "val"))
    logger.info("Train: %d rows, Val: %d rows", len(train_pdf), len(val_pdf))

    # Train
    ranker = TreeRanker(config=cfg)
    ranker.fit(train_pdf, val_pdf)
    ranker.save(cfg)

    elapsed = time.time() - start

    # Feature importance plot
    if ranker.feature_importance_:
        plot_feature_importance(ranker.feature_importance_, config=cfg)

    # SHAP analysis
    try:
        from src.recommender.explainability.shap_analysis import run_shap_analysis
        run_shap_analysis(ranker.model, train_pdf[ranker.feature_cols].head(500), config=cfg)
    except Exception as e:
        logger.warning("SHAP analysis skipped: %s", e)

    # Log to MLflow
    try:
        exp_name = cfg.mlflow_config.get("experiments", {}).get("xgboost", "content-ranking-xgboost")
        log_training_run(
            experiment_name=exp_name,
            model_type="xgboost_ranker",
            params=ranker.best_params,
            metrics={"n_features": len(ranker.feature_cols), "training_time": ranker.training_time},
            training_time=elapsed,
        )
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)

    logger.info("XGBoost training complete in %.1fs", elapsed)


if __name__ == "__main__":
    main()
