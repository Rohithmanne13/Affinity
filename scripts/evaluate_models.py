"""Evaluate all models and produce comparison table."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recommender.config import get_config, set_seeds, setup_logging
from src.recommender.evaluation.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    cfg = get_config()
    set_seeds(cfg.seed)

    import pandas as pd

    features_dir = cfg.features_dir
    test_pdf = pd.read_parquet(str(features_dir / "test"))
    logger.info("Test set: %d rows", len(test_pdf))

    k_values = cfg.evaluation.get("k_values", [5, 10])
    evaluator = ModelEvaluator(k_values=k_values)

    # 1. Baseline
    try:
        from src.recommender.models.baseline import PopularityBaseline
        baseline = PopularityBaseline.load(cfg)
        evaluator.evaluate_model(
            "Popularity Baseline",
            test_pdf,
            lambda df: baseline.predict(df["news_id"].tolist()),
        )
    except Exception as e:
        logger.warning("Baseline evaluation failed: %s", e)

    # 2. XGBoost
    try:
        from src.recommender.models.tree_ranker import TreeRanker
        tree = TreeRanker.load(cfg)
        evaluator.evaluate_model("XGBoost Ranker", test_pdf, tree.predict)
    except Exception as e:
        logger.warning("XGBoost evaluation failed: %s", e)

    # 3. Neural
    try:
        from src.recommender.models.neural_ranker import NeuralRanker
        neural = NeuralRanker.load(cfg)
        evaluator.evaluate_model("Neural Ranker", test_pdf, neural.predict)
    except Exception as e:
        logger.warning("Neural evaluation failed: %s", e)

    # Print comparison
    evaluator.print_comparison()

    # Save results
    table = evaluator.comparison_table()
    if not table.empty:
        out_dir = cfg.figures_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_dir / "model_comparison.csv")
        logger.info("Results saved to %s", out_dir / "model_comparison.csv")


if __name__ == "__main__":
    main()
