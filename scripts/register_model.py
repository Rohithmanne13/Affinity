"""Register the best model in MLflow Model Registry."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recommender.config import get_config, setup_logging
from src.recommender.models.model_registry import register_model, setup_mlflow

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    cfg = get_config()
    setup_mlflow(cfg)

    model_name = cfg.mlflow_config.get("registry", {}).get("model_name", "content-ranker")
    stage = cfg.mlflow_config.get("registry", {}).get("stage", "Production")

    logger.info("Model registration — use MLflow UI to find the run_id")
    logger.info("Example: python scripts/register_model.py --run-id <RUN_ID>")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="MLflow run ID to register")
    args = parser.parse_args()

    register_model(model_name, args.run_id, stage=stage)
    logger.info("Done!")


if __name__ == "__main__":
    main()
