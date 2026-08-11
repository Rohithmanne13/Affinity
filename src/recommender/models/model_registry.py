"""
MLflow Model Registry Integration.

Handles experiment creation, metric/artifact logging,
and model registration/versioning.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from recommender.config import ProjectConfig, get_config

logger = logging.getLogger(__name__)


def get_mlflow():
    """Import mlflow lazily to avoid startup cost."""
    import mlflow

    return mlflow


def setup_mlflow(config: ProjectConfig | None = None) -> None:
    """Configure MLflow tracking URI."""
    cfg = config or get_config()
    mlflow = get_mlflow()
    uri = cfg.mlflow_config.get("tracking_uri", "./mlruns")
    mlflow.set_tracking_uri(uri)
    logger.info("MLflow tracking URI: %s", uri)


def create_experiment(name: str) -> str:
    """Create or get an MLflow experiment."""
    mlflow = get_mlflow()
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
        logger.info("Created MLflow experiment: %s (id=%s)", name, exp_id)
    else:
        exp_id = exp.experiment_id
    return exp_id


def log_training_run(
    experiment_name: str,
    model_type: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: dict[str, str] | None = None,
    training_time: float = 0.0,
    tags: dict[str, str] | None = None,
) -> str:
    """
    Log a complete training run to MLflow.

    Parameters
    ----------
    experiment_name : str
    model_type : str
    params : dict
    metrics : dict
    artifacts : dict of name → file_path, optional
    training_time : float
    tags : dict, optional

    Returns
    -------
    str
        MLflow run ID.
    """
    mlflow = get_mlflow()
    setup_mlflow()

    exp_id = create_experiment(experiment_name)

    with mlflow.start_run(experiment_id=exp_id) as run:
        # Tags
        mlflow.set_tag("model_type", model_type)
        if tags:
            for k, v in tags.items():
                mlflow.set_tag(k, v)

        # Parameters
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.log_metric("training_time_seconds", training_time)

        # Artifacts
        if artifacts:
            for name, path in artifacts.items():
                try:
                    mlflow.log_artifact(path, artifact_path=name)
                except Exception as e:
                    logger.warning("Failed to log artifact %s: %s", name, e)

        run_id = run.info.run_id
        logger.info("MLflow run logged: %s (experiment=%s)", run_id, experiment_name)
        return run_id


def register_model(
    model_name: str,
    run_id: str,
    model_path: str = "model",
    stage: str = "Production",
) -> None:
    """Register a model version in the MLflow Model Registry."""
    mlflow = get_mlflow()
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    try:
        client.create_registered_model(model_name)
    except Exception:
        pass  # Already exists

    model_uri = f"runs:/{run_id}/{model_path}"
    try:
        result = mlflow.register_model(model_uri, model_name)
        logger.info(
            "Model registered: %s version %s",
            model_name,
            result.version,
        )
    except Exception as e:
        logger.warning("Model registration failed: %s", e)
