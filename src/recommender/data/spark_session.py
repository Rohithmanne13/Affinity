"""
Reusable SparkSession factory.

Provides a configured SparkSession for local development with
sensible defaults that also work on a cluster with minimal changes.
"""

from __future__ import annotations

import logging
from typing import Any

from pyspark.sql import SparkSession

from recommender.config import ProjectConfig, get_config

logger = logging.getLogger(__name__)

_spark_session: SparkSession | None = None


def create_spark_session(
    config: ProjectConfig | None = None,
    app_name: str | None = None,
) -> SparkSession:
    """
    Create or return a cached SparkSession.

    Parameters
    ----------
    config : ProjectConfig, optional
        Project configuration. Loads default if not provided.
    app_name : str, optional
        Override the Spark application name.

    Returns
    -------
    SparkSession
    """
    global _spark_session

    if _spark_session is not None and not _spark_session.sparkContext._jsc.sc().isStopped():
        return _spark_session

    cfg = config or get_config()
    spark_cfg: dict[str, Any] = cfg.spark.get("spark", {})

    name = app_name or spark_cfg.get("app_name", "ContentRankingETL")
    master = spark_cfg.get("master", "local[*]")
    spark_config: dict[str, str] = spark_cfg.get("config", {})

    builder = SparkSession.builder.appName(name).master(master)

    for key, value in spark_config.items():
        builder = builder.config(key, str(value))

    session = builder.getOrCreate()

    # Reduce Spark logging noise
    session.sparkContext.setLogLevel("WARN")

    logger.info(
        "SparkSession created — app=%s, master=%s, partitions=%s",
        name,
        master,
        spark_config.get("spark.sql.shuffle.partitions", "default"),
    )

    _spark_session = session
    return session


def stop_spark() -> None:
    """Stop the cached SparkSession."""
    global _spark_session
    if _spark_session is not None:
        _spark_session.stop()
        _spark_session = None
        logger.info("SparkSession stopped")
