"""
Configuration management for the recommendation system.

Loads YAML configs and environment variables, provides a single
ProjectConfig object used across all modules. No hard-coded paths.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

_FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _FILE_DIR.parent.parent  # src/recommender -> project root


def _resolve(path_str: str) -> Path:
    """Resolve a path relative to PROJECT_ROOT."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class ProjectConfig:
    """Centralised, immutable project configuration."""

    # Raw dicts loaded from YAML
    _main: dict[str, Any] = field(default_factory=dict, repr=False)
    _spark: dict[str, Any] = field(default_factory=dict, repr=False)
    _model: dict[str, Any] = field(default_factory=dict, repr=False)

    # ---- Convenience accessors ----

    @property
    def seed(self) -> int:
        return int(os.getenv("RANDOM_SEED", self._main.get("project", {}).get("seed", 42)))

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    # Paths
    @property
    def data_dir(self) -> Path:
        return _resolve(self._main.get("paths", {}).get("data_dir", "data"))

    @property
    def raw_dir(self) -> Path:
        return _resolve(self._main.get("paths", {}).get("raw_dir", "data/raw"))

    @property
    def processed_dir(self) -> Path:
        return _resolve(self._main.get("paths", {}).get("processed_dir", "data/processed"))

    @property
    def features_dir(self) -> Path:
        return _resolve(self._main.get("paths", {}).get("features_dir", "data/features"))

    @property
    def artifacts_dir(self) -> Path:
        return _resolve(self._main.get("paths", {}).get("artifacts_dir", "artifacts"))

    @property
    def figures_dir(self) -> Path:
        return _resolve(self._main.get("paths", {}).get("figures_dir", "docs/figures"))

    # Dataset
    @property
    def dataset(self) -> dict[str, Any]:
        return self._main.get("dataset", {})

    # Preprocessing
    @property
    def preprocessing(self) -> dict[str, Any]:
        return self._main.get("preprocessing", {})

    # Features
    @property
    def features(self) -> dict[str, Any]:
        return self._main.get("features", {})

    # Splitting
    @property
    def splitting(self) -> dict[str, Any]:
        return self._main.get("splitting", {})

    # Segmentation
    @property
    def segmentation(self) -> dict[str, Any]:
        return self._main.get("segmentation", {})

    # Candidates
    @property
    def candidates(self) -> dict[str, Any]:
        return self._main.get("candidates", {})

    # Reranking
    @property
    def reranking(self) -> dict[str, Any]:
        return self._main.get("reranking", {})

    # Evaluation
    @property
    def evaluation(self) -> dict[str, Any]:
        return self._main.get("evaluation", {})

    # API
    @property
    def api(self) -> dict[str, Any]:
        return self._main.get("api", {})

    # Spark config dict
    @property
    def spark(self) -> dict[str, Any]:
        return self._spark

    # Model configs
    @property
    def xgboost(self) -> dict[str, Any]:
        return self._model.get("xgboost", {})

    @property
    def neural(self) -> dict[str, Any]:
        return self._model.get("neural", {})

    @property
    def baseline(self) -> dict[str, Any]:
        return self._model.get("baseline", {})

    @property
    def mlflow_config(self) -> dict[str, Any]:
        return self._model.get("mlflow", {})


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning empty dict on failure."""
    if not path.exists():
        logger.warning("Config file not found: %s — using defaults", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def load_config(
    config_dir: Path | str | None = None,
) -> ProjectConfig:
    """
    Load all configuration files and return a ProjectConfig.

    Parameters
    ----------
    config_dir : Path or str, optional
        Directory containing config.yaml, spark.yaml, model.yaml.
        Defaults to ``PROJECT_ROOT / "configs"``.

    Returns
    -------
    ProjectConfig
    """
    if config_dir is None:
        config_dir = PROJECT_ROOT / "configs"
    config_dir = Path(config_dir)

    main_cfg = _load_yaml(config_dir / "config.yaml")
    spark_cfg = _load_yaml(config_dir / "spark.yaml")
    model_cfg = _load_yaml(config_dir / "model.yaml")

    cfg = ProjectConfig(_main=main_cfg, _spark=spark_cfg, _model=model_cfg)
    logger.info("Configuration loaded from %s (seed=%d)", config_dir, cfg.seed)
    return cfg


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seeds(seed: int = 42) -> None:
    """Set seeds for Python, NumPy, and (optionally) PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except ImportError:
        pass

    logger.info("Random seeds set to %d", seed)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(level: str = "INFO") -> None:
    """Configure project-wide logging format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Module-level singleton for convenience
# ---------------------------------------------------------------------------

_config: ProjectConfig | None = None


def get_config() -> ProjectConfig:
    """Return the cached global config (lazy-loaded)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
