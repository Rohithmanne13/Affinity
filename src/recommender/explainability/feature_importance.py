"""
Feature Importance Analysis.

Extracts and visualizes XGBoost feature importance.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recommender.config import ProjectConfig, get_config

logger = logging.getLogger(__name__)


def plot_feature_importance(
    importance: dict[str, float],
    top_n: int = 20,
    title: str = "XGBoost Feature Importance (Gain)",
    config: ProjectConfig | None = None,
) -> Path:
    """
    Create and save a feature importance bar chart.

    Parameters
    ----------
    importance : dict
        Feature name → importance score.
    top_n : int
        Number of top features to display.
    title : str
    config : ProjectConfig, optional

    Returns
    -------
    Path
        Path to saved figure.
    """
    cfg = config or get_config()
    fig_dir = cfg.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Sort by importance
    sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [f[0] for f in reversed(sorted_feats)]
    values = [f[1] for f in reversed(sorted_feats)]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    bars = ax.barh(names, values, color="#2196F3", edgecolor="#1565C0", linewidth=0.5)

    ax.set_xlabel("Importance (Gain)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.tick_params(axis="y", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = fig_dir / "feature_importance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Feature importance plot saved to %s", out_path)
    return out_path
