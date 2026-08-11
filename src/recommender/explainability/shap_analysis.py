"""
SHAP Analysis for Model Explainability.

Generates SHAP explanations for the XGBoost tree ranker.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recommender.config import ProjectConfig, get_config

logger = logging.getLogger(__name__)


def run_shap_analysis(
    model,
    X: pd.DataFrame | np.ndarray,
    feature_names: list[str] | None = None,
    max_samples: int = 1000,
    config: ProjectConfig | None = None,
) -> Path | None:
    """
    Run SHAP analysis on XGBoost model and save summary plot.

    Parameters
    ----------
    model : xgb.Booster or compatible
        Trained model.
    X : pd.DataFrame or np.ndarray
        Feature matrix for explanation.
    feature_names : list of str, optional
    max_samples : int
        Max samples for SHAP computation.
    config : ProjectConfig, optional

    Returns
    -------
    Path or None
        Path to saved SHAP plot, or None if SHAP unavailable.
    """
    cfg = config or get_config()
    fig_dir = cfg.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed — skipping SHAP analysis")
        return None

    logger.info("Running SHAP analysis (max %d samples)...", max_samples)

    if isinstance(X, pd.DataFrame):
        if len(X) > max_samples:
            X = X.sample(max_samples, random_state=42)
        X_arr = X.values
        feature_names = feature_names or X.columns.tolist()
    else:
        if len(X) > max_samples:
            indices = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
            X_arr = X[indices]
        else:
            X_arr = X

    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_arr)

    # Summary plot
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_arr,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.title("SHAP Feature Impact on Ranking", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = fig_dir / "shap_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("SHAP plot saved to %s", out_path)
    return out_path
