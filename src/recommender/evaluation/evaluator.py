"""
Unified Evaluator.

Evaluates all models (baseline, XGBoost, neural) on the same test data
and produces comparison tables.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from recommender.evaluation.ranking_metrics import evaluate_ranking

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates and compares ranking models."""

    def __init__(self, k_values: list[int] | None = None):
        self.k_values = k_values or [5, 10]
        self.results: dict[str, dict[str, float]] = {}

    def evaluate_model(
        self,
        model_name: str,
        test_pdf: pd.DataFrame,
        predict_fn: Any,
    ) -> dict[str, float]:
        """
        Evaluate a model on test data.

        Parameters
        ----------
        model_name : str
        test_pdf : pd.DataFrame
            Must have: impression_id, label, plus feature columns.
        predict_fn : callable
            Function that takes a DataFrame and returns scores.

        Returns
        -------
        dict of metric_name → value
        """
        logger.info("Evaluating %s...", model_name)

        scores = predict_fn(test_pdf)
        test_pdf = test_pdf.copy()
        test_pdf["_pred_score"] = scores

        # Group by impression
        groups = []
        for _, grp in test_pdf.groupby("impression_id"):
            relevant = grp["label"].values.astype(np.float64)
            pred = grp["_pred_score"].values.astype(np.float64)
            if len(relevant) > 0:
                groups.append((relevant, pred))

        metrics = evaluate_ranking(groups, self.k_values)
        self.results[model_name] = metrics

        logger.info("%s metrics:", model_name)
        for k, v in sorted(metrics.items()):
            logger.info("  %s: %.4f", k, v)

        return metrics

    def comparison_table(self) -> pd.DataFrame:
        """Create a comparison table of all evaluated models."""
        if not self.results:
            return pd.DataFrame()

        rows = []
        for model_name, metrics in self.results.items():
            row = {"Model": model_name}
            row.update(metrics)
            rows.append(row)

        df = pd.DataFrame(rows).set_index("Model")
        return df

    def print_comparison(self) -> None:
        """Print a formatted comparison table."""
        table = self.comparison_table()
        if table.empty:
            logger.info("No models evaluated yet.")
            return

        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)
        print(table.to_string(float_format="%.4f"))
        print("=" * 70 + "\n")
