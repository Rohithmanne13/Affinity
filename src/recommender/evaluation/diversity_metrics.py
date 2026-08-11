"""
Diversity Metrics.

Measures recommendation list diversity:
  - Category coverage
  - Intra-list diversity (ILD)
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def category_coverage(recommended_categories: list[str], all_categories: set[str]) -> float:
    """Fraction of all categories represented in recommendations."""
    if not all_categories:
        return 0.0
    return len(set(recommended_categories) & all_categories) / len(all_categories)


def intra_list_diversity(categories: list[str]) -> float:
    """
    ILD = fraction of pairs with different categories.

    1.0 = maximally diverse, 0.0 = all same category.
    """
    n = len(categories)
    if n <= 1:
        return 0.0

    different_pairs = 0
    total_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            if categories[i] != categories[j]:
                different_pairs += 1

    return different_pairs / total_pairs if total_pairs > 0 else 0.0


def evaluate_diversity(
    recommendations: list[list[dict]],
    all_categories: set[str] | None = None,
) -> dict[str, float]:
    """
    Evaluate diversity metrics across recommendation lists.

    Parameters
    ----------
    recommendations : list of list of dict
        Each inner list is a user's recommendations with "category" key.
    all_categories : set of str, optional

    Returns
    -------
    dict with avg_category_coverage, avg_ild
    """
    coverages = []
    ilds = []

    for rec_list in recommendations:
        cats = [r.get("category", "unknown") for r in rec_list]
        if all_categories:
            coverages.append(category_coverage(cats, all_categories))
        ilds.append(intra_list_diversity(cats))

    return {
        "avg_category_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "avg_intra_list_diversity": float(np.mean(ilds)) if ilds else 0.0,
    }
