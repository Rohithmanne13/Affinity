"""
Diversity-Aware Reranking.

Implements Maximal Marginal Relevance (MMR) to balance
relevance and diversity in the final recommendation list.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def mmr_rerank(
    candidates: list[dict[str, Any]],
    lambda_param: float = 0.6,
    top_k: int = 10,
    category_key: str = "category",
) -> list[dict[str, Any]]:
    """
    Maximal Marginal Relevance reranking for diversity.

    MMR selects items that are both relevant AND dissimilar to
    already-selected items. Uses category as the diversity signal.

    Score = λ * relevance - (1-λ) * max_similarity_to_selected

    Parameters
    ----------
    candidates : list of dict
        Each dict must have "score" and optionally category_key.
    lambda_param : float
        Balance parameter. 1.0 = pure relevance, 0.0 = pure diversity.
    top_k : int
        Number of items to select.
    category_key : str
        Key for category in candidate dicts.

    Returns
    -------
    list of dict
        Reranked candidates.
    """
    if not candidates or top_k <= 0:
        return []

    n = min(top_k, len(candidates))
    selected: list[dict[str, Any]] = []
    remaining = list(range(len(candidates)))

    # Normalize scores to [0, 1]
    scores = np.array([c.get("score", 0.0) for c in candidates])
    max_score = scores.max() if scores.max() > 0 else 1.0
    norm_scores = scores / max_score

    for _ in range(n):
        best_idx = -1
        best_mmr = -float("inf")

        for idx in remaining:
            relevance = norm_scores[idx]

            # Category similarity to selected items
            if selected:
                cat = candidates[idx].get(category_key, "")
                selected_cats = [s.get(category_key, "") for s in selected]
                similarity = sum(1 for sc in selected_cats if sc == cat) / len(selected_cats)
            else:
                similarity = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * similarity

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        if best_idx >= 0:
            selected.append(candidates[best_idx])
            remaining.remove(best_idx)

    return selected
