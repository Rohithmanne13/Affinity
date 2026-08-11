"""
Combined Reranking Pipeline.

ML Score → Freshness Adjustment → Diversity Reranking → Final Top-K
"""

from __future__ import annotations

import logging
from typing import Any

from recommender.config import ProjectConfig, get_config
from recommender.reranking.diversity import mmr_rerank
from recommender.reranking.freshness import apply_freshness_boost

logger = logging.getLogger(__name__)


class Reranker:
    """Combined freshness + diversity reranking pipeline."""

    def __init__(self, config: ProjectConfig | None = None):
        cfg = config or get_config()
        rr_cfg = cfg.reranking

        div_cfg = rr_cfg.get("diversity", {})
        self.diversity_enabled = div_cfg.get("enabled", True)
        self.lambda_param = div_cfg.get("lambda_param", 0.6)

        fresh_cfg = rr_cfg.get("freshness", {})
        self.freshness_enabled = fresh_cfg.get("enabled", True)
        self.half_life = fresh_cfg.get("half_life_hours", 24.0)
        self.boost_weight = fresh_cfg.get("boost_weight", 0.1)

    def rerank(
        self,
        candidates: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Apply freshness and diversity reranking.

        Parameters
        ----------
        candidates : list of dict
            Must have "score" key. Optionally "category", "item_age_hours".
        top_k : int

        Returns
        -------
        list of dict
            Final reranked top-K.
        """
        if not candidates:
            return []

        # 1. Freshness adjustment
        if self.freshness_enabled:
            candidates = apply_freshness_boost(
                candidates,
                half_life_hours=self.half_life,
                boost_weight=self.boost_weight,
            )

        # 2. Diversity reranking (MMR)
        if self.diversity_enabled:
            # Take top-N before reranking
            candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            top_n = min(len(candidates), top_k * 5)
            candidates = mmr_rerank(
                candidates[:top_n],
                lambda_param=self.lambda_param,
                top_k=top_k,
            )
        else:
            candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            candidates = candidates[:top_k]

        return candidates
