"""
Multi-Source Candidate Generator.

Orchestrates popularity, content-similarity, and collaborative
candidate generation. Handles cold-start fallback.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from recommender.candidates.collaborative import CollaborativeCandidateGenerator
from recommender.candidates.content_similarity import ContentSimilarityCandidateGenerator
from recommender.candidates.popularity import PopularityCandidateGenerator
from recommender.config import ProjectConfig, get_config

logger = logging.getLogger(__name__)


class CandidateGenerator:
    """
    Unified multi-source candidate generator.

    Combines popularity, content-similarity, and collaborative sources,
    deduplicates, and handles cold-start users/items.
    """

    def __init__(self, config: ProjectConfig | None = None):
        cfg = config or get_config()
        cand_cfg = cfg.candidates

        self.pool_size = cand_cfg.get("pool_size", 200)

        pop_cfg = cand_cfg.get("popularity", {})
        self.popularity = PopularityCandidateGenerator(
            top_n=pop_cfg.get("top_n", 100),
            recency_hours=pop_cfg.get("recency_hours", 72),
        )

        cs_cfg = cand_cfg.get("content_similarity", {})
        self.content_sim = ContentSimilarityCandidateGenerator(
            top_n=cs_cfg.get("top_n", 50),
        )

        collab_cfg = cand_cfg.get("collaborative", {})
        self.collaborative = CollaborativeCandidateGenerator(
            top_n=collab_cfg.get("top_n", 50),
            min_common=collab_cfg.get("min_common_interactions", 3),
            max_similar_users=collab_cfg.get("max_similar_users", 20),
        )

        self._fitted = False

    def fit(
        self,
        item_features_pdf: pd.DataFrame,
        interactions_pdf: pd.DataFrame,
    ) -> "CandidateGenerator":
        """Fit all candidate generators."""
        logger.info("Fitting candidate generators...")
        self.popularity.fit(item_features_pdf)
        self.content_sim.fit(item_features_pdf)
        self.collaborative.fit(interactions_pdf)
        self._fitted = True
        logger.info("All candidate generators fitted")
        return self

    def generate(
        self,
        user_id: str,
        user_history: list[str] | None = None,
        user_top_category: str | None = None,
        is_cold_start: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Generate a deduplicated candidate pool for a user.

        Parameters
        ----------
        user_id : str
        user_history : list of str, optional
            User's historical clicked news IDs.
        user_top_category : str, optional
        is_cold_start : bool
            If True, use cold-start fallback (popularity + diverse categories).

        Returns
        -------
        list of dict
            Each dict has: news_id, source, candidate_score.
        """
        if not self._fitted:
            raise RuntimeError("CandidateGenerator not fitted. Call .fit() first.")

        seen: set[str] = set(user_history or [])
        all_candidates: list[dict[str, Any]] = []

        if is_cold_start:
            # Cold-start: popularity + category diversity
            pop = self.popularity.generate(
                user_id,
                user_top_category=None,
                exclude_ids=seen,
                n=self.pool_size,
            )
            all_candidates.extend(pop)
        else:
            # Popularity candidates
            pop = self.popularity.generate(
                user_id,
                user_top_category=user_top_category,
                exclude_ids=seen,
                n=self.pool_size // 3,
            )
            all_candidates.extend(pop)
            seen.update(c["news_id"] for c in pop)

            # Content-similarity candidates
            cs = self.content_sim.generate(
                user_id,
                user_history=user_history,
                exclude_ids=seen,
                n=self.pool_size // 3,
            )
            all_candidates.extend(cs)
            seen.update(c["news_id"] for c in cs)

            # Collaborative candidates
            collab = self.collaborative.generate(
                user_id,
                exclude_ids=seen,
                n=self.pool_size // 3,
            )
            all_candidates.extend(collab)

        # Deduplicate
        unique: dict[str, dict[str, Any]] = {}
        for c in all_candidates:
            nid = c["news_id"]
            if nid not in unique:
                unique[nid] = c

        result = list(unique.values())[: self.pool_size]

        logger.debug(
            "User %s: %d candidates (pop=%d, cs=%d, collab=%d, cold=%s)",
            user_id,
            len(result),
            sum(1 for c in result if "popularity" in c.get("source", "")),
            sum(1 for c in result if c.get("source") == "content_similarity"),
            sum(1 for c in result if c.get("source") == "collaborative"),
            is_cold_start,
        )

        return result
