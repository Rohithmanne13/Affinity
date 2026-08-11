"""
Popularity-Based Candidate Generation.

Generates candidates from recent popular content,
with optional category-aware popularity.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PopularityCandidateGenerator:
    """Generate candidate items based on popularity signals."""

    def __init__(
        self,
        top_n: int = 100,
        recency_hours: float = 72.0,
        time_decay: bool = True,
    ):
        self.top_n = top_n
        self.recency_hours = recency_hours
        self.time_decay = time_decay
        self._global_popular: list[str] = []
        self._category_popular: dict[str, list[str]] = {}

    def fit(self, item_features_pdf: pd.DataFrame) -> "PopularityCandidateGenerator":
        """
        Compute popularity rankings from item features.

        Parameters
        ----------
        item_features_pdf : pd.DataFrame
            Must have: news_id, item_total_clicks, item_freshness_score, category.
        """
        df = item_features_pdf.copy()

        # Compute popularity score (clicks * freshness)
        if self.time_decay and "item_freshness_score" in df.columns:
            df["popularity_score"] = df["item_total_clicks"] * df["item_freshness_score"]
        else:
            df["popularity_score"] = df["item_total_clicks"]

        # Global top-N
        top = df.nlargest(self.top_n, "popularity_score")
        self._global_popular = top["news_id"].tolist()

        # Category-aware top-N
        if "category" in df.columns:
            for cat, group in df.groupby("category"):
                cat_top = group.nlargest(min(self.top_n // 5, len(group)), "popularity_score")
                self._category_popular[str(cat)] = cat_top["news_id"].tolist()

        logger.info(
            "Popularity candidates: %d global, %d categories",
            len(self._global_popular),
            len(self._category_popular),
        )
        return self

    def generate(
        self,
        user_id: str,
        user_top_category: str | None = None,
        exclude_ids: set[str] | None = None,
        n: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate popularity-based candidates for a user.

        Returns list of {"news_id": ..., "source": "popularity", "score": ...}
        """
        n = n or self.top_n
        exclude = exclude_ids or set()
        candidates = []

        # Category-specific popular items first
        if user_top_category and user_top_category in self._category_popular:
            for nid in self._category_popular[user_top_category]:
                if nid not in exclude and len(candidates) < n // 3:
                    candidates.append({"news_id": nid, "source": "popularity_category"})
                    exclude.add(nid)

        # Global popular items
        for nid in self._global_popular:
            if nid not in exclude and len(candidates) < n:
                candidates.append({"news_id": nid, "source": "popularity_global"})
                exclude.add(nid)

        # Assign rank-based scores
        for i, c in enumerate(candidates):
            c["candidate_score"] = 1.0 - (i / max(len(candidates), 1))

        return candidates
