"""
Collaborative Candidate Generation.

Identifies articles consumed by similar users based on
interaction overlap (user-user collaborative filtering).
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CollaborativeCandidateGenerator:
    """Generate candidates from similar users' interactions."""

    def __init__(
        self,
        top_n: int = 50,
        min_common: int = 3,
        max_similar_users: int = 20,
    ):
        self.top_n = top_n
        self.min_common = min_common
        self.max_similar_users = max_similar_users
        self._user_items: dict[str, set[str]] = {}
        self._item_users: dict[str, set[str]] = {}

    def fit(self, interactions_pdf: pd.DataFrame) -> "CollaborativeCandidateGenerator":
        """
        Build user-item interaction index.

        Parameters
        ----------
        interactions_pdf : pd.DataFrame
            Must have: user_id, news_id, label. Uses clicked items only.
        """
        clicked = interactions_pdf[interactions_pdf["label"] == 1]

        self._user_items = defaultdict(set)
        self._item_users = defaultdict(set)

        for _, row in clicked[["user_id", "news_id"]].iterrows():
            uid, nid = row["user_id"], row["news_id"]
            self._user_items[uid].add(nid)
            self._item_users[nid].add(uid)

        logger.info(
            "Collaborative index: %d users, %d items",
            len(self._user_items),
            len(self._item_users),
        )
        return self

    def _find_similar_users(self, user_id: str) -> list[tuple[str, float]]:
        """Find users with highest interaction overlap (Jaccard)."""
        user_items = self._user_items.get(user_id, set())
        if not user_items:
            return []

        # Find candidate similar users via inverted index
        neighbor_counts: Counter = Counter()
        for item in user_items:
            for other_user in self._item_users.get(item, set()):
                if other_user != user_id:
                    neighbor_counts[other_user] += 1

        # Filter by minimum common items and compute Jaccard
        similar = []
        for other_uid, common_count in neighbor_counts.most_common(self.max_similar_users * 3):
            if common_count >= self.min_common:
                other_items = self._user_items[other_uid]
                jaccard = common_count / len(user_items | other_items)
                similar.append((other_uid, jaccard))

        # Sort by similarity
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar[: self.max_similar_users]

    def generate(
        self,
        user_id: str,
        exclude_ids: set[str] | None = None,
        n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Generate collaborative candidates from similar users."""
        n = n or self.top_n
        exclude = exclude_ids or set()
        user_items = self._user_items.get(user_id, set())
        exclude = exclude | user_items

        similar_users = self._find_similar_users(user_id)
        if not similar_users:
            return []

        # Collect items from similar users, weighted by similarity
        item_scores: Counter = Counter()
        for other_uid, sim in similar_users:
            for nid in self._user_items[other_uid]:
                if nid not in exclude:
                    item_scores[nid] += sim

        # Rank by aggregated similarity score
        candidates = []
        for nid, score in item_scores.most_common(n):
            candidates.append(
                {
                    "news_id": nid,
                    "source": "collaborative",
                    "candidate_score": float(score),
                }
            )

        return candidates
