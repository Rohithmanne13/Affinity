"""
Content-Similarity Candidate Generation.

Uses TF-IDF on article titles/categories to find
content-similar items for each user based on their history.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ContentSimilarityCandidateGenerator:
    """Generate candidates based on content similarity to user history."""

    def __init__(self, top_n: int = 50, max_features: int = 5000):
        self.top_n = top_n
        self.max_features = max_features
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix: np.ndarray | None = None
        self._news_ids: list[str] = []
        self._news_id_to_idx: dict[str, int] = {}

    def fit(self, item_features_pdf: pd.DataFrame) -> "ContentSimilarityCandidateGenerator":
        """
        Build TF-IDF representations from article content.

        Parameters
        ----------
        item_features_pdf : pd.DataFrame
            Must have: news_id, title, category, subcategory.
        """
        df = item_features_pdf.copy()

        # Combine text features into a single document per article
        df["_text"] = (
            df["category"].fillna("")
            + " "
            + df["subcategory"].fillna("")
            + " "
            + df["title"].fillna("")
        ).str.strip()

        # Remove empty documents
        df = df[df["_text"].str.len() > 0].reset_index(drop=True)

        self._news_ids = df["news_id"].tolist()
        self._news_id_to_idx = {nid: i for i, nid in enumerate(self._news_ids)}

        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=(1, 2),
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(df["_text"])

        logger.info(
            "Content similarity model: %d articles, %d features",
            len(self._news_ids),
            self._tfidf_matrix.shape[1],
        )
        return self

    def generate(
        self,
        user_id: str,
        user_history: list[str] | None = None,
        exclude_ids: set[str] | None = None,
        n: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate content-similar candidates based on user history.

        Returns list of {"news_id": ..., "source": "content_similarity", "candidate_score": ...}
        """
        n = n or self.top_n
        exclude = exclude_ids or set()

        if not user_history or self._tfidf_matrix is None:
            return []

        # Build user profile from history items
        hist_indices = [self._news_id_to_idx[h] for h in user_history if h in self._news_id_to_idx]
        if not hist_indices:
            return []

        # Average TF-IDF vector of history items
        user_profile = self._tfidf_matrix[hist_indices].mean(axis=0)

        # Compute similarity to all items
        sims = cosine_similarity(user_profile, self._tfidf_matrix).flatten()

        # Rank and filter
        ranked_indices = np.argsort(-sims)
        candidates = []
        for idx in ranked_indices:
            nid = self._news_ids[idx]
            if nid not in exclude and nid not in set(user_history):
                candidates.append(
                    {
                        "news_id": nid,
                        "source": "content_similarity",
                        "candidate_score": float(sims[idx]),
                    }
                )
                if len(candidates) >= n:
                    break

        return candidates
