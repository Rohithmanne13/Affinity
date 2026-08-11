"""
Popularity Baseline Recommender.

Ranks items by global popularity score. Used as a baseline
for comparison against ML ranking models.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recommender.config import ProjectConfig, get_config

logger = logging.getLogger(__name__)


class PopularityBaseline:
    """Simple popularity-based ranking baseline."""

    def __init__(self, recency_weight: float = 0.3, time_decay: bool = True):
        self.recency_weight = recency_weight
        self.time_decay = time_decay
        self._item_scores: dict[str, float] = {}

    def fit(self, item_features_pdf: pd.DataFrame) -> "PopularityBaseline":
        """
        Compute popularity scores for all items.

        Score = (1 - w) * normalized_clicks + w * freshness_score
        """
        df = item_features_pdf.copy()

        # Normalize clicks
        max_clicks = df["item_total_clicks"].max()
        if max_clicks > 0:
            df["_norm_clicks"] = df["item_total_clicks"] / max_clicks
        else:
            df["_norm_clicks"] = 0.0

        # Freshness
        if self.time_decay and "item_freshness_score" in df.columns:
            freshness = df["item_freshness_score"]
        else:
            freshness = pd.Series(1.0, index=df.index)

        df["_score"] = (1 - self.recency_weight) * df[
            "_norm_clicks"
        ] + self.recency_weight * freshness

        self._item_scores = dict(zip(df["news_id"], df["_score"]))
        logger.info("Popularity baseline fitted: %d items", len(self._item_scores))
        return self

    def predict(self, news_ids: list[str]) -> np.ndarray:
        """Return popularity scores for given news IDs."""
        return np.array([self._item_scores.get(nid, 0.0) for nid in news_ids])

    def rank(self, news_ids: list[str], top_k: int = 10) -> list[tuple[str, float]]:
        """Rank news IDs by popularity and return top-K."""
        scores = self.predict(news_ids)
        indices = np.argsort(-scores)[:top_k]
        return [(news_ids[i], float(scores[i])) for i in indices]

    def save(self, config: ProjectConfig | None = None) -> None:
        """Save baseline model."""
        cfg = config or get_config()
        out_dir = cfg.artifacts_dir / "baseline"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "item_scores.json", "w") as f:
            json.dump(self._item_scores, f)
        logger.info("Baseline saved to %s", out_dir)

    @classmethod
    def load(cls, config: ProjectConfig | None = None) -> "PopularityBaseline":
        """Load baseline model."""
        cfg = config or get_config()
        path = cfg.artifacts_dir / "baseline" / "item_scores.json"
        with open(path) as f:
            scores = json.load(f)
        model = cls()
        model._item_scores = scores
        logger.info("Baseline loaded: %d items", len(scores))
        return model
