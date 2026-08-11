"""
Recommendation Service.

Separates business logic from HTTP concerns. Handles:
artifact loading, cold-start detection, candidate generation,
scoring, reranking, and final top-K selection.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recommender.config import ProjectConfig, get_config
from recommender.reranking.reranker import Reranker
from recommender.serving.schemas import RecommendationItem

logger = logging.getLogger(__name__)


class RecommendationService:
    """
    Core recommendation service.

    Responsibilities:
        1. Load model artifacts
        2. Retrieve user information
        3. Handle cold start
        4. Score candidates
        5. Apply reranking
        6. Return final Top-K
    """

    def __init__(self, config: ProjectConfig | None = None):
        self.cfg = config or get_config()
        self.model = None
        self.model_type = "unknown"
        self.model_version = "1.0.0"
        self.user_features: pd.DataFrame | None = None
        self.item_features: pd.DataFrame | None = None
        self.user_ids: set[str] = set()
        self.item_ids: set[str] = set()
        self.reranker = Reranker(config=self.cfg)
        self._loaded = False

    def load_artifacts(self) -> None:
        """Load all required model artifacts."""
        artifacts_dir = self.cfg.artifacts_dir
        features_dir = self.cfg.features_dir

        logger.info("Loading artifacts from %s", artifacts_dir)

        # Load model (prefer tree ranker, fallback to baseline)
        try:
            from recommender.models.tree_ranker import TreeRanker

            self.model = TreeRanker.load(self.cfg)
            self.model_type = "xgboost"
            logger.info("XGBoost tree ranker loaded")
        except Exception:
            try:
                from recommender.models.baseline import PopularityBaseline

                self.model = PopularityBaseline.load(self.cfg)
                self.model_type = "popularity_baseline"
                logger.info("Popularity baseline loaded (fallback)")
            except Exception as e:
                logger.error("No model available: %s", e)
                raise RuntimeError("No trained model found. Run training first.") from e

        # Load user features
        try:
            uf_path = features_dir / "user_features"
            if uf_path.exists():
                self.user_features = pd.read_parquet(str(uf_path))
                self.user_ids = set(self.user_features["user_id"].tolist())
                logger.info("User features loaded: %d users", len(self.user_ids))
        except Exception as e:
            logger.warning("Could not load user features: %s", e)

        # Load item features
        try:
            if_path = features_dir / "item_features"
            if if_path.exists():
                self.item_features = pd.read_parquet(str(if_path))
                self.item_ids = set(self.item_features["news_id"].tolist())
                logger.info("Item features loaded: %d items", len(self.item_ids))
        except Exception as e:
            logger.warning("Could not load item features: %s", e)

        self._loaded = True
        logger.info("Artifact loading complete")

    def recommend(
        self,
        user_id: str,
        top_k: int = 10,
    ) -> tuple[list[RecommendationItem], bool, dict[str, int]]:
        """
        Generate recommendations for a user.

        Parameters
        ----------
        user_id : str
        top_k : int

        Returns
        -------
        tuple of (recommendations, is_cold_start, source_counts)
        """
        if not self._loaded:
            raise RuntimeError("Artifacts not loaded. Call load_artifacts() first.")

        is_cold_start = user_id not in self.user_ids

        if is_cold_start:
            return self._cold_start_recommend(top_k), True, {"popularity": top_k}

        return self._personalized_recommend(user_id, top_k)

    def _personalized_recommend(
        self,
        user_id: str,
        top_k: int,
    ) -> tuple[list[RecommendationItem], bool, dict[str, int]]:
        """Generate personalized recommendations using the ranking model."""
        # Get top items by popularity as candidates
        if self.item_features is not None:
            candidates_df = self.item_features.nlargest(
                min(200, len(self.item_features)),
                "item_total_clicks",
            ).copy()
        else:
            return self._cold_start_recommend(top_k), True, {"popularity": top_k}

        # Add user features to candidates
        if self.user_features is not None:
            user_row = self.user_features[self.user_features["user_id"] == user_id]
            if not user_row.empty:
                for col in user_row.columns:
                    if col != "user_id" and col not in candidates_df.columns:
                        candidates_df[col] = user_row[col].values[0]

        # Score candidates
        try:
            if self.model_type == "xgboost":
                scores = self.model.predict(candidates_df)
            else:
                scores = self.model.predict(candidates_df["news_id"].tolist())
        except Exception as e:
            logger.warning("Scoring failed: %s — using popularity fallback", e)
            scores = candidates_df.get("item_total_clicks", pd.Series(0)).values.astype(float)

        # Build candidate list for reranking
        rerank_candidates = []
        for i, (_, row) in enumerate(candidates_df.iterrows()):
            rerank_candidates.append(
                {
                    "news_id": row.get("news_id", f"item_{i}"),
                    "score": float(scores[i]) if i < len(scores) else 0.0,
                    "category": row.get("category", "unknown"),
                    "item_age_hours": row.get("item_age_hours", 0.0),
                    "source": "personalized",
                }
            )

        # Rerank
        reranked = self.reranker.rerank(rerank_candidates, top_k=top_k)

        # Convert to response items
        recs = [
            RecommendationItem(
                item_id=c["news_id"],
                score=round(c.get("score", 0.0), 4),
                source=c.get("source", ""),
                category=c.get("category", ""),
            )
            for c in reranked
        ]

        return recs, False, {"personalized": len(recs)}

    def _cold_start_recommend(self, top_k: int) -> list[RecommendationItem]:
        """Generate cold-start recommendations (popular + diverse)."""
        if self.item_features is None:
            return []

        # Top popular items with category diversity
        top = self.item_features.nlargest(top_k * 3, "item_total_clicks")

        # Diversify by category
        seen_cats: set[str] = set()
        recs: list[RecommendationItem] = []

        for _, row in top.iterrows():
            cat = row.get("category", "unknown")
            if len(recs) < top_k:
                recs.append(
                    RecommendationItem(
                        item_id=row.get("news_id", ""),
                        score=round(float(row.get("item_smoothed_ctr", 0.0)), 4),
                        source="cold_start_popular",
                        category=str(cat),
                    )
                )
                seen_cats.add(str(cat))

        return recs[:top_k]

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata."""
        return {
            "model_type": self.model_type,
            "model_version": self.model_version,
            "n_users": len(self.user_ids),
            "n_items": len(self.item_ids),
            "features": len(getattr(self.model, "feature_cols", [])),
            "clustering_k": 0,
        }
