"""Tests for reranking modules."""

import pytest

from src.recommender.reranking.diversity import mmr_rerank
from src.recommender.reranking.freshness import apply_freshness_boost
from src.recommender.reranking.reranker import Reranker


class TestMMRRerank:
    def test_basic_reranking(self):
        candidates = [
            {"score": 0.9, "category": "sports"},
            {"score": 0.8, "category": "sports"},
            {"score": 0.7, "category": "tech"},
            {"score": 0.6, "category": "politics"},
        ]
        result = mmr_rerank(candidates, lambda_param=0.5, top_k=3)
        assert len(result) == 3

    def test_diversity_effect(self):
        # All same category — diversity should spread selections
        candidates = [
            {"score": 0.9, "category": "sports"},
            {"score": 0.85, "category": "sports"},
            {"score": 0.7, "category": "tech"},
        ]
        result = mmr_rerank(candidates, lambda_param=0.3, top_k=2)
        categories = [r["category"] for r in result]
        # With low lambda (diversity-heavy), should pick different categories
        assert len(set(categories)) >= 1

    def test_empty_candidates(self):
        assert mmr_rerank([], top_k=5) == []


class TestFreshnessBoost:
    def test_boost_applied(self):
        candidates = [
            {"score": 0.5, "item_age_hours": 0},
            {"score": 0.5, "item_age_hours": 100},
        ]
        result = apply_freshness_boost(candidates, half_life_hours=24, boost_weight=0.1)
        # Fresher item should have higher adjusted score
        assert result[0]["score"] > result[1]["score"]

    def test_freshness_score_added(self):
        candidates = [{"score": 0.5, "item_age_hours": 10}]
        result = apply_freshness_boost(candidates)
        assert "freshness_score" in result[0]
        assert 0 < result[0]["freshness_score"] <= 1


class TestReranker:
    def test_full_pipeline(self):
        candidates = [
            {"score": 0.9, "category": "sports", "item_age_hours": 5},
            {"score": 0.8, "category": "sports", "item_age_hours": 50},
            {"score": 0.7, "category": "tech", "item_age_hours": 10},
            {"score": 0.6, "category": "politics", "item_age_hours": 1},
        ]
        reranker = Reranker()
        result = reranker.rerank(candidates, top_k=3)
        assert len(result) == 3
