"""Tests for ranking metrics."""

import numpy as np
import pytest

from src.recommender.evaluation.ranking_metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    evaluate_ranking,
)


class TestPrecisionAtK:
    def test_all_relevant(self):
        rel = np.array([1, 1, 1, 1, 1])
        scores = np.array([5, 4, 3, 2, 1])
        assert precision_at_k(rel, scores, 5) == 1.0

    def test_none_relevant(self):
        rel = np.array([0, 0, 0, 0, 0])
        scores = np.array([5, 4, 3, 2, 1])
        assert precision_at_k(rel, scores, 5) == 0.0

    def test_half_relevant(self):
        rel = np.array([1, 0, 1, 0, 0])
        scores = np.array([5, 4, 3, 2, 1])
        assert precision_at_k(rel, scores, 2) == 0.5

    def test_k_larger_than_list(self):
        rel = np.array([1, 0])
        scores = np.array([2, 1])
        result = precision_at_k(rel, scores, 10)
        assert result == 0.5

    def test_empty_input(self):
        assert precision_at_k(np.array([]), np.array([]), 5) == 0.0

    def test_k_zero(self):
        assert precision_at_k(np.array([1, 0]), np.array([2, 1]), 0) == 0.0


class TestRecallAtK:
    def test_all_recalled(self):
        rel = np.array([1, 1, 0, 0, 0])
        scores = np.array([5, 4, 3, 2, 1])
        assert recall_at_k(rel, scores, 2) == 1.0

    def test_none_recalled(self):
        rel = np.array([0, 0, 1, 1, 0])
        scores = np.array([5, 4, 3, 2, 1])
        assert recall_at_k(rel, scores, 2) == 0.0

    def test_no_relevant_items(self):
        rel = np.array([0, 0, 0])
        scores = np.array([3, 2, 1])
        assert recall_at_k(rel, scores, 3) == 0.0


class TestNDCGAtK:
    def test_perfect_ranking(self):
        rel = np.array([1, 1, 0, 0, 0])
        scores = np.array([5, 4, 3, 2, 1])
        assert ndcg_at_k(rel, scores, 5) == 1.0

    def test_worst_ranking(self):
        rel = np.array([0, 0, 0, 1, 1])
        scores = np.array([5, 4, 3, 2, 1])
        assert ndcg_at_k(rel, scores, 5) < 1.0

    def test_no_relevant(self):
        rel = np.array([0, 0, 0])
        scores = np.array([3, 2, 1])
        assert ndcg_at_k(rel, scores, 3) == 0.0

    def test_empty(self):
        assert ndcg_at_k(np.array([]), np.array([]), 5) == 0.0


class TestEvaluateRanking:
    def test_multiple_groups(self):
        groups = [
            (np.array([1, 0, 1]), np.array([3, 2, 1])),
            (np.array([0, 1, 0]), np.array([1, 3, 2])),
        ]
        results = evaluate_ranking(groups, k_values=[2])
        assert "precision@2" in results
        assert "recall@2" in results
        assert "ndcg@2" in results
        assert all(0 <= v <= 1 for v in results.values())
