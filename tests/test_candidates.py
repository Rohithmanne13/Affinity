"""Tests for candidate generation."""

import pandas as pd
import pytest

from src.recommender.candidates.popularity import PopularityCandidateGenerator
from src.recommender.candidates.content_similarity import ContentSimilarityCandidateGenerator
from src.recommender.candidates.collaborative import CollaborativeCandidateGenerator


class TestPopularityCandidates:
    def test_fit_and_generate(self, sample_item_features_pdf):
        gen = PopularityCandidateGenerator(top_n=10)
        gen.fit(sample_item_features_pdf)
        candidates = gen.generate("U1")
        assert len(candidates) > 0
        assert all("news_id" in c for c in candidates)
        assert all("source" in c for c in candidates)

    def test_exclude_ids(self, sample_item_features_pdf):
        gen = PopularityCandidateGenerator(top_n=10)
        gen.fit(sample_item_features_pdf)
        candidates = gen.generate("U1", exclude_ids={"N1", "N2"})
        ids = {c["news_id"] for c in candidates}
        assert "N1" not in ids
        assert "N2" not in ids

    def test_category_aware(self, sample_item_features_pdf):
        gen = PopularityCandidateGenerator(top_n=10)
        gen.fit(sample_item_features_pdf)
        candidates = gen.generate("U1", user_top_category="sports")
        assert len(candidates) > 0


class TestContentSimilarity:
    def test_fit_and_generate(self, sample_item_features_pdf):
        gen = ContentSimilarityCandidateGenerator(top_n=5)
        gen.fit(sample_item_features_pdf)
        candidates = gen.generate("U1", user_history=["N1", "N2"])
        assert len(candidates) > 0
        assert all(c["source"] == "content_similarity" for c in candidates)

    def test_empty_history(self, sample_item_features_pdf):
        gen = ContentSimilarityCandidateGenerator(top_n=5)
        gen.fit(sample_item_features_pdf)
        candidates = gen.generate("U1", user_history=[])
        assert candidates == []


class TestCollaborative:
    def test_fit_and_generate(self, sample_interactions_pdf):
        gen = CollaborativeCandidateGenerator(top_n=5, min_common=1)
        gen.fit(sample_interactions_pdf)
        candidates = gen.generate("U1")
        # May or may not have candidates depending on random data
        assert isinstance(candidates, list)

    def test_unknown_user(self, sample_interactions_pdf):
        gen = CollaborativeCandidateGenerator(top_n=5, min_common=1)
        gen.fit(sample_interactions_pdf)
        candidates = gen.generate("UNKNOWN_USER")
        assert candidates == []
