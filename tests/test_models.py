"""Tests for model training/inference/save/load on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src.recommender.models.baseline import PopularityBaseline


class TestPopularityBaseline:
    def test_fit_and_predict(self, sample_item_features_pdf):
        model = PopularityBaseline()
        model.fit(sample_item_features_pdf)
        scores = model.predict(["N1", "N2", "N3"])
        assert len(scores) == 3
        assert all(s >= 0 for s in scores)

    def test_rank(self, sample_item_features_pdf):
        model = PopularityBaseline()
        model.fit(sample_item_features_pdf)
        ranked = model.rank(["N1", "N2", "N3", "N4"], top_k=2)
        assert len(ranked) == 2
        assert ranked[0][1] >= ranked[1][1]  # Sorted by score

    def test_unknown_item(self, sample_item_features_pdf):
        model = PopularityBaseline()
        model.fit(sample_item_features_pdf)
        scores = model.predict(["UNKNOWN"])
        assert scores[0] == 0.0
