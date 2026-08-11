"""Tests for feature generation logic."""

import numpy as np
import pandas as pd
import pytest


class TestFeatureGeneration:
    def test_user_features_have_expected_columns(self, sample_user_features_pdf):
        expected = {"user_id", "user_total_interactions", "user_ctr"}
        assert expected.issubset(set(sample_user_features_pdf.columns))

    def test_item_features_have_expected_columns(self, sample_item_features_pdf):
        expected = {"news_id", "item_total_clicks", "item_ctr"}
        assert expected.issubset(set(sample_item_features_pdf.columns))

    def test_ctr_in_valid_range(self, sample_item_features_pdf):
        assert (sample_item_features_pdf["item_ctr"] >= 0).all()
        assert (sample_item_features_pdf["item_ctr"] <= 1).all()

    def test_no_negative_counts(self, sample_item_features_pdf):
        assert (sample_item_features_pdf["item_total_clicks"] >= 0).all()
        assert (sample_item_features_pdf["item_total_impressions"] >= 0).all()
