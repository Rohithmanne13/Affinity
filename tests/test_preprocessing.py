"""Tests for preprocessing logic."""

import numpy as np
import pandas as pd
import pytest


class TestPreprocessingLogic:
    """Test preprocessing utilities without Spark."""

    def test_interaction_label_distribution(self, sample_interactions_pdf):
        labels = sample_interactions_pdf["label"]
        assert set(labels.unique()).issubset({0, 1})

    def test_no_null_user_ids(self, sample_interactions_pdf):
        assert sample_interactions_pdf["user_id"].isna().sum() == 0

    def test_no_null_news_ids(self, sample_interactions_pdf):
        assert sample_interactions_pdf["news_id"].isna().sum() == 0

    def test_timestamp_ordering(self, sample_interactions_pdf):
        ts = sample_interactions_pdf["timestamp_unix"].values
        assert ts[-1] >= ts[0]  # Chronological order


class TestNewsSchema:
    def test_required_columns(self, sample_news_pdf):
        required = {"news_id", "category", "subcategory", "title"}
        assert required.issubset(set(sample_news_pdf.columns))

    def test_no_duplicate_news_ids(self, sample_news_pdf):
        assert sample_news_pdf["news_id"].is_unique
