"""Tests for user clustering."""

import numpy as np
import pandas as pd
import pytest

from src.recommender.segmentation.user_clustering import (
    run_user_clustering,
    CLUSTERING_FEATURES,
)


class TestUserClustering:
    def test_clustering_runs(self, sample_user_features_pdf):
        result = run_user_clustering(sample_user_features_pdf)
        assert result.n_clusters >= 2
        assert result.silhouette > -1  # Valid silhouette range is [-1, 1]
        assert len(result.labels) == len(sample_user_features_pdf)
        assert sum(result.cluster_sizes.values()) == len(sample_user_features_pdf)

    def test_cluster_assignment(self, sample_user_features_pdf):
        result = run_user_clustering(sample_user_features_pdf)
        # Each user should get a cluster ID
        assert all(0 <= label < result.n_clusters for label in result.labels)

    def test_cluster_stats(self, sample_user_features_pdf):
        result = run_user_clustering(sample_user_features_pdf)
        assert not result.cluster_stats.empty
        assert result.cluster_stats.shape[0] == result.n_clusters
