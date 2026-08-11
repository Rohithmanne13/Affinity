"""
Shared test fixtures.

Provides synthetic data fixtures so unit tests do NOT require
the complete MIND dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_news_pdf() -> pd.DataFrame:
    """Synthetic news articles."""
    return pd.DataFrame(
        {
            "news_id": [f"N{i}" for i in range(1, 21)],
            "category": ["sports"] * 5 + ["tech"] * 5 + ["politics"] * 5 + ["entertainment"] * 5,
            "subcategory": [f"sub_{i % 5}" for i in range(20)],
            "title": [f"Title of article {i}" for i in range(1, 21)],
            "abstract": [f"Abstract for article {i}" for i in range(1, 21)],
            "url": [f"https://example.com/{i}" for i in range(1, 21)],
            "title_entities": [""] * 20,
            "abstract_entities": [""] * 20,
        }
    )


@pytest.fixture
def sample_interactions_pdf() -> pd.DataFrame:
    """Synthetic user-item interactions."""
    np.random.seed(42)
    n = 200
    users = [f"U{i % 10 + 1}" for i in range(n)]
    items = [f"N{np.random.randint(1, 21)}" for _ in range(n)]
    labels = np.random.randint(0, 2, n).tolist()
    impressions = [f"IMP{i // 10 + 1}" for i in range(n)]
    timestamps = np.linspace(1572000000, 1572600000, n).tolist()

    return pd.DataFrame(
        {
            "user_id": users,
            "news_id": items,
            "label": labels,
            "impression_id": impressions,
            "timestamp_unix": timestamps,
            "history_length": np.random.randint(0, 20, n).tolist(),
        }
    )


@pytest.fixture
def sample_item_features_pdf(sample_news_pdf: pd.DataFrame) -> pd.DataFrame:
    """Synthetic item features."""
    np.random.seed(42)
    df = sample_news_pdf.copy()
    df["item_total_impressions"] = np.random.randint(10, 500, len(df))
    df["item_total_clicks"] = (
        df["item_total_impressions"] * np.random.uniform(0.05, 0.4, len(df))
    ).astype(int)
    df["item_ctr"] = df["item_total_clicks"] / df["item_total_impressions"]
    df["item_unique_users"] = np.random.randint(5, 200, len(df))
    df["item_freshness_score"] = np.random.uniform(0.1, 1.0, len(df))
    df["item_age_hours"] = np.random.uniform(1, 200, len(df))
    df["item_smoothed_ctr"] = (df["item_total_clicks"] + 1) / (df["item_total_impressions"] + 10)
    df["item_title_length"] = np.random.randint(3, 15, len(df))
    df["item_abstract_length"] = np.random.randint(10, 50, len(df))
    df["item_has_abstract"] = 1
    df["item_non_clicks"] = df["item_total_impressions"] - df["item_total_clicks"]
    df["item_recency_hours"] = np.random.uniform(0, 100, len(df))
    df["item_first_seen_ts"] = 1572000000.0
    df["item_last_seen_ts"] = 1572500000.0
    df["category_total_impressions"] = np.random.randint(100, 5000, len(df))
    df["category_total_clicks"] = np.random.randint(10, 1000, len(df))
    df["category_avg_ctr"] = np.random.uniform(0.05, 0.3, len(df))
    return df


@pytest.fixture
def sample_user_features_pdf() -> pd.DataFrame:
    """Synthetic user features."""
    np.random.seed(42)
    users = [f"U{i}" for i in range(1, 11)]
    return pd.DataFrame(
        {
            "user_id": users,
            "user_total_interactions": np.random.randint(5, 100, 10),
            "user_total_clicks": np.random.randint(1, 50, 10),
            "user_ctr": np.random.uniform(0.05, 0.5, 10),
            "user_unique_items": np.random.randint(3, 50, 10),
            "user_interaction_frequency": np.random.uniform(0.5, 10, 10),
            "user_click_diversity": np.random.uniform(0.1, 1.0, 10),
            "user_category_diversity": np.random.randint(1, 5, 10),
            "user_subcategory_diversity": np.random.randint(1, 10, 10),
            "user_total_sessions": np.random.randint(1, 20, 10),
            "user_avg_session_size": np.random.uniform(2, 20, 10),
            "user_avg_session_clicks": np.random.uniform(0.5, 5, 10),
            "user_history_length": np.random.randint(0, 30, 10),
            "user_top_category": np.random.choice(["sports", "tech", "politics"], 10),
            "user_unique_clicked_items": np.random.randint(1, 30, 10),
            "user_max_session_size": np.random.randint(5, 30, 10),
            "user_active_duration_days": np.random.uniform(1, 30, 10),
            "user_first_interaction_ts": [1572000000.0] * 10,
            "user_last_interaction_ts": [1572500000.0] * 10,
        }
    )
