"""
Freshness-Aware Ranking.

Applies a principled freshness adjustment to ranking scores
using exponential time decay.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


def apply_freshness_boost(
    candidates: list[dict[str, Any]],
    half_life_hours: float = 24.0,
    boost_weight: float = 0.1,
    age_key: str = "item_age_hours",
    score_key: str = "score",
) -> list[dict[str, Any]]:
    """
    Adjust ranking scores with freshness boost.

    adjusted_score = score + boost_weight * exp(-age / half_life)

    Parameters
    ----------
    candidates : list of dict
        Must have score_key and age_key.
    half_life_hours : float
        Half-life for freshness decay in hours.
    boost_weight : float
        Weight of freshness boost relative to relevance score.
    age_key : str
        Key for article age in hours.
    score_key : str
        Key for relevance score.

    Returns
    -------
    list of dict
        Candidates with adjusted scores.
    """
    decay_rate = math.log(2) / half_life_hours

    for c in candidates:
        age = c.get(age_key, 0.0)
        freshness = math.exp(-decay_rate * max(age, 0.0))
        original = c.get(score_key, 0.0)
        c[score_key] = original + boost_weight * freshness
        c["freshness_score"] = freshness

    return candidates
