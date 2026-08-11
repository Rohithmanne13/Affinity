"""
Pydantic Schemas for the Recommendation API.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    """Request schema for /recommend endpoint."""

    user_id: str = Field(..., description="User ID to generate recommendations for")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of recommendations")


class RecommendationItem(BaseModel):
    """A single recommendation."""

    item_id: str
    score: float
    source: str = ""
    category: str = ""


class RecommendResponse(BaseModel):
    """Response schema for /recommend endpoint."""

    user_id: str
    recommendations: list[RecommendationItem]
    model_version: str = "1.0.0"
    is_cold_start: bool = False
    candidate_sources: dict[str, int] = {}


class HealthResponse(BaseModel):
    """Response for /health endpoint."""

    status: str = "healthy"
    model_loaded: bool = False
    version: str = "1.0.0"


class ModelInfoResponse(BaseModel):
    """Response for /model-info endpoint."""

    model_type: str = ""
    model_version: str = "1.0.0"
    n_users: int = 0
    n_items: int = 0
    features: int = 0
    clustering_k: int = 0


class ErrorResponse(BaseModel):
    """Error response."""

    detail: str
    error_code: str = "UNKNOWN"
