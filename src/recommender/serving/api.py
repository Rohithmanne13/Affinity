"""
FastAPI Recommendation Service.

Endpoints:
    GET  /health       — Service health check
    GET  /model-info   — Model metadata
    POST /recommend    — Generate recommendations
    GET  /docs         — Swagger UI (auto-generated)
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from recommender.config import get_config, setup_logging
from recommender.serving.recommender_service import RecommendationService
from recommender.serving.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    RecommendRequest,
    RecommendResponse,
)

logger = logging.getLogger(__name__)

# Global service instance
_service: RecommendationService | None = None


def get_service() -> RecommendationService:
    """Get the recommendation service singleton."""
    global _service
    if _service is None:
        raise RuntimeError("Service not initialized")
    return _service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artifacts on startup."""
    global _service
    setup_logging()
    logger.info("Starting recommendation service...")

    _service = RecommendationService()
    try:
        _service.load_artifacts()
        logger.info("Service ready!")
    except Exception as e:
        logger.error("Failed to load artifacts: %s", e)
        logger.warning("Service running in degraded mode — no recommendations available")

    yield

    logger.info("Shutting down...")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Personalized Content Ranking API",
    description=(
        "End-to-end personalized content recommendation and ranking service. "
        "Supports personalized ranking, cold-start handling, and "
        "diversity/freshness-aware reranking."
    ),
    version="1.0.0",
    lifespan=lifespan,
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Service health check."""
    service = _service
    return HealthResponse(
        status="healthy",
        model_loaded=service is not None and service._loaded,
        version="1.0.0",
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Return model metadata."""
    try:
        service = get_service()
        info = service.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend", response_model=RecommendResponse, tags=["Recommendations"])
async def recommend(request: RecommendRequest):
    """
    Generate personalized recommendations for a user.

    - **user_id**: User identifier (from MIND dataset)
    - **top_k**: Number of recommendations (1-100)
    """
    start = time.time()

    try:
        service = get_service()
    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized. Model artifacts may not be loaded.",
        )

    if not request.user_id or not request.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id is required")

    try:
        recs, is_cold_start, sources = service.recommend(
            user_id=request.user_id,
            top_k=request.top_k,
        )
    except Exception as e:
        logger.error("Recommendation failed for user %s: %s", request.user_id, e)
        raise HTTPException(status_code=500, detail=f"Recommendation error: {e}")

    elapsed_ms = (time.time() - start) * 1000
    logger.info(
        "Recommend: user=%s, top_k=%d, results=%d, cold_start=%s, latency=%.1fms",
        request.user_id,
        request.top_k,
        len(recs),
        is_cold_start,
        elapsed_ms,
    )

    return RecommendResponse(
        user_id=request.user_id,
        recommendations=recs,
        model_version=service.model_version,
        is_cold_start=is_cold_start,
        candidate_sources=sources,
    )
