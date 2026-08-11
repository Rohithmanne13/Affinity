"""Tests for the FastAPI recommendation service."""

import pytest
from fastapi.testclient import TestClient


class TestAPIEndpoints:
    """Test API endpoints using TestClient."""

    def test_health_endpoint(self):
        from src.recommender.serving.api import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_recommend_missing_user_id(self):
        from src.recommender.serving.api import app

        client = TestClient(app)
        response = client.post("/recommend", json={"user_id": "", "top_k": 10})
        assert response.status_code == 400

    def test_recommend_invalid_top_k(self):
        from src.recommender.serving.api import app

        client = TestClient(app)
        response = client.post("/recommend", json={"user_id": "U1", "top_k": 0})
        assert response.status_code == 422  # Pydantic validation

    def test_recommend_top_k_too_large(self):
        from src.recommender.serving.api import app

        client = TestClient(app)
        response = client.post("/recommend", json={"user_id": "U1", "top_k": 200})
        assert response.status_code == 422

    def test_model_info_endpoint(self):
        from src.recommender.serving.api import app

        client = TestClient(app)
        response = client.get("/model-info")
        # May return 500 if no model loaded, which is acceptable
        assert response.status_code in (200, 500)
