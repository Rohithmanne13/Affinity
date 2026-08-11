# API Documentation

## Endpoints

### GET /health
Health check.

**Response:**
```json
{"status": "healthy", "model_loaded": true, "version": "1.0.0"}
```

### GET /model-info
Model metadata.

**Response:**
```json
{
  "model_type": "xgboost",
  "model_version": "1.0.0",
  "n_users": 50000,
  "n_items": 51282,
  "features": 52,
  "clustering_k": 5
}
```

### POST /recommend
Generate personalized recommendations.

**Request:**
```json
{"user_id": "U10000", "top_k": 10}
```

**Response:**
```json
{
  "user_id": "U10000",
  "recommendations": [
    {"item_id": "N12345", "score": 0.91, "source": "personalized", "category": "tech"}
  ],
  "model_version": "1.0.0",
  "is_cold_start": false,
  "candidate_sources": {"personalized": 10}
}
```

### Error Responses

| Code | Description |
|---|---|
| 400 | Invalid user_id (empty) |
| 422 | Validation error (invalid top_k) |
| 500 | Internal server error |
| 503 | Service not initialized |

## Swagger UI
Available at `http://localhost:8000/docs` when the API is running.
