# =============================================================================
# Personalized Content Ranking — Dockerfile
# =============================================================================
# Production-style container for the recommendation API.

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY configs/ configs/
COPY pyproject.toml .

# Install the package
RUN pip install -e .

# Create directories for artifacts and data
RUN mkdir -p data/raw data/processed data/features artifacts docs/figures

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()" || exit 1

# Start Uvicorn
CMD ["uvicorn", "src.recommender.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
