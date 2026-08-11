# =============================================================================
# Personalized Content Ranking — Makefile
# =============================================================================
# Usage: make <target>
# Run `make help` to see all available targets.

.PHONY: help install install-dev lint format typecheck test test-cov \
        preprocess features train-baseline train-tree train-neural train \
        evaluate api docker-build docker-run benchmark clean

PYTHON ?= python
PIP ?= pip
PYTEST ?= pytest

# ---- Help ----
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---- Setup ----
install: ## Install production dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev: ## Install all dependencies (production + dev)
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .

# ---- Code Quality ----
lint: ## Run linting (ruff + black check)
	ruff check src/ tests/ scripts/
	black --check src/ tests/ scripts/

format: ## Auto-format code (ruff fix + black)
	ruff check --fix src/ tests/ scripts/
	black src/ tests/ scripts/

typecheck: ## Run mypy type checking
	mypy src/recommender/

# ---- Testing ----
test: ## Run unit tests
	$(PYTEST) tests/ -v

test-cov: ## Run tests with coverage
	$(PYTEST) tests/ -v --cov=src/recommender --cov-report=html --cov-report=term

# ---- Data Pipeline ----
preprocess: ## Download and preprocess the MIND dataset
	$(PYTHON) scripts/download_data.py
	$(PYTHON) scripts/preprocess.py

features: ## Build feature engineering pipeline
	$(PYTHON) scripts/build_features.py

# ---- Training ----
train-baseline: ## Train the popularity baseline model
	$(PYTHON) scripts/train_baseline.py

train-tree: ## Train the XGBoost tree ranker
	$(PYTHON) scripts/train_tree_model.py

train-neural: ## Train the PyTorch neural ranker
	$(PYTHON) scripts/train_neural_model.py

train: train-baseline train-tree train-neural ## Train all models

# ---- Evaluation ----
evaluate: ## Evaluate all models
	$(PYTHON) scripts/evaluate_models.py

# ---- API ----
api: ## Start the FastAPI recommendation service
	uvicorn src.recommender.serving.api:app --host 0.0.0.0 --port 8000 --reload

# ---- Docker ----
docker-build: ## Build Docker image
	docker build -t content-ranking-api .

docker-run: ## Run Docker container
	docker run -p 8000:8000 content-ranking-api

# ---- Benchmarking ----
benchmark: ## Run API latency benchmark
	$(PYTHON) scripts/benchmark_api.py

# ---- Cleanup ----
clean: ## Remove generated files and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ htmlcov/ .coverage
