# System Architecture

## Overview

The Personalized Content Ranking Platform follows a classic multi-stage recommendation system architecture:

```
Raw Data → ETL → Features → Segmentation → Candidates → Ranking → Evaluation → Reranking → Serving
```

## Pipeline Stages

### 1. Data Ingestion (Apache Spark)
- **Input**: MIND dataset (`news.tsv`, `behaviors.tsv`)
- **Processing**: PySpark DataFrames with schema validation
- **Output**: Cleaned, deduplicated interaction records

### 2. Feature Engineering (Spark + Pandas)
- **User features**: Behavioral statistics, CTR, category preferences, session patterns
- **Item features**: Popularity, freshness, content metadata
- **Context features**: Temporal signals (cyclic hour/day encoding), position
- **Relational features**: User-category affinity, user-subcategory affinity
- **Leakage prevention**: Features computed only from training data

### 3. User Segmentation (K-Means)
- StandardScaler normalization
- Silhouette score for optimal K selection
- Cluster assignment as downstream feature

### 4. Candidate Generation (Multi-Source)
- **Popularity**: Time-decayed global + category-aware popular items
- **Content Similarity**: TF-IDF cosine similarity on article text
- **Collaborative**: User-user Jaccard similarity with inverted index
- **Cold-start**: Fallback to popular + diverse items

### 5. Ranking (XGBoost + PyTorch)
- **XGBoost**: `rank:pairwise` learning-to-rank objective
- **Neural**: MLP with BatchNorm + dropout, BCELoss
- Both models rank the candidate pool per user

### 6. Reranking (Freshness + Diversity)
- **Freshness**: Exponential time-decay score adjustment
- **Diversity**: MMR (Maximal Marginal Relevance) for category diversity

### 7. Serving (FastAPI + Docker)
- RecommendationService (business logic)
- FastAPI (HTTP layer)
- Docker (containerized deployment)

## Data Flow

```
MIND Dataset
    ↓
Spark ETL (load, validate, parse impressions)
    ↓
Time-aware Train/Val/Test Split
    ↓
Feature Pipeline (user + item + context + relational)
    ↓
User Clustering → cluster_id feature
    ↓
Candidate Generation (200 candidates per user)
    ↓
Model Training (XGBoost + Neural)
    ↓
Evaluation (Precision@K, Recall@K, NDCG@K)
    ↓
MLflow Logging + Model Registry
    ↓
FastAPI Serving → /recommend → Top-K with reranking
```
