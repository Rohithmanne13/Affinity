# Modeling

## Popularity Baseline

Simple non-personalized baseline ranking items by:
```
score = (1 - w) × normalized_clicks + w × freshness_score
```

## XGBoost Tree Ranker

### Objective
`rank:pairwise` — native learning-to-rank with pairwise loss.

### Features
All user, item, context, and relational features (50+ dimensions).

### Hyperparameter Search
Small grid search over: max_depth ∈ {4, 6, 8}, learning_rate ∈ {0.05, 0.1}, n_estimators ∈ {200, 300}.

### Training
- Group-based ranking: items grouped by impression_id
- Early stopping on validation NDCG@10
- Feature importance extraction (gain-based)

## PyTorch Neural Ranker

### Architecture
```
Input (all features, normalized)
    → Linear(D, 256) → BatchNorm → ReLU → Dropout(0.3)
    → Linear(256, 128) → BatchNorm → ReLU → Dropout(0.3)
    → Linear(128, 64) → BatchNorm → ReLU → Dropout(0.3)
    → Linear(64, 1) → Sigmoid
```

### Training
- Optimizer: Adam (lr=0.001, weight_decay=0.0001)
- Loss: BCELoss
- Early stopping: patience=7 on validation loss
- Checkpointing: best model state restored
- CPU-compatible (no GPU required)

## Model Selection

Models are compared on the same test set using identical evaluation methodology. Results are reported honestly — no fabrication.
