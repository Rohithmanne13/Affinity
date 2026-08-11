# MLflow Integration

## Setup

MLflow uses a local file store (`./mlruns`). No external server required.

```bash
# View the tracking UI
mlflow ui --port 5000
# Open http://localhost:5000
```

## Experiments

| Experiment | Model |
|---|---|
| `content-ranking-baseline` | Popularity baseline |
| `content-ranking-xgboost` | XGBoost tree ranker |
| `content-ranking-neural` | PyTorch neural ranker |

## What Gets Logged

- **Parameters**: Model hyperparameters, configuration
- **Metrics**: Precision@K, Recall@K, NDCG@K, training time
- **Artifacts**: Model files, feature metadata
- **Tags**: Model type, dataset version

## Model Registry

```bash
# Register a model version
python scripts/register_model.py --run-id <RUN_ID>
```

The system functions without the MLflow UI running.
