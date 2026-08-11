"""
PyTorch Neural Ranking Model.

A neural ranker with embedding layers for categorical features
and an MLP for relevance scoring. Supports training with early stopping,
checkpointing, and CPU-compatible execution.

Architecture:
    User features + Item features + Context features
        → Concatenation
        → MLP (BatchNorm + ReLU + Dropout)
        → Relevance score (sigmoid)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from recommender.config import ProjectConfig, get_config, set_seeds

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RankingDataset(Dataset):
    """PyTorch Dataset for ranking interactions."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class NeuralRankerModel(nn.Module):
    """MLP-based neural ranking model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
        batch_norm: bool = True,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128, 64]

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class NeuralRanker:
    """Training and inference wrapper for the neural ranking model."""

    def __init__(self, config: ProjectConfig | None = None):
        cfg = config or get_config()
        self.neural_cfg = cfg.neural
        self.model: NeuralRankerModel | None = None
        self.feature_cols: list[str] = []
        self.training_history: list[dict[str, float]] = []
        self.training_time: float = 0.0
        self.device = torch.device(self.neural_cfg.get("device", "cpu"))

    def fit(self, train_pdf: pd.DataFrame, val_pdf: pd.DataFrame | None = None) -> "NeuralRanker":
        """Train the neural ranking model."""
        from recommender.models.tree_ranker import get_feature_columns

        self.feature_cols = get_feature_columns(train_pdf)
        logger.info("Training neural ranker with %d features", len(self.feature_cols))

        X_train = train_pdf[self.feature_cols].fillna(0).values.astype(np.float32)
        y_train = train_pdf["label"].values.astype(np.float32)

        # Normalize features
        self._mean = X_train.mean(axis=0)
        self._std = X_train.std(axis=0) + 1e-8
        X_train = (X_train - self._mean) / self._std

        arch_cfg = self.neural_cfg.get("architecture", {})
        train_cfg = self.neural_cfg.get("training", {})
        es_cfg = self.neural_cfg.get("early_stopping", {})

        self.model = NeuralRankerModel(
            input_dim=len(self.feature_cols),
            hidden_dims=arch_cfg.get("hidden_dims", [256, 128, 64]),
            dropout=arch_cfg.get("dropout", 0.3),
            batch_norm=arch_cfg.get("batch_norm", True),
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=train_cfg.get("learning_rate", 0.001),
            weight_decay=train_cfg.get("weight_decay", 0.0001),
        )
        criterion = nn.BCELoss()
        epochs = train_cfg.get("epochs", 50)
        batch_size = train_cfg.get("batch_size", 512)
        patience = es_cfg.get("patience", 7)

        train_loader = DataLoader(
            RankingDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
        )

        # Validation
        val_loader = None
        if val_pdf is not None:
            X_val = val_pdf[self.feature_cols].fillna(0).values.astype(np.float32)
            X_val = (X_val - self._mean) / self._std
            y_val = val_pdf["label"].values.astype(np.float32)
            val_loader = DataLoader(
                RankingDataset(X_val, y_val),
                batch_size=batch_size,
                shuffle=False,
            )

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        start = time.time()

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1

            avg_train_loss = train_loss / max(n_batches, 1)
            epoch_metrics = {"epoch": epoch + 1, "train_loss": avg_train_loss}

            # Validation
            if val_loader is not None:
                val_loss = self._evaluate_loss(val_loader, criterion)
                epoch_metrics["val_loss"] = val_loss

                if val_loss < best_val_loss - es_cfg.get("min_delta", 0.0001):
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

            self.training_history.append(epoch_metrics)

            if (epoch + 1) % 5 == 0:
                msg = f"Epoch {epoch + 1}/{epochs} — train_loss={avg_train_loss:.4f}"
                if "val_loss" in epoch_metrics:
                    msg += f", val_loss={epoch_metrics['val_loss']:.4f}"
                logger.info(msg)

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.training_time = time.time() - start
        logger.info("Neural ranker training complete in %.1fs", self.training_time)
        return self

    def _evaluate_loss(self, loader: DataLoader, criterion: nn.Module) -> float:
        """Evaluate loss on a DataLoader."""
        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                pred = self.model(X_batch)
                total_loss += criterion(pred, y_batch).item()
                n += 1
        return total_loss / max(n, 1)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict relevance scores."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        X = df[self.feature_cols].fillna(0).values.astype(np.float32)
        X = (X - self._mean) / self._std

        self.model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(X).to(self.device)
            scores = self.model(tensor).cpu().numpy()
        return scores

    def save(self, config: ProjectConfig | None = None) -> None:
        """Save model, weights, and metadata."""
        cfg = config or get_config()
        out_dir = cfg.artifacts_dir / "neural_ranker"
        out_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), out_dir / "model_weights.pt")
        np.save(out_dir / "feature_mean.npy", self._mean)
        np.save(out_dir / "feature_std.npy", self._std)

        meta = {
            "feature_cols": self.feature_cols,
            "architecture": self.neural_cfg.get("architecture", {}),
            "training_time": self.training_time,
            "training_history": self.training_history,
            "input_dim": len(self.feature_cols),
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Neural ranker saved to %s", out_dir)

    @classmethod
    def load(cls, config: ProjectConfig | None = None) -> "NeuralRanker":
        """Load saved model."""
        cfg = config or get_config()
        path = cfg.artifacts_dir / "neural_ranker"

        ranker = cls(config=cfg)
        with open(path / "metadata.json") as f:
            meta = json.load(f)

        ranker.feature_cols = meta["feature_cols"]
        ranker._mean = np.load(path / "feature_mean.npy")
        ranker._std = np.load(path / "feature_std.npy")

        arch = meta.get("architecture", {})
        ranker.model = NeuralRankerModel(
            input_dim=meta["input_dim"],
            hidden_dims=arch.get("hidden_dims", [256, 128, 64]),
            dropout=arch.get("dropout", 0.3),
            batch_norm=arch.get("batch_norm", True),
        )
        ranker.model.load_state_dict(torch.load(path / "model_weights.pt", map_location="cpu"))
        ranker.model.eval()
        logger.info("Neural ranker loaded from %s", path)
        return ranker
