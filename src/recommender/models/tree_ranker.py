"""
XGBoost Tree-Based Ranking Model.

Uses XGBoost with rank:pairwise objective for learning-to-rank.
Supports hyperparameter search, feature importance, and MLflow logging.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import ParameterGrid

from recommender.config import ProjectConfig, get_config

logger = logging.getLogger(__name__)

# Features used for ranking (exclude IDs, labels, metadata)
EXCLUDE_COLS = {
    "user_id",
    "news_id",
    "label",
    "impression_id",
    "timestamp",
    "timestamp_unix",
    "history",
    "history_items",
    "history_length",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
    "user_top_category",
    "category",
    "subcategory",
    "item_first_seen_ts",
    "item_last_seen_ts",
    "user_first_interaction_ts",
    "user_last_interaction_ts",
    "source",
    "candidate_score",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get numeric feature columns, excluding IDs and metadata."""
    cols = []
    for c in df.columns:
        if c in EXCLUDE_COLS:
            continue
        if df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int):
            cols.append(c)
    return sorted(cols)


def prepare_ranking_data(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for XGBoost ranking.

    Returns (X, y, group) where group sizes are per impression_id.
    """
    # Sort by impression_id for group-based ranking
    df = df.sort_values("impression_id").reset_index(drop=True)

    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df["label"].values.astype(np.float32)

    # Group sizes (items per impression)
    groups = df.groupby("impression_id", sort=False).size().values

    return X, y, groups


class TreeRanker:
    """XGBoost learning-to-rank model."""

    def __init__(self, config: ProjectConfig | None = None):
        cfg = config or get_config()
        self.xgb_cfg = cfg.xgboost
        self.model: xgb.Booster | None = None
        self.feature_cols: list[str] = []
        self.best_params: dict[str, Any] = {}
        self.feature_importance_: dict[str, float] = {}
        self.training_time: float = 0.0

    def fit(
        self,
        train_pdf: pd.DataFrame,
        val_pdf: pd.DataFrame | None = None,
    ) -> "TreeRanker":
        """
        Train the XGBoost ranking model.

        Parameters
        ----------
        train_pdf : pd.DataFrame
            Training data with features, label, impression_id.
        val_pdf : pd.DataFrame, optional
            Validation data.
        """
        self.feature_cols = get_feature_columns(train_pdf)
        logger.info(
            "Training XGBoost ranker with %d features: %s",
            len(self.feature_cols),
            self.feature_cols[:10],
        )

        X_train, y_train, g_train = prepare_ranking_data(train_pdf, self.feature_cols)

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_cols)
        dtrain.set_group(g_train)

        evals = [(dtrain, "train")]
        if val_pdf is not None:
            X_val, y_val, g_val = prepare_ranking_data(val_pdf, self.feature_cols)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_cols)
            dval.set_group(g_val)
            evals.append((dval, "val"))

        # Hyperparameter search or direct training
        search_cfg = self.xgb_cfg.get("search", {})
        if search_cfg.get("enabled", False):
            self._hyperparameter_search(dtrain, evals, search_cfg)
        else:
            self._train_single(dtrain, evals)

        # Feature importance
        self.feature_importance_ = self.model.get_score(importance_type="gain")
        logger.info(
            "Top features: %s",
            dict(
                sorted(
                    self.feature_importance_.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
            ),
        )

        return self

    def _train_single(self, dtrain: xgb.DMatrix, evals: list) -> None:
        """Train with default hyperparameters."""
        params = {
            "objective": self.xgb_cfg.get("objective", "rank:pairwise"),
            "eval_metric": self.xgb_cfg.get("eval_metric", "ndcg@10"),
            "tree_method": self.xgb_cfg.get("tree_method", "hist"),
            "max_depth": self.xgb_cfg.get("max_depth", 6),
            "learning_rate": self.xgb_cfg.get("learning_rate", 0.1),
            "subsample": self.xgb_cfg.get("subsample", 0.8),
            "colsample_bytree": self.xgb_cfg.get("colsample_bytree", 0.8),
            "min_child_weight": self.xgb_cfg.get("min_child_weight", 5),
            "reg_alpha": self.xgb_cfg.get("reg_alpha", 0.1),
            "reg_lambda": self.xgb_cfg.get("reg_lambda", 1.0),
            "verbosity": 1,
        }
        n_rounds = self.xgb_cfg.get("n_estimators", 300)
        early_stop = self.xgb_cfg.get("early_stopping_rounds", 30)

        start = time.time()
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_rounds,
            evals=evals,
            early_stopping_rounds=early_stop if len(evals) > 1 else None,
            verbose_eval=self.xgb_cfg.get("verbose_eval", 50),
        )
        self.training_time = time.time() - start
        self.best_params = params
        logger.info("Training complete in %.1fs", self.training_time)

    def _hyperparameter_search(self, dtrain: xgb.DMatrix, evals: list, search_cfg: dict) -> None:
        """Grid search for best hyperparameters."""
        param_grid = search_cfg.get("param_grid", {})
        grid = list(ParameterGrid(param_grid))
        logger.info("Hyperparameter search: %d combinations", len(grid))

        best_score = -np.inf
        base_params = {
            "objective": self.xgb_cfg.get("objective", "rank:pairwise"),
            "eval_metric": self.xgb_cfg.get("eval_metric", "ndcg@10"),
            "tree_method": self.xgb_cfg.get("tree_method", "hist"),
            "verbosity": 0,
        }

        for i, combo in enumerate(grid):
            params = {**base_params, **combo}
            n_rounds = combo.pop("n_estimators", self.xgb_cfg.get("n_estimators", 300))

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=n_rounds,
                evals=evals,
                early_stopping_rounds=30 if len(evals) > 1 else None,
                verbose_eval=0,
            )

            score = model.best_score if hasattr(model, "best_score") else 0.0
            logger.info("  [%d/%d] %s → score=%.4f", i + 1, len(grid), combo, score)

            if score > best_score:
                best_score = score
                self.model = model
                self.best_params = params

        logger.info("Best params: %s (score=%.4f)", self.best_params, best_score)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict relevance scores."""
        if self.model is None:
            raise RuntimeError("Model not trained")
        X = df[self.feature_cols].fillna(0).values.astype(np.float32)
        dmat = xgb.DMatrix(X, feature_names=self.feature_cols)
        return self.model.predict(dmat)

    def save(self, config: ProjectConfig | None = None) -> None:
        """Save model and metadata."""
        cfg = config or get_config()
        out_dir = cfg.artifacts_dir / "tree_ranker"
        out_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_model(str(out_dir / "model.json"))
        meta = {
            "feature_cols": self.feature_cols,
            "best_params": {k: str(v) for k, v in self.best_params.items()},
            "training_time": self.training_time,
            "feature_importance": self.feature_importance_,
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Tree ranker saved to %s", out_dir)

    @classmethod
    def load(cls, config: ProjectConfig | None = None) -> "TreeRanker":
        """Load saved model."""
        cfg = config or get_config()
        path = cfg.artifacts_dir / "tree_ranker"
        ranker = cls(config=cfg)
        ranker.model = xgb.Booster()
        ranker.model.load_model(str(path / "model.json"))
        with open(path / "metadata.json") as f:
            meta = json.load(f)
        ranker.feature_cols = meta["feature_cols"]
        ranker.best_params = meta["best_params"]
        ranker.feature_importance_ = meta.get("feature_importance", {})
        logger.info("Tree ranker loaded from %s", path)
        return ranker
