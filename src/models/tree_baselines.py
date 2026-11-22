"""Tree baseline models (XGBoost, LightGBM, Random Forest).

Implements exact Table 4 hyperparameters from 2611.04660v1.pdf.
Tabular ML only - zero graph/GNN dependencies.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb


def get_xgboost_model(
    task: str = "binary",
    seed: int = 0,
    n_estimators: int = 400,
    max_depth: int = 8,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    tree_method: str = "hist",
    early_stopping_rounds: int = 20,
    n_jobs: int = -1,
    device: str = "cpu",
    **kwargs,
) -> xgb.XGBClassifier:
    """Create XGBoost classifier with Table 4 hyperparameters."""
    if task == "binary":
        objective = "binary:logistic"
        eval_metric = "logloss"
    elif task == "multiclass":
        objective = "multi:softprob"
        eval_metric = "mlogloss"
    else:
        raise ValueError(f"Unknown task: {task}. Expected 'binary' or 'multiclass'.")

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "tree_method": tree_method,
        "early_stopping_rounds": early_stopping_rounds,
        "random_state": seed,
        "objective": objective,
        "eval_metric": eval_metric,
        "n_jobs": n_jobs,
        "device": device,
    }
    params.update(kwargs)
    return xgb.XGBClassifier(**params)


def get_lightgbm_model(
    task: str = "binary",
    seed: int = 0,
    n_estimators: int = 400,
    num_leaves: int = 63,
    max_depth: int = 8,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    n_jobs: int = -1,
    **kwargs,
) -> lgb.LGBMClassifier:
    """Create LightGBM classifier with Table 4 hyperparameters."""
    if task == "binary":
        objective = "binary"
    elif task == "multiclass":
        objective = "multiclass"
    else:
        raise ValueError(f"Unknown task: {task}. Expected 'binary' or 'multiclass'.")

    params = {
        "n_estimators": n_estimators,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "subsample_freq": 1 if subsample < 1.0 else 0,
        "colsample_bytree": colsample_bytree,
        "random_state": seed,
        "objective": objective,
        "n_jobs": n_jobs,
        "verbose": -1,
    }
    params.update(kwargs)
    return lgb.LGBMClassifier(**params)


def get_random_forest_model(
    seed: int = 0,
    n_estimators: int = 100,
    max_samples: float | None = 0.3,
    class_weight: str = "balanced_subsample",
    n_jobs: int = -1,
    **kwargs,
) -> RandomForestClassifier:
    """Create Random Forest classifier with Table 4 hyperparameters."""
    params = {
        "n_estimators": n_estimators,
        "max_samples": max_samples,
        "class_weight": class_weight,
        "random_state": seed,
        "n_jobs": n_jobs,
    }
    params.update(kwargs)
    return RandomForestClassifier(**params)


def train_baseline_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    task: str = "binary",
    seed: int = 0,
    sample_weight_train: np.ndarray | None = None,
    sample_weight_val: np.ndarray | None = None,
    **kwargs,
) -> Any:
    """Instantiate and train a tree baseline model with early stopping on validation.

    Supported model_name: 'xgboost' (or 'xgb'), 'lightgbm' (or 'lgb'),
    'random_forest' (or 'rf').
    """
    canonical_name = model_name.lower().replace("-", "_").replace(" ", "_")

    if canonical_name in ["xgboost", "xgb"]:
        model = get_xgboost_model(task=task, seed=seed, **kwargs)
        fit_kwargs: dict[str, Any] = {"verbose": False}
        if sample_weight_train is not None:
            fit_kwargs["sample_weight"] = sample_weight_train
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            if sample_weight_val is not None:
                fit_kwargs["sample_weight_eval_set"] = [sample_weight_val]
        model.fit(X_train, y_train, **fit_kwargs)
        return model

    elif canonical_name in ["lightgbm", "lgb"]:
        model = get_lightgbm_model(task=task, seed=seed, **kwargs)
        callbacks = [
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        fit_kwargs = {"callbacks": callbacks}
        if sample_weight_train is not None:
            fit_kwargs["sample_weight"] = sample_weight_train
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_X"] = X_val
            fit_kwargs["eval_y"] = y_val
            if sample_weight_val is not None:
                fit_kwargs["eval_sample_weight"] = [sample_weight_val]
        model.fit(X_train, y_train, **fit_kwargs)
        return model

    elif canonical_name in ["random_forest", "rf"]:
        model = get_random_forest_model(seed=seed, **kwargs)
        # Random Forest uses class_weight='balanced_subsample' in constructor
        model.fit(X_train, y_train)
        return model

    else:
        raise ValueError(
            f"Unsupported model: {model_name}. Supported: 'xgboost', 'lightgbm', 'random_forest'."
        )


def predict_baseline_model(
    model: Any,
    X: np.ndarray,
    task: str = "binary",
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions and probabilities from a trained model.

    Returns:
        (y_pred, y_prob)
        - For binary:
            y_pred: 1D array of predicted class (0 or 1) based on threshold
            y_prob: 1D array of positive class probability P(Y=1)
        - For multiclass:
            y_pred: 1D array of predicted class indices
            y_prob: 2D array of class probabilities (N, C)
    """
    # Ensure float32 array
    X_arr = np.asarray(X, dtype=np.float32)

    probs = model.predict_proba(X_arr)

    if task == "binary":
        # Handle 1D or 2D probability outputs
        if probs.ndim == 2:
            if probs.shape[1] == 2:
                pos_prob = probs[:, 1]
            elif probs.shape[1] == 1:
                pos_prob = probs[:, 0]
            else:
                pos_prob = probs[:, -1]
        else:
            pos_prob = probs
        y_pred = (pos_prob >= threshold).astype(np.int64)
        return y_pred, pos_prob.astype(np.float32)

    elif task == "multiclass":
        y_pred = np.argmax(probs, axis=1).astype(np.int64)
        return y_pred, probs.astype(np.float32)

    else:
        raise ValueError(f"Unknown task: {task}. Expected 'binary' or 'multiclass'.")
