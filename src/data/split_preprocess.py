"""Leakage-free dataset splitting and preprocessing module.

Adheres strictly to the protocol from 2611.04660v1.pdf and AGENTS.md:
1. Excludes Destination Port by default (with option to include for ablation).
2. Stratified 60% train / 20% val / 20% test per seed s in {0, 1, 2}.
3. Preprocessing (signed log selection, mean, std, clipping) is fitted ONLY on train.
4. Class weights w_c = min(N / (C * n_c), 100.0) computed ONLY from train split.
5. All feature arrays are strictly float32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DESTINATION_PORT_NAMES = {
    "destination port",
    "destination_port",
    "destinationport",
    "l4_dst_port",
    "l4_destination_port",
}


def compute_class_weights(y_train: np.ndarray | Sequence[int]) -> dict[int, float]:
    """Compute w_c = min(N / (C * n_c), 100.0) strictly from training labels."""
    y_arr = np.asarray(y_train)
    classes, counts = np.unique(y_arr, return_counts=True)
    num_classes = len(classes)
    total_samples = len(y_arr)
    weights: dict[int, float] = {}
    for c, count in zip(classes, counts):
        raw_w = total_samples / (num_classes * float(count))
        weights[int(c)] = float(min(raw_w, 100.0))
    return weights


def map_sample_weights(
    y: np.ndarray | Sequence[int],
    class_weights: dict[int, float],
) -> np.ndarray:
    """Map integer class labels to float32 sample weights."""
    y_arr = np.asarray(y)
    weights = np.empty(len(y_arr), dtype=np.float32)
    for i, val in enumerate(y_arr):
        weights[i] = class_weights.get(int(val), 1.0)
    return weights


class LeakageFreePreprocessor:
    """Preprocessor that fits signed-log feature selection, z-score parameters,

    and class weights ONLY on the training split, then transforms train/val/test
    without any data leakage.
    """

    def __init__(self, clip_range: tuple[float, float] = (-10.0, 10.0)) -> None:
        self.clip_range = clip_range
        self.log_cols_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.class_weights_: dict[int, float] = {}
        self.n_features_: int = 0
        self.is_fitted_: bool = False

    def fit(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series | None = None,
    ) -> LeakageFreePreprocessor:
        """Fit preprocessor parameters strictly on training split."""
        if isinstance(X_train, pd.DataFrame):
            X_arr = X_train.to_numpy(dtype=np.float32)
        else:
            X_arr = np.asarray(X_train, dtype=np.float32)

        n_samples, n_features = X_arr.shape
        self.n_features_ = n_features

        # 1. Identify columns j with train_min >= 0 and train_p99 > 10
        train_min = np.nanmin(X_arr, axis=0)
        train_p99 = np.nanpercentile(X_arr, 99.0, axis=0)
        log_cols = np.where((train_min >= 0.0) & (train_p99 > 10.0))[0]
        self.log_cols_ = log_cols

        # 2. Apply signed log transform on training copy to compute mean and std
        X_transformed = X_arr.copy()
        if len(self.log_cols_) > 0:
            cols = self.log_cols_
            X_transformed[:, cols] = np.sign(X_transformed[:, cols]) * np.log1p(
                np.abs(X_transformed[:, cols])
            )

        # 3. Compute train mean and std; guard std == 0 -> 1.0
        mean = np.nanmean(X_transformed, axis=0)
        std = np.nanstd(X_transformed, axis=0)
        std = np.where((std == 0.0) | np.isnan(std) | (std < 1e-8), 1.0, std)

        self.mean_ = mean.astype(np.float32)
        self.std_ = std.astype(np.float32)

        # 4. Compute class weights on train split if y_train is provided
        if y_train is not None:
            self.class_weights_ = compute_class_weights(y_train)

        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Transform features using train-fitted parameters and clip to [-10, 10]."""
        if not self.is_fitted_ or self.mean_ is None or self.std_ is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=np.float32, copy=True)
        else:
            X_arr = np.array(X, dtype=np.float32, copy=True)

        # Apply signed log to columns identified during training
        if self.log_cols_ is not None and len(self.log_cols_) > 0:
            cols = self.log_cols_
            X_arr[:, cols] = np.sign(X_arr[:, cols]) * np.log1p(np.abs(X_arr[:, cols]))

        # Standardize using train mean and train std
        X_arr = (X_arr - self.mean_) / self.std_

        # Clip to specified range (default [-10, 10])
        np.clip(X_arr, self.clip_range[0], self.clip_range[1], out=X_arr)
        return X_arr.astype(np.float32)

    def fit_transform(
        self,
        X_train: np.ndarray | pd.DataFrame,
        y_train: np.ndarray | pd.Series | None = None,
    ) -> np.ndarray:
        """Fit on train and transform train."""
        self.fit(X_train, y_train=y_train)
        return self.transform(X_train)

    def get_sample_weights(self, y: np.ndarray | Sequence[int]) -> np.ndarray:
        """Map class labels to train-derived class weights w_c."""
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        return map_sample_weights(y, self.class_weights_)


@dataclass
class DatasetSplits:
    """Container for 60/20/20 train/val/test splits."""

    X_train: np.ndarray
    y_train_multiclass: np.ndarray
    y_train_binary: np.ndarray
    sample_weight_train: np.ndarray
    sample_weight_train_binary: np.ndarray

    X_val: np.ndarray
    y_val_multiclass: np.ndarray
    y_val_binary: np.ndarray
    sample_weight_val: np.ndarray
    sample_weight_val_binary: np.ndarray

    X_test: np.ndarray
    y_test_multiclass: np.ndarray
    y_test_binary: np.ndarray
    sample_weight_test: np.ndarray
    sample_weight_test_binary: np.ndarray

    feature_names: list[str]
    label_mapping: dict[str, int]
    inverse_label_mapping: dict[int, str]
    preprocessor: LeakageFreePreprocessor


def find_destination_port_columns(columns: Sequence[str]) -> list[str]:
    """Find any column name matching destination port patterns."""
    matched = []
    for col in columns:
        col_clean = str(col).strip().lower().replace("-", "_").replace(" ", "_")
        if (
            col_clean in DESTINATION_PORT_NAMES
            or "dst_port" in col_clean
            or "destination_port" in col_clean
        ):
            matched.append(col)
    return matched


def prepare_splits(
    df: pd.DataFrame,
    seed: int = 0,
    label_col: str | None = None,
    binary_label_col: str | None = None,
    exclude_destination_port: bool = True,
    benign_label_name: str = "Benign",
) -> DatasetSplits:
    """Split data into 60% train / 20% val / 20% test stratified on multiclass label.

    Preprocesses features ONLY on the train split and applies the transform
    consistently across train, val, and test.
    """
    # Identify label column if not provided (read directly from df without copying/mutating)
    if label_col is None:
        for candidate in ["label_multiclass", "Label", "label", "Attack", "attack"]:
            if candidate in df.columns:
                label_col = candidate
                break
    if label_col is None:
        raise ValueError("Could not automatically identify label column in DataFrame.")

    # Identify or construct binary label column
    if binary_label_col is None:
        for candidate in ["label_binary", "Binary_Label", "binary_label"]:
            if candidate in df.columns and candidate != label_col:
                binary_label_col = candidate
                break

    # Build numeric multiclass labels and mapping
    raw_labels = df[label_col].to_numpy()
    if pd.api.types.is_numeric_dtype(df[label_col]):
        unique_classes = sorted(list(np.unique(raw_labels)))
        label_mapping = {str(c): int(c) for c in unique_classes}
        inverse_label_mapping = {int(c): str(c) for c in unique_classes}
        y_multiclass = raw_labels.astype(np.int64, copy=True)
    else:
        # String labels: ensure Benign is 0 if present
        unique_classes = sorted(list(set(str(x) for x in raw_labels)))
        # Put benign first if present
        benign_matches = [
            c for c in unique_classes if c.lower() == benign_label_name.lower()
        ]
        if benign_matches:
            benign_actual = benign_matches[0]
            ordered_classes = [benign_actual] + [
                c for c in unique_classes if c != benign_actual
            ]
        else:
            ordered_classes = unique_classes

        label_mapping = {c: idx for idx, c in enumerate(ordered_classes)}
        inverse_label_mapping = {idx: c for idx, c in enumerate(ordered_classes)}
        y_multiclass = np.array([label_mapping[str(x)] for x in raw_labels], dtype=np.int64)

    # Build binary labels (0: Benign, 1: Attack)
    if binary_label_col and binary_label_col in df.columns:
        raw_binary = df[binary_label_col].to_numpy()
        if pd.api.types.is_numeric_dtype(df[binary_label_col]):
            y_binary = raw_binary.astype(np.int64, copy=True)
        else:
            y_binary = np.array(
                [
                    0 if str(x).lower() == benign_label_name.lower() else 1
                    for x in raw_binary
                ],
                dtype=np.int64,
            )
    else:
        # Derive from multiclass: class 0 is Benign (or matching benign name)
        benign_id = label_mapping.get(
            benign_label_name,
            0 if 0 in inverse_label_mapping else None,
        )
        if benign_id is None:
            benign_id = 0
        y_binary = (y_multiclass != benign_id).astype(np.int64)

    # Drop label columns from features via column-list selection (no full DataFrame copy)
    cols_to_drop_set: set[str] = {label_col}
    if binary_label_col and binary_label_col in df.columns:
        cols_to_drop_set.add(binary_label_col)

    # Exclude Destination Port if requested
    if exclude_destination_port:
        dst_port_cols = find_destination_port_columns(df.columns)
        cols_to_drop_set.update(dst_port_cols)

    feature_names = [c for c in df.columns if c not in cols_to_drop_set]
    X_raw = df[feature_names].to_numpy(dtype=np.float32)

    gc.collect()

    # Stratified split: 60% train / 40% temp
    indices = np.arange(len(X_raw))
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=0.6,
        random_state=seed,
        stratify=y_multiclass,
    )

    # Stratified split: 50% val / 50% test from temp (yields 20% val / 20% test overall)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=seed,
        stratify=y_multiclass[temp_idx],
    )

    # Extract splits
    X_train_raw = X_raw[train_idx]
    y_train_multi = y_multiclass[train_idx]
    y_train_bin = y_binary[train_idx]

    X_val_raw = X_raw[val_idx]
    y_val_multi = y_multiclass[val_idx]
    y_val_bin = y_binary[val_idx]

    X_test_raw = X_raw[test_idx]
    y_test_multi = y_multiclass[test_idx]
    y_test_bin = y_binary[test_idx]

    del X_raw, indices, train_idx, temp_idx, val_idx, test_idx
    gc.collect()

    # Fit preprocessor ONLY on train split
    preprocessor = LeakageFreePreprocessor(clip_range=(-10.0, 10.0))
    X_train = preprocessor.fit_transform(X_train_raw, y_train=y_train_multi)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    del X_train_raw, X_val_raw, X_test_raw
    gc.collect()

    # Compute sample weights from train class weights (multiclass C and binary C=2)
    sample_weight_train = preprocessor.get_sample_weights(y_train_multi)
    sample_weight_val = preprocessor.get_sample_weights(y_val_multi)
    sample_weight_test = preprocessor.get_sample_weights(y_test_multi)

    binary_class_weights = compute_class_weights(y_train_bin)
    sample_weight_train_binary = map_sample_weights(y_train_bin, binary_class_weights)
    sample_weight_val_binary = map_sample_weights(y_val_bin, binary_class_weights)
    sample_weight_test_binary = map_sample_weights(y_test_bin, binary_class_weights)

    return DatasetSplits(
        X_train=X_train,
        y_train_multiclass=y_train_multi,
        y_train_binary=y_train_bin,
        sample_weight_train=sample_weight_train,
        sample_weight_train_binary=sample_weight_train_binary,
        X_val=X_val,
        y_val_multiclass=y_val_multi,
        y_val_binary=y_val_bin,
        sample_weight_val=sample_weight_val,
        sample_weight_val_binary=sample_weight_val_binary,
        X_test=X_test,
        y_test_multiclass=y_test_multi,
        y_test_binary=y_test_bin,
        sample_weight_test=sample_weight_test,
        sample_weight_test_binary=sample_weight_test_binary,
        feature_names=feature_names,
        label_mapping=label_mapping,
        inverse_label_mapping=inverse_label_mapping,
        preprocessor=preprocessor,
    )
