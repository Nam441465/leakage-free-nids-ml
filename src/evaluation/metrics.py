"""Evaluation metrics for binary and multiclass network intrusion detection.

Implements all required metrics from 2611.04660v1.pdf:
- Binary: F1, MCC, PR-AUC, FPR at threshold 0.5, Recall at 0.1% FPR and 1% FPR.
- Multiclass: Macro-F1 (over classes present in test), Weighted F1, Accuracy,
  Per-class F1, row-normalized confusion matrix.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    average_precision_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
)


def compute_recall_at_fixed_fpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_fpr: float = 0.001,
) -> float:
    """Compute empirical recall (TPR) when false positive rate (FPR) <= target_fpr.

    Parameters:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probability of positive class (1).
        target_fpr: Target maximum false positive rate (e.g. 0.001 for 0.1% FPR,
          0.01 for 1% FPR).

    Returns:
        float: Maximum TPR achieved at or below the target FPR threshold.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float32)

    # Check for single class edge cases
    if len(np.unique(y_true_arr)) < 2:
        return 0.0

    fpr_arr, tpr_arr, _ = roc_curve(y_true_arr, y_prob_arr)

    # Find points on the ROC curve where FPR <= target_fpr
    valid_mask = fpr_arr <= target_fpr
    if not np.any(valid_mask):
        return 0.0

    return float(np.max(tpr_arr[valid_mask]))


def evaluate_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute comprehensive binary classification metrics.

    Metrics computed:
    - f1: F1-score at decision threshold (default 0.5)
    - mcc: Matthews Correlation Coefficient
    - pr_auc: Area Under the Precision-Recall Curve (Average Precision)
    - fpr_at_threshold: False Positive Rate at specified decision threshold
    - recall_at_0.1_fpr: Recall at 0.1% (0.001) FPR limit
    - recall_at_1.0_fpr: Recall at 1.0% (0.01) FPR limit
    - recall: Standard recall at decision threshold
    - precision: Standard precision at decision threshold
    - accuracy: Accuracy at decision threshold
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float32)

    # Handle 2D probability outputs
    if y_prob_arr.ndim == 2:
        if y_prob_arr.shape[1] == 2:
            y_prob_arr = y_prob_arr[:, 1]
        elif y_prob_arr.shape[1] == 1:
            y_prob_arr = y_prob_arr[:, 0]
        else:
            y_prob_arr = y_prob_arr[:, -1]

    y_pred = (y_prob_arr >= threshold).astype(np.int64)

    # F1 score
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0.0))

    # Matthews Correlation Coefficient
    try:
        mcc = float(matthews_corrcoef(y_true_arr, y_pred))
    except Exception:
        mcc = 0.0

    # PR-AUC (Average Precision)
    try:
        pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    except Exception:
        pr_auc = 0.0

    # False Positive Rate at threshold 0.5
    fp = np.sum((y_pred == 1) & (y_true_arr == 0))
    tn = np.sum((y_pred == 0) & (y_true_arr == 0))
    total_neg = fp + tn
    fpr_at_threshold = float(fp / total_neg) if total_neg > 0 else 0.0

    # True Positive Rate (Recall) and Precision at threshold
    tp = np.sum((y_pred == 1) & (y_true_arr == 1))
    fn = np.sum((y_pred == 0) & (y_true_arr == 1))
    total_pos = tp + fn
    recall = float(tp / total_pos) if total_pos > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    accuracy = float((tp + tn) / len(y_true_arr)) if len(y_true_arr) > 0 else 0.0

    # Recall at 0.1% and 1.0% FPR
    recall_01 = compute_recall_at_fixed_fpr(
        y_true_arr, y_prob_arr, target_fpr=0.001
    )
    recall_10 = compute_recall_at_fixed_fpr(y_true_arr, y_prob_arr, target_fpr=0.01)

    return {
        "f1": f1,
        "mcc": mcc,
        "pr_auc": pr_auc,
        "fpr_at_0.5": fpr_at_threshold,
        "recall_at_0.1_fpr": recall_01,
        "recall_at_1.0_fpr": recall_10,
        "recall": recall,
        "precision": precision,
        "accuracy": accuracy,
    }


def evaluate_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    all_classes: Sequence[int] | None = None,
    class_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Compute comprehensive multiclass classification metrics.

    Metrics computed:
    - macro_f1: Macro-F1 strictly across classes present in y_true test set.
    - weighted_f1: Weighted F1-score.
    - accuracy: Classification accuracy.
    - per_class_f1: Dictionary mapping each class (or name) to its F1 score.
    - confusion_matrix_normalized: Row-normalized confusion matrix (recall per true class).
    - confusion_matrix_raw: Raw integer confusion matrix.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)

    # Classes present in test set
    classes_present = np.unique(y_true_arr)

    # Macro-F1 strictly over classes present in y_true
    macro_f1 = float(
        f1_score(
            y_true_arr,
            y_pred_arr,
            labels=classes_present,
            average="macro",
            zero_division=0.0,
        )
    )

    # Weighted F1
    weighted_f1 = float(
        f1_score(
            y_true_arr,
            y_pred_arr,
            labels=classes_present,
            average="weighted",
            zero_division=0.0,
        )
    )

    # Accuracy
    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))

    # All classes for confusion matrix and per-class metrics
    if all_classes is None:
        all_classes_arr = np.unique(np.concatenate([classes_present, np.unique(y_pred_arr)]))
    else:
        all_classes_arr = np.asarray(all_classes, dtype=np.int64)

    # Per-class F1
    per_class_scores = f1_score(
        y_true_arr,
        y_pred_arr,
        labels=all_classes_arr,
        average=None,
        zero_division=0.0,
    )
    per_class_dict = {}
    for c_id, score in zip(all_classes_arr, per_class_scores):
        key_name = (
            class_names.get(int(c_id), str(c_id)) if class_names else str(c_id)
        )
        per_class_dict[key_name] = float(score)

    # Confusion matrix
    cm_raw = confusion_matrix(y_true_arr, y_pred_arr, labels=all_classes_arr)
    row_sums = cm_raw.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.divide(
            cm_raw.astype(np.float64),
            row_sums,
            out=np.zeros_like(cm_raw, dtype=np.float64),
            where=row_sums != 0,
        )

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "per_class_f1": per_class_dict,
        "confusion_matrix_raw": cm_raw.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "classes_evaluated": all_classes_arr.tolist(),
    }
