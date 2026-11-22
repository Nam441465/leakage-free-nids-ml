"""Live demo and inference verification script for saved NIDS ML models.

Loads saved model bundles (model + train-fitted preprocessor + metadata) from
`saved_models/<dataset>/<task>/seed_<seed>/`, reconstructs the exact
deterministic 60/20/20 test split matching `prepare_splits`, transforms ONLY
the test rows using the saved preprocessor (zero refitting, zero data leakage),
evaluates metrics, and verifies consistency against `*_detailed_results.csv`.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluation.metrics import (
    evaluate_binary_metrics,
    evaluate_multiclass_metrics,
)
from src.models.persistence import load_model_bundle
from src.models.tree_baselines import predict_baseline_model

DATASET_PARQUET_MAP: dict[str, str] = {
    "cicids2017": "cicids2017.parquet",
    "nf_unsw": "nf_unsw_nb15_v2.parquet",
}

ALL_MODELS: list[str] = ["xgboost", "lightgbm", "random_forest"]


def _resolve_parquet_path(dataset: str, data_dir: Path) -> Path:
    """Resolve processed parquet path for a dataset name."""
    filename = DATASET_PARQUET_MAP.get(dataset, f"{dataset}.parquet")
    data_path = data_dir / filename
    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed dataset file not found at {data_path} for dataset='{dataset}'."
        )
    return data_path


def _extract_labels_and_test_indices(
    df: pd.DataFrame,
    seed: int,
    label_mapping: dict[str, int],
    inverse_label_mapping: dict[int, str],
    benign_label_name: str = "Benign",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct exact deterministic 60/20/20 test indices matching `prepare_splits`."""
    label_col: str | None = None
    for candidate in ["label_multiclass", "Label", "label", "Attack", "attack"]:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        raise ValueError("Could not automatically identify label column in DataFrame.")

    binary_label_col: str | None = None
    for candidate in ["label_binary", "Binary_Label", "binary_label"]:
        if candidate in df.columns and candidate != label_col:
            binary_label_col = candidate
            break

    raw_labels = df[label_col].to_numpy()
    if pd.api.types.is_numeric_dtype(df[label_col]):
        y_multiclass = raw_labels.astype(np.int64, copy=True)
    else:
        y_multiclass = np.array(
            [label_mapping[str(x)] for x in raw_labels], dtype=np.int64
        )

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
        benign_id = label_mapping.get(
            benign_label_name,
            0 if 0 in inverse_label_mapping else None,
        )
        if benign_id is None:
            benign_id = 0
        y_binary = (y_multiclass != benign_id).astype(np.int64)

    # Exact 60/20/20 stratified split matching src.data.split_preprocess.prepare_splits
    indices = np.arange(len(df))
    _, temp_idx = train_test_split(
        indices,
        train_size=0.6,
        random_state=seed,
        stratify=y_multiclass,
    )
    _, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=seed,
        stratify=y_multiclass[temp_idx],
    )

    y_test_multiclass = y_multiclass[test_idx]
    y_test_binary = y_binary[test_idx]

    del indices, temp_idx, y_multiclass, y_binary
    return test_idx, y_test_binary, y_test_multiclass


def _format_summary_table(result: dict[str, Any]) -> str:
    """Format a clean human-readable summary table for live evaluation output."""
    task = result["task"]
    metrics = result["metrics"]
    comp = result["reported_comparison"]

    lines = [
        "=" * 72,
        f"Live Demo Evaluation | dataset={result['dataset']} | task={task} | "
        f"model={result['model']} | seed={result['seed']}",
        "-" * 72,
        f"  {'Model File':<26}: {result['model_path']}",
        f"  {'Model Size (MB)':<26}: {result['model_size_mb']:.3f} MB",
        f"  {'Test Samples':<26}: {result['n_test_samples']:,}",
        f"  {'Inference Time':<26}: {result['inference_time_sec']:.4f} s",
        "-" * 72,
    ]

    if task == "binary":
        lines.extend(
            [
                f"  {'F1 (%)':<26}: {metrics['f1'] * 100.0:.2f}%",
                f"  {'MCC':<26}: {metrics['mcc']:.3f}",
                f"  {'PR-AUC (%)':<26}: {metrics['pr_auc'] * 100.0:.2f}%",
                f"  {'FPR @ 0.5 (%)':<26}: {metrics['fpr_at_0.5'] * 100.0:.4f}%",
                f"  {'Recall @ 0.1% FPR (%)':<26}: {metrics['recall_at_0.1_fpr'] * 100.0:.2f}%",
                f"  {'Recall @ 1.0% FPR (%)':<26}: {metrics['recall_at_1.0_fpr'] * 100.0:.2f}%",
            ]
        )
    else:
        lines.extend(
            [
                f"  {'Macro-F1 (%)':<26}: {metrics['macro_f1'] * 100.0:.2f}%",
                f"  {'Weighted-F1 (%)':<26}: {metrics['weighted_f1'] * 100.0:.2f}%",
                f"  {'Accuracy (%)':<26}: {metrics['accuracy'] * 100.0:.2f}%",
            ]
        )
        per_class = metrics.get("per_class_f1", {})
        if per_class:
            lines.append("  Per-Class F1 (%):")
            for cls_name, cls_score in per_class.items():
                lines.append(f"    - {cls_name:<22}: {cls_score * 100.0:.2f}%")

    lines.append("-" * 72)
    if comp is not None:
        status = "MATCH (< 1e-6)" if comp["matched"] else "MISMATCH (>= 1e-6)"
        lines.extend(
            [
                f"  {'CSV Verification Metric':<26}: {comp['metric']}",
                f"  {'Reported (CSV)':<26}: {comp['reported_value']:.8f}",
                f"  {'Live (Saved Bundle)':<26}: {comp['live_value']:.8f}",
                f"  {'Abs Diff':<26}: {comp['abs_diff']:.3e}  [{status}]",
            ]
        )
    else:
        lines.append(f"  {'CSV Verification':<26}: N/A (no matching CSV row found)")
    lines.append("=" * 72)
    return "\n".join(lines)


def evaluate_saved_model(
    dataset: str,
    task: str,
    model_name: str,
    seed: int = 0,
    models_dir: Path = Path("saved_models"),
    data_dir: Path = Path("data/processed"),
    results_dir: Path | None = Path("results/tables"),
) -> dict[str, Any]:
    """Load a saved model bundle and run leakage-free evaluation on the test split.

    Parameters:
        dataset: Dataset identifier ('cicids2017' or 'nf_unsw').
        task: Task type ('binary' or 'multiclass').
        model_name: Model name ('xgboost', 'lightgbm', 'random_forest').
        seed: Random seed matching the saved split.
        models_dir: Root directory containing saved model bundles.
        data_dir: Directory containing processed parquet files.
        results_dir: Optional directory containing `<dataset>_detailed_results.csv`.

    Returns:
        Dictionary containing metrics, timing, file size, paths, and CSV comparison.
    """
    if task not in {"binary", "multiclass"}:
        raise ValueError(f"Unsupported task: '{task}'. Expected 'binary' or 'multiclass'.")

    models_root = Path(models_dir)
    data_root = Path(data_dir)
    results_root = Path(results_dir) if results_dir is not None else None

    # 1. Load saved model + train-fitted preprocessor + metadata
    bundle = load_model_bundle(
        models_dir=models_root,
        dataset=dataset,
        task=task,
        seed=seed,
        model_name=model_name,
    )
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    metadata = bundle["metadata"]
    model_path = Path(bundle["model_path"])
    preprocessor_path = Path(bundle["preprocessor_path"])

    feature_names: list[str] = list(metadata["feature_names"])
    label_mapping: dict[str, int] = dict(metadata["label_mapping"])
    inverse_label_mapping: dict[int, str] = {
        int(k): str(v) for k, v in metadata["inverse_label_mapping"].items()
    }

    # 2. Load processed parquet and reconstruct exact deterministic test indices
    data_path = _resolve_parquet_path(dataset, data_root)
    df = pd.read_parquet(data_path)

    test_idx, y_test_binary, y_test_multiclass = _extract_labels_and_test_indices(
        df=df,
        seed=seed,
        label_mapping=label_mapping,
        inverse_label_mapping=inverse_label_mapping,
    )

    # 3. Extract ONLY test rows; NEVER touch train/val features or refit preprocessor
    X_test_raw = df[feature_names].iloc[test_idx].to_numpy(dtype=np.float32)
    y_test = y_test_binary if task == "binary" else y_test_multiclass
    n_test_samples = int(len(y_test))

    # Free full DataFrame immediately to keep RAM low
    del df, test_idx, y_test_binary, y_test_multiclass
    gc.collect()

    # Transform using the SAVED train-fitted preprocessor ONLY
    X_test = preprocessor.transform(X_test_raw)
    del X_test_raw
    gc.collect()

    # 4. Run inference and measure latency
    t_start = time.perf_counter()
    y_pred, y_prob = predict_baseline_model(model, X_test, task=task)
    inference_time_sec = float(time.perf_counter() - t_start)

    # 5. Compute full evaluation metrics
    if task == "binary":
        metrics: dict[str, Any] = evaluate_binary_metrics(y_test, y_prob)
        primary_metric_name = "f1"
    else:
        all_classes = sorted(inverse_label_mapping.keys())
        metrics = evaluate_multiclass_metrics(
            y_test,
            y_pred,
            y_prob=y_prob,
            all_classes=all_classes,
            class_names=inverse_label_mapping,
        )
        primary_metric_name = "macro_f1"

    # 6. Compare against reported detailed results CSV if available
    reported_comparison: dict[str, Any] | None = None
    if results_root is not None:
        csv_path = results_root / f"{dataset}_detailed_results.csv"
        if csv_path.exists():
            detailed_df = pd.read_csv(csv_path)
            required_cols = {"dataset", "task", "seed", "model", primary_metric_name}
            if required_cols.issubset(detailed_df.columns):
                mask = (
                    (detailed_df["dataset"] == dataset)
                    & (detailed_df["task"] == task)
                    & (detailed_df["seed"] == seed)
                    & (detailed_df["model"] == model_name)
                )
                matched_rows = detailed_df.loc[mask]
                if not matched_rows.empty:
                    reported_value = float(matched_rows.iloc[-1][primary_metric_name])
                    live_value = float(metrics[primary_metric_name])
                    abs_diff = float(abs(live_value - reported_value))
                    reported_comparison = {
                        "metric": primary_metric_name,
                        "reported_value": reported_value,
                        "live_value": live_value,
                        "abs_diff": abs_diff,
                        "matched": bool(abs_diff < 1e-6),
                    }
            del detailed_df

    model_size_mb = float(model_path.stat().st_size / (1024.0 * 1024.0))

    result: dict[str, Any] = {
        "dataset": dataset,
        "task": task,
        "model": model_name,
        "seed": seed,
        "n_test_samples": n_test_samples,
        "inference_time_sec": inference_time_sec,
        "model_size_mb": model_size_mb,
        "model_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
        "metrics": metrics,
        "reported_comparison": reported_comparison,
    }

    print(_format_summary_table(result))

    # Release model and arrays from RAM before returning
    del model, preprocessor, X_test, y_test, y_pred, y_prob
    gc.collect()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live evaluation of saved leakage-free NIDS tabular ML models."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nf_unsw",
        choices=["nf_unsw", "cicids2017"],
        help="Dataset to evaluate",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="binary",
        choices=["binary", "multiclass"],
        help="Classification task",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["xgboost", "lightgbm", "random_forest", "all"],
        help="Model to evaluate (or 'all' for sequential evaluation of all 3 models)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed of the saved model bundle",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="saved_models",
        help="Root directory of saved model bundles",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed parquet files",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/tables",
        help="Directory containing *_detailed_results.csv for verification",
    )

    args = parser.parse_args()
    models_to_run = ALL_MODELS if args.model == "all" else [args.model]

    for model_name in models_to_run:
        res = evaluate_saved_model(
            dataset=args.dataset,
            task=args.task,
            model_name=model_name,
            seed=args.seed,
            models_dir=Path(args.models_dir),
            data_dir=Path(args.data_dir),
            results_dir=Path(args.results_dir) if args.results_dir else None,
        )
        del res
        gc.collect()


if __name__ == "__main__":
    main()
