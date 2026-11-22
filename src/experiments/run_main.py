"""Main benchmark runner for Tabular ML models on NIDS datasets.

Evaluates XGBoost, LightGBM, and Random Forest across seeds {0, 1, 2}
on binary and multiclass intrusion detection tasks without data leakage.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from src.data.split_preprocess import prepare_splits
from src.models.persistence import save_model_artifact, save_seed_artifacts
from src.models.tree_baselines import train_baseline_model, predict_baseline_model
from src.evaluation.metrics import evaluate_binary_metrics, evaluate_multiclass_metrics
from src.evaluation.stats import compute_mean_std, paired_t_test_with_holm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_main")


def run_benchmark_for_dataset(
    dataset_name: str,
    data_path: Path,
    tasks: list[str],
    models: list[str],
    seeds: list[int],
    output_dir: Path,
    exclude_destination_port: bool = True,
    sample_size: int | None = None,
    save_models: bool = False,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute main benchmark for a specific dataset across tasks, seeds, and models."""
    logger.info("Loading dataset: %s from %s", dataset_name, data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    if sample_size is not None and len(df) > sample_size:
        logger.info("Downsampling %s to %d rows for quick run", dataset_name, sample_size)
        label_col = "label_multiclass" if "label_multiclass" in df.columns else (
            "Label" if "Label" in df.columns else "label"
        )
        df = df.groupby(label_col, group_keys=False).apply(
            lambda x: x.sample(
                min(len(x), max(2, int(sample_size * len(x) / len(df)))),
                random_state=42,
            )
        ).reset_index(drop=True)

    results_records: list[dict[str, Any]] = []
    effective_models_dir = (
        Path(models_dir) if models_dir is not None else Path("saved_models")
    )

    for task in tasks:
        logger.info("=== Running Task: %s for %s ===", task.upper(), dataset_name)
        # Store per-model seed values for stats
        model_seed_metrics: dict[str, dict[str, list[float]]] = {
            m: {} for m in models
        }
        model_seed_per_class_f1: dict[str, dict[str, list[float]]] = {
            m: {} for m in models
        }
        model_seed_cm_raw: dict[str, list[np.ndarray]] = {
            m: [] for m in models
        }
        class_order: list[str] | None = None

        for seed in seeds:
            logger.info("Preparing splits for seed=%d...", seed)
            splits = prepare_splits(
                df,
                seed=seed,
                exclude_destination_port=exclude_destination_port,
            )

            if save_models:
                seed_dir = save_seed_artifacts(
                    models_dir=effective_models_dir,
                    dataset=dataset_name,
                    task=task,
                    seed=seed,
                    preprocessor=splits.preprocessor,
                    feature_names=splits.feature_names,
                    label_mapping=splits.label_mapping,
                    inverse_label_mapping=splits.inverse_label_mapping,
                    exclude_destination_port=exclude_destination_port,
                )
                logger.info("Saved seed preprocessor and metadata to %s", seed_dir)

            # Select target labels and task-specific class weights (C=2 for binary, C=num_classes for multiclass)
            if task == "binary":
                y_train = splits.y_train_binary
                y_val = splits.y_val_binary
                y_test = splits.y_test_binary
                sw_train = splits.sample_weight_train_binary
                sw_val = splits.sample_weight_val_binary
                all_classes: list[int] | None = None
            else:
                y_train = splits.y_train_multiclass
                y_val = splits.y_val_multiclass
                y_test = splits.y_test_multiclass
                sw_train = splits.sample_weight_train
                sw_val = splits.sample_weight_val
                all_classes = sorted(splits.inverse_label_mapping.keys())
                if class_order is None:
                    class_order = [splits.inverse_label_mapping[c] for c in all_classes]

            for model_name in models:
                logger.info("Training %s (task=%s, seed=%d)...", model_name, task, seed)
                model = train_baseline_model(
                    model_name=model_name,
                    X_train=splits.X_train,
                    y_train=y_train,
                    X_val=splits.X_val,
                    y_val=y_val,
                    task=task,
                    seed=seed,
                    sample_weight_train=sw_train,
                    sample_weight_val=sw_val,
                )

                logger.info("Evaluating %s on test split...", model_name)
                y_pred, y_prob = predict_baseline_model(model, splits.X_test, task=task)

                if task == "binary":
                    metrics = evaluate_binary_metrics(y_test, y_prob)
                else:
                    metrics = evaluate_multiclass_metrics(
                        y_test,
                        y_pred,
                        y_prob=y_prob,
                        all_classes=all_classes,
                        class_names=splits.inverse_label_mapping,
                    )

                record: dict[str, Any] = {
                    "dataset": dataset_name,
                    "task": task,
                    "seed": seed,
                    "model": model_name,
                    **{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))},
                }
                if task == "multiclass" and "per_class_f1" in metrics:
                    for cls_name, cls_f1 in metrics["per_class_f1"].items():
                        record[f"per_class_f1_{cls_name}"] = float(cls_f1)
                        if cls_name not in model_seed_per_class_f1[model_name]:
                            model_seed_per_class_f1[model_name][cls_name] = []
                        model_seed_per_class_f1[model_name][cls_name].append(float(cls_f1))
                    if "confusion_matrix_raw" in metrics:
                        model_seed_cm_raw[model_name].append(
                            np.asarray(metrics["confusion_matrix_raw"], dtype=np.int64)
                        )

                results_records.append(record)

                # Collect scalar metrics across seeds
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        if k not in model_seed_metrics[model_name]:
                            model_seed_metrics[model_name][k] = []
                        model_seed_metrics[model_name][k].append(float(v))

                if save_models:
                    model_path = save_model_artifact(
                        models_dir=effective_models_dir,
                        dataset=dataset_name,
                        task=task,
                        seed=seed,
                        model_name=model_name,
                        model=model,
                    )
                    logger.info("Saved model artifact to %s", model_path)

                # Free model memory
                del model, y_pred, y_prob
                gc.collect()

            # Free splits memory
            del splits, y_train, y_val, y_test, sw_train, sw_val
            gc.collect()

        # Compute mean ± SD and pairwise Holm stats for this task
        summary_path = output_dir / f"{dataset_name}_{task}_summary.json"
        task_summary: dict[str, Any] = {}
        if task == "multiclass" and class_order is not None:
            task_summary["class_order"] = class_order
        task_summary["models"] = {}

        for model_name, m_dict in model_seed_metrics.items():
            task_summary["models"][model_name] = {}
            for metric_key, values in m_dict.items():
                if task == "binary" and metric_key == "mcc":
                    task_summary["models"][model_name][metric_key] = compute_mean_std(
                        values, as_percentage=False, precision=3
                    )
                else:
                    task_summary["models"][model_name][metric_key] = compute_mean_std(
                        values, as_percentage=True, precision=2
                    )

            if task == "multiclass":
                per_class_summary: dict[str, Any] = {}
                ordered_names = (
                    class_order
                    if class_order is not None
                    else list(model_seed_per_class_f1[model_name].keys())
                )
                for cls_name in ordered_names:
                    if cls_name in model_seed_per_class_f1[model_name]:
                        per_class_summary[cls_name] = compute_mean_std(
                            model_seed_per_class_f1[model_name][cls_name],
                            as_percentage=True,
                            precision=2,
                        )
                task_summary["models"][model_name]["per_class_f1"] = per_class_summary

                if model_seed_cm_raw[model_name]:
                    cm_sum = np.sum(model_seed_cm_raw[model_name], axis=0).astype(np.int64)
                    row_sums = cm_sum.sum(axis=1, keepdims=True)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        cm_norm_pct = np.divide(
                            cm_sum.astype(np.float64) * 100.0,
                            row_sums,
                            out=np.zeros_like(cm_sum, dtype=np.float64),
                            where=row_sums != 0,
                        )
                    task_summary["models"][model_name]["confusion_matrix_raw"] = cm_sum.tolist()
                    task_summary["models"][model_name]["confusion_matrix_normalized"] = cm_norm_pct.tolist()

        # Pairwise statistical comparisons
        comparison_metrics = ["f1", "pr_auc"] if task == "binary" else ["macro_f1"]
        pairwise_by_metric: dict[str, list[dict[str, Any]]] = {}
        model_list = list(models)
        for metric_key in comparison_metrics:
            comparisons = []
            for i in range(len(model_list)):
                for j in range(i + 1, len(model_list)):
                    m_a = model_list[i]
                    m_b = model_list[j]
                    if (
                        metric_key in model_seed_metrics[m_a]
                        and metric_key in model_seed_metrics[m_b]
                    ):
                        comparisons.append(
                            {
                                "name_a": m_a,
                                "name_b": m_b,
                                "values_a": model_seed_metrics[m_a][metric_key],
                                "values_b": model_seed_metrics[m_b][metric_key],
                            }
                        )
            if comparisons:
                pairwise_by_metric[metric_key] = paired_t_test_with_holm(
                    comparisons, as_percentage=True
                )

        if pairwise_by_metric:
            task_summary["pairwise_comparisons"] = pairwise_by_metric

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(task_summary, f, indent=2)
        logger.info("Saved task summary to %s", summary_path)

    # Save detailed CSV
    results_df = pd.DataFrame(results_records)
    csv_path = output_dir / f"{dataset_name}_detailed_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Saved detailed results to %s", csv_path)

    del df
    gc.collect()
    return {"results": results_records}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run leakage-free NIDS tabular ML benchmarks."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cicids2017",
        choices=["cicids2017", "nf_unsw", "all"],
        help="Dataset to benchmark",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed parquet files",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="both",
        choices=["binary", "multiclass", "both"],
        help="Classification task to evaluate",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["xgboost", "lightgbm", "random_forest"],
        help="Models to evaluate",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Random seeds to run across",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/tables",
        help="Directory to save output CSVs and JSON summaries",
    )
    parser.add_argument(
        "--include-destination-port",
        action="store_true",
        default=False,
        help="Include Destination Port in features (for ablation only)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional downsampling size for quick runs",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        default=False,
        help="Persist trained models, preprocessors, and metadata before freeing RAM",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="saved_models",
        help="Root directory for persisted model artifacts when --save-models is set",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = ["binary", "multiclass"] if args.task == "both" else [args.task]
    datasets = (
        ["cicids2017", "nf_unsw"] if args.dataset == "all" else [args.dataset]
    )

    for ds in datasets:
        file_name = (
            "cicids2017.parquet" if ds == "cicids2017" else "nf_unsw_nb15_v2.parquet"
        )
        data_path = data_dir / file_name
        run_benchmark_for_dataset(
            dataset_name=ds,
            data_path=data_path,
            tasks=tasks,
            models=args.models,
            seeds=args.seeds,
            output_dir=output_dir,
            exclude_destination_port=not args.include_destination_port,
            sample_size=args.sample_size,
            save_models=args.save_models,
            models_dir=Path(args.models_dir),
        )


if __name__ == "__main__":
    main()
