"""Leave-one-family-out stress test on CICIDS2017.

Protocol from Section 4 and Table 11 of 2611.04660v1.pdf:
1. Sequentially withhold one attack family from train and val splits:
   - DoS, DDoS, PortScan, BruteForce, Web, Bot.
2. Keep full test split (including the withheld family).
3. Preprocessing and binary class weights (C=2) fitted ONLY on withheld train split.
4. Train binary detector with early stopping on validation.
5. Persist train-fitted preprocessor, metadata, and trained model artifacts under
   `saved_models/cicids2017/leave_one_out/<family>/seed_<seed>/` before freeing RAM.
6. Evaluate binary detector at threshold 0.5 on:
   - Held-out family recall
   - Seen attacks recall
   - Benign FPR
7. Aggregate across seeds {0, 1, 2} (mean ± SD), including overall Table 11 summary
   rows for seen attacks recall and benign FPR, and save results.
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
from sklearn.model_selection import train_test_split

from src.data.split_preprocess import (
    LeakageFreePreprocessor,
    find_destination_port_columns,
)
from src.evaluation.stats import compute_mean_std
from src.models.persistence import save_model_artifact, save_seed_artifacts
from src.models.tree_baselines import predict_baseline_model, train_baseline_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_leave_one_out")

HELD_OUT_FAMILIES = ["DoS", "DDoS", "PortScan", "BruteForce", "Web", "Bot"]
DEFAULT_MODELS = ["xgboost"]
DEFAULT_SEEDS = [0, 1, 2]


def run_leave_one_out_experiment(
    data_path: Path,
    output_dir: Path,
    models: list[str] | None = None,
    seeds: list[int] | None = None,
    held_out_families: list[str] | None = None,
    exclude_destination_port: bool = True,
    sample_size: int | None = None,
    save_models: bool = True,
    models_dir: Path | None = Path("saved_models"),
    device: str = "cpu",
) -> dict[str, Any]:
    """Execute leave-one-family-out stress test across attack families and seeds."""
    model_list = list(DEFAULT_MODELS) if models is None else list(models)
    seed_list = list(DEFAULT_SEEDS) if seeds is None else list(seeds)
    family_list = (
        list(HELD_OUT_FAMILIES)
        if held_out_families is None
        else list(held_out_families)
    )
    effective_models_dir = (
        Path(models_dir) if models_dir is not None else Path("saved_models")
    )

    logger.info("Loading CICIDS2017 from %s", data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    label_col = (
        "label_multiclass"
        if "label_multiclass" in df.columns
        else ("Label" if "Label" in df.columns else "label")
    )

    if sample_size is not None and len(df) > sample_size:
        logger.info("Downsampling for quick test to %d rows", sample_size)
        sampled_parts = [
            grp.sample(
                min(len(grp), max(2, int(sample_size * len(grp) / len(df)))),
                random_state=42,
            )
            for _, grp in df.groupby(label_col, sort=False)
        ]
        df = pd.concat(sampled_parts, ignore_index=True)

    # Build numeric labels and mapping (matches prepare_splits)
    raw_labels = df[label_col].to_numpy()
    if pd.api.types.is_numeric_dtype(df[label_col]):
        unique_classes = sorted(list(np.unique(raw_labels)))
        label_mapping = {str(c): int(c) for c in unique_classes}
        inverse_label_mapping = {int(c): str(c) for c in unique_classes}
        y_multi_all = raw_labels.astype(np.int64, copy=True)
        benign_class_id = label_mapping.get("Benign", 0)
    else:
        unique_classes = sorted(list(set(str(x) for x in raw_labels)))
        benign_matches = [c for c in unique_classes if c.lower() == "benign"]
        if benign_matches:
            benign_actual = benign_matches[0]
            ordered_classes = [benign_actual] + [
                c for c in unique_classes if c != benign_actual
            ]
        else:
            benign_actual = None
            ordered_classes = unique_classes

        label_mapping = {c: idx for idx, c in enumerate(ordered_classes)}
        inverse_label_mapping = {idx: c for idx, c in enumerate(ordered_classes)}
        y_multi_all = np.array(
            [label_mapping[str(x)] for x in raw_labels], dtype=np.int64
        )
        benign_class_id = (
            label_mapping[benign_actual] if benign_actual is not None else 0
        )

    # Drop label columns and destination port without full DataFrame copy
    cols_to_drop_set: set[str] = {label_col}
    for c in ["label_binary", "Binary_Label", "binary_label"]:
        if c in df.columns:
            cols_to_drop_set.add(c)
    if exclude_destination_port:
        cols_to_drop_set.update(find_destination_port_columns(df.columns))

    feature_names = [c for c in df.columns if c not in cols_to_drop_set]
    X_raw_all = df[feature_names].to_numpy(dtype=np.float32)

    del df
    gc.collect()

    results_records: list[dict[str, Any]] = []
    evaluated_families: list[str] = []
    summary_by_model_family: dict[str, dict[str, dict[str, list[Any]]]] = {
        m: {
            f: {
                "held_out_recall": [],
                "seen_recall": [],
                "benign_fpr": [],
                "n_held_test": [],
            }
            for f in family_list
        }
        for m in model_list
    }

    # Iterate over held-out families
    for family in family_list:
        matching_c_ids = [
            cid
            for cid, name in inverse_label_mapping.items()
            if name.lower() == family.lower()
        ]
        if not matching_c_ids:
            logger.warning("Family %s not found in dataset labels, skipping.", family)
            continue
        held_out_cid = matching_c_ids[0]
        evaluated_families.append(family)
        logger.info(
            "=== Running Leave-One-Out for Family: %s (id=%d) ===",
            family,
            held_out_cid,
        )

        for seed in seed_list:
            # 60/40 then 50/50 stratified split (exact match to prepare_splits)
            indices = np.arange(len(X_raw_all))
            train_idx, temp_idx = train_test_split(
                indices, train_size=0.6, random_state=seed, stratify=y_multi_all
            )
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=0.5,
                random_state=seed,
                stratify=y_multi_all[temp_idx],
            )

            # Filter held-out family from train and val ONLY
            train_sub_mask = y_multi_all[train_idx] != held_out_cid
            val_sub_mask = y_multi_all[val_idx] != held_out_cid

            final_train_idx = train_idx[train_sub_mask]
            final_val_idx = val_idx[val_sub_mask]
            final_test_idx = test_idx  # Test retains all classes!

            X_train_raw = X_raw_all[final_train_idx]
            y_train_multi = y_multi_all[final_train_idx]
            y_train_bin = (y_train_multi != benign_class_id).astype(np.int64)

            X_val_raw = X_raw_all[final_val_idx]
            y_val_multi = y_multi_all[final_val_idx]
            y_val_bin = (y_val_multi != benign_class_id).astype(np.int64)

            X_test_raw = X_raw_all[final_test_idx]
            y_test_multi = y_multi_all[final_test_idx]

            # Fit preprocessor and binary class weights (C=2) ONLY on filtered train split
            prep = LeakageFreePreprocessor()
            X_train = prep.fit_transform(X_train_raw, y_train=y_train_bin)
            X_val = prep.transform(X_val_raw)
            X_test = prep.transform(X_test_raw)

            sample_weight_train = prep.get_sample_weights(y_train_bin)
            sample_weight_val = prep.get_sample_weights(y_val_bin)

            del X_train_raw, X_val_raw, X_test_raw, indices, train_idx, temp_idx, val_idx, test_idx
            gc.collect()

            lofo_task = f"leave_one_out/{family}"
            if save_models:
                seed_dir = save_seed_artifacts(
                    models_dir=effective_models_dir,
                    dataset="cicids2017",
                    task=lofo_task,
                    seed=seed,
                    preprocessor=prep,
                    feature_names=feature_names,
                    label_mapping=label_mapping,
                    inverse_label_mapping=inverse_label_mapping,
                    exclude_destination_port=exclude_destination_port,
                )
                logger.info("Saved LOFO preprocessor and metadata to %s", seed_dir)

            for model_name in model_list:
                logger.info(
                    "Training %s (withheld=%s, seed=%d)...",
                    model_name,
                    family,
                    seed,
                )
                train_kwargs: dict[str, Any] = {}
                canonical_name = model_name.lower().replace("-", "_").replace(" ", "_")
                if canonical_name in ("xgboost", "xgb"):
                    train_kwargs["device"] = device

                model = train_baseline_model(
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train_bin,
                    X_val=X_val,
                    y_val=y_val_bin,
                    task="binary",
                    seed=seed,
                    sample_weight_train=sample_weight_train,
                    sample_weight_val=sample_weight_val,
                    **train_kwargs,
                )

                y_pred, _ = predict_baseline_model(
                    model, X_test, task="binary", threshold=0.5
                )

                # 1. Held-out family recall
                held_mask = y_test_multi == held_out_cid
                n_held = int(np.sum(held_mask))
                held_recall = (
                    float(np.sum((y_pred == 1) & held_mask) / n_held)
                    if n_held > 0
                    else 0.0
                )

                # 2. Seen attack recall
                seen_mask = (y_test_multi != held_out_cid) & (
                    y_test_multi != benign_class_id
                )
                n_seen = int(np.sum(seen_mask))
                seen_recall = (
                    float(np.sum((y_pred == 1) & seen_mask) / n_seen)
                    if n_seen > 0
                    else 0.0
                )

                # 3. Benign FPR
                benign_mask = y_test_multi == benign_class_id
                n_benign = int(np.sum(benign_mask))
                benign_fpr = (
                    float(np.sum((y_pred == 1) & benign_mask) / n_benign)
                    if n_benign > 0
                    else 0.0
                )

                record = {
                    "held_out_family": family,
                    "model": model_name,
                    "seed": seed,
                    "held_out_recall": held_recall,
                    "seen_attack_recall": seen_recall,
                    "benign_fpr": benign_fpr,
                    "n_held_test": n_held,
                    "n_seen_test": n_seen,
                    "n_benign_test": n_benign,
                }
                results_records.append(record)

                summary_by_model_family[model_name][family]["held_out_recall"].append(
                    held_recall
                )
                summary_by_model_family[model_name][family]["seen_recall"].append(
                    seen_recall
                )
                summary_by_model_family[model_name][family]["benign_fpr"].append(
                    benign_fpr
                )
                summary_by_model_family[model_name][family]["n_held_test"].append(
                    n_held
                )

                if save_models:
                    model_path = save_model_artifact(
                        models_dir=effective_models_dir,
                        dataset="cicids2017",
                        task=lofo_task,
                        seed=seed,
                        model_name=model_name,
                        model=model,
                    )
                    logger.info("Saved LOFO model artifact to %s", model_path)

                del model, y_pred
                gc.collect()

            del X_train, X_val, X_test, prep, sample_weight_train, sample_weight_val
            gc.collect()

    del X_raw_all, y_multi_all
    gc.collect()

    # Save detailed CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results_records)
    csv_path = output_dir / "leave_one_out_detailed.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Saved detailed results to %s", csv_path)

    # Compute summary mean ± SD (%) per family and Table 11 overall rows
    summary: dict[str, Any] = {"models": {}}
    for m in model_list:
        summary["models"][m] = {}
        for fam in evaluated_families:
            d = summary_by_model_family[m][fam]
            summary["models"][m][fam] = {
                "test_flows": int(round(float(np.mean(d["n_held_test"])))),
                "test_flows_per_seed": [int(x) for x in d["n_held_test"]],
                "held_out_recall": compute_mean_std(
                    d["held_out_recall"], as_percentage=True
                ),
                "seen_attack_recall": compute_mean_std(
                    d["seen_recall"], as_percentage=True
                ),
                "benign_fpr": compute_mean_std(d["benign_fpr"], as_percentage=True),
            }

        if evaluated_families and seed_list:
            pooled_seen = [
                val
                for fam in evaluated_families
                for val in summary_by_model_family[m][fam]["seen_recall"]
            ]
            pooled_fpr = [
                val
                for fam in evaluated_families
                for val in summary_by_model_family[m][fam]["benign_fpr"]
            ]
            per_seed_seen = [
                float(
                    np.mean(
                        [
                            summary_by_model_family[m][fam]["seen_recall"][s_idx]
                            for fam in evaluated_families
                        ]
                    )
                )
                for s_idx in range(len(seed_list))
            ]
            per_seed_fpr = [
                float(
                    np.mean(
                        [
                            summary_by_model_family[m][fam]["benign_fpr"][s_idx]
                            for fam in evaluated_families
                        ]
                    )
                )
                for s_idx in range(len(seed_list))
            ]

            overall_seen = compute_mean_std(pooled_seen, as_percentage=True)
            overall_seen["n_runs"] = len(pooled_seen)
            overall_seen["per_seed_mean"] = compute_mean_std(
                per_seed_seen, as_percentage=True
            )

            overall_fpr = compute_mean_std(pooled_fpr, as_percentage=True)
            overall_fpr["n_runs"] = len(pooled_fpr)
            overall_fpr["per_seed_mean"] = compute_mean_std(
                per_seed_fpr, as_percentage=True
            )

            summary["models"][m]["overall_seen_attack_recall"] = overall_seen
            summary["models"][m]["overall_benign_fpr"] = overall_fpr

    summary_path = output_dir / "leave_one_out_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary to %s", summary_path)

    return {"records": results_records, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run leave-one-family-out stress test on CICIDS2017."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/cicids2017.parquet",
        help="Path to cleaned CICIDS2017 parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/tables",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Models to evaluate (e.g. xgboost lightgbm random_forest)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Random seeds",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional sample size for quick verification",
    )
    parser.add_argument(
        "--save-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist LOFO models, preprocessors, and metadata before freeing RAM (use --no-save-models to disable)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="saved_models",
        help="Root directory for persisted LOFO model artifacts",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Execution device for XGBoost (default: cpu)",
    )
    parser.add_argument(
        "--include-destination-port",
        action="store_true",
        default=False,
        help="Include Destination Port in features (for ablation only)",
    )

    args = parser.parse_args()
    run_leave_one_out_experiment(
        data_path=Path(args.data_path),
        output_dir=Path(args.output_dir),
        models=args.models,
        seeds=args.seeds,
        exclude_destination_port=not args.include_destination_port,
        sample_size=args.sample_size,
        save_models=args.save_models,
        models_dir=Path(args.models_dir),
        device=args.device,
    )


if __name__ == "__main__":
    main()
