"""Phase 4 Main Benchmark Runner: Post-hoc Calibration + Behavioral Ratio Engineering.

Implements two leakage-free tabular ML improvements over the Stage 1–3 paper baselines
on NF-UNSW-NB15-v2 and CICIDS2017 across seeds {0, 1, 2}:

1. Method 1 (`<model>_calibrated` for xgboost, lightgbm, random_forest):
   - Reloads saved baseline bundles from `saved_models/<dataset>/<task>/seed_<s>/`
     (zero refitting, zero retraining).
   - Tunes binary decision threshold or multiclass per-class prior multipliers
     STRICTLY on validation `(y_val, prob_val)`.
   - Evaluates once on test and persists calibration parameters under
     `saved_models/phase4_improved/<dataset>/<task>_calibrated/seed_<s>/`.

2. Method 2 (`<model>_ratio_calibrated` for xgboost, lightgbm):
   - Appends 12 stateless per-flow behavioral ratio features BEFORE fitting a fresh
     `LeakageFreePreprocessor` strictly on `X_train` of seed `s`.
   - Computes tempered class weights strictly on `y_train` (`alpha=0.35, max_weight=10.0`
     for binary; `alpha=0.5, max_weight=25.0` for multiclass).
   - Trains XGBoost and LightGBM with early stopping on validation, calibrates strictly
     on validation, and saves every trained model + ratio preprocessor + metadata +
     calibration artifact via `save_model_bundle` (`joblib.dump(..., compress=3)`)
     BEFORE `del model` and `gc.collect()`.

3. Zero-Leakage Validation Selection Safeguard (`phase4_best_<model>`):
   - Selects between `<model>_calibrated` and `<model>_ratio_calibrated` strictly by
     higher validation primary score (Binary F1 or Multiclass Macro-F1), never peeking
     at test scores.

4. Statistical Reporting:
   - Aggregates `mean ± SD` (`ddof=1`) across seeds and computes paired differences
     (`Phase4 - PaperBaseline`), 95% Student-t CIs, and Holm-Bonferroni paired t-tests
     against `results/tables/<dataset>_detailed_results.csv`.
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

from src.data.feature_engineering import append_behavioral_ratios
from src.data.split_preprocess import (
    LeakageFreePreprocessor,
    map_sample_weights,
)
from src.evaluation.metrics import (
    evaluate_binary_metrics,
    evaluate_multiclass_metrics,
)
from src.evaluation.stats import compute_mean_std, paired_t_test_with_holm
from src.models.calibration import (
    apply_multiclass_priors,
    compute_tempered_weights,
    predict_binary_with_threshold,
    tune_binary_threshold_on_val,
    tune_multiclass_priors_on_val,
)
from src.models.persistence import (
    get_seed_dir,
    load_model_bundle,
    save_model_artifact,
    save_seed_artifacts,
)
from src.models.tree_baselines import predict_baseline_model, train_baseline_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_phase4_main")

DATASET_PARQUET_MAP: dict[str, str] = {
    "cicids2017": "cicids2017.parquet",
    "nf_unsw": "nf_unsw_nb15_v2.parquet",
}

CALIBRATED_MODELS: list[str] = ["xgboost", "lightgbm", "random_forest"]
RATIO_MODELS: list[str] = ["xgboost", "lightgbm"]

BINARY_TEMPERED_ALPHA: float = 0.35
BINARY_TEMPERED_MAX_WEIGHT: float = 10.0
MULTICLASS_TEMPERED_ALPHA: float = 0.50
MULTICLASS_TEMPERED_MAX_WEIGHT: float = 25.0

BINARY_TRACKED_METRICS: list[str] = [
    "f1",
    "mcc",
    "pr_auc",
    "fpr_at_0.5",
    "recall_at_0.1_fpr",
    "recall_at_1.0_fpr",
    "recall",
    "precision",
    "accuracy",
]

MULTICLASS_TRACKED_METRICS: list[str] = [
    "macro_f1",
    "weighted_f1",
    "accuracy",
]


def _to_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy types and Path objects into JSON-safe Python values."""
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_calibration_artifact(
    save_dir: Path | str,
    dataset: str,
    task: str,
    seed: int,
    model_name: str,
    calibration: dict[str, Any],
) -> Path:
    """Save a calibration parameter JSON artifact under the canonical seed directory."""
    seed_dir = get_seed_dir(save_dir, dataset, task, seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    calib_path = seed_dir / f"calibration_{model_name}.json"
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_serializable(calibration), f, indent=2)
    return calib_path


def save_model_bundle(
    save_dir: Path | str,
    dataset: str,
    task: str,
    seed: int,
    model_name: str,
    model: Any,
    preprocessor: Any,
    feature_names: list[str],
    label_mapping: dict[str, int],
    inverse_label_mapping: dict[int, str],
    calibration: dict[str, Any] | None = None,
    exclude_destination_port: bool = True,
    compress: int = 3,
) -> Path:
    """Persist model + train-fitted preprocessor + metadata + optional calibration.

    Uses `joblib.dump(..., compress=3)` via `save_seed_artifacts` and
    `save_model_artifact` so every bundle is directly loadable by
    `src.models.persistence.load_model_bundle`.
    """
    save_seed_artifacts(
        models_dir=save_dir,
        dataset=dataset,
        task=task,
        seed=seed,
        preprocessor=preprocessor,
        feature_names=feature_names,
        label_mapping=label_mapping,
        inverse_label_mapping=inverse_label_mapping,
        exclude_destination_port=exclude_destination_port,
        compress=compress,
    )
    model_path = save_model_artifact(
        models_dir=save_dir,
        dataset=dataset,
        task=task,
        seed=seed,
        model_name=model_name,
        model=model,
        compress=compress,
    )
    if calibration is not None:
        save_calibration_artifact(
            save_dir=save_dir,
            dataset=dataset,
            task=task,
            seed=seed,
            model_name=model_name,
            calibration=calibration,
        )
    return model_path


def _resolve_parquet_path(dataset: str, data_dir: Path) -> Path:
    filename = DATASET_PARQUET_MAP.get(dataset, f"{dataset}.parquet")
    data_path = data_dir / filename
    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed dataset file not found at {data_path} for dataset='{dataset}'."
        )
    return data_path


def _extract_labels_and_mappings(
    df: pd.DataFrame,
    label_mapping_override: dict[str, int] | None = None,
    inverse_label_mapping_override: dict[int, str] | None = None,
    benign_label_name: str = "Benign",
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[int, str], set[str]]:
    """Extract numeric multiclass and binary labels matching `prepare_splits`."""
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
    if label_mapping_override is not None and inverse_label_mapping_override is not None:
        label_mapping = {str(k): int(v) for k, v in label_mapping_override.items()}
        inverse_label_mapping = {
            int(k): str(v) for k, v in inverse_label_mapping_override.items()
        }
        if pd.api.types.is_numeric_dtype(df[label_col]):
            y_multiclass = raw_labels.astype(np.int64, copy=True)
        else:
            y_multiclass = np.array(
                [label_mapping[str(x)] for x in raw_labels], dtype=np.int64
            )
    elif pd.api.types.is_numeric_dtype(df[label_col]):
        unique_classes = sorted(list(np.unique(raw_labels)))
        label_mapping = {str(c): int(c) for c in unique_classes}
        inverse_label_mapping = {int(c): str(c) for c in unique_classes}
        y_multiclass = raw_labels.astype(np.int64, copy=True)
    else:
        unique_classes = sorted(list(set(str(x) for x in raw_labels)))
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
            0 if 0 in inverse_label_mapping else 0,
        )
        y_binary = (y_multiclass != benign_id).astype(np.int64)

    label_cols_used: set[str] = {label_col}
    if binary_label_col:
        label_cols_used.add(binary_label_col)

    return y_multiclass, y_binary, label_mapping, inverse_label_mapping, label_cols_used


def _reconstruct_split_indices(
    y_multiclass: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct exact deterministic 60/20/20 stratified indices matching `prepare_splits`."""
    indices = np.arange(len(y_multiclass))
    train_idx, temp_idx = train_test_split(
        indices,
        train_size=0.6,
        random_state=seed,
        stratify=y_multiclass,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=seed,
        stratify=y_multiclass[temp_idx],
    )
    return train_idx, val_idx, test_idx


def _load_baseline_seed_values(
    baseline_csv_path: Path,
    dataset: str,
    task: str,
    seeds: list[int],
) -> tuple[dict[str, dict[str, list[float]]], dict[str, dict[str, list[float]]]]:
    """Load exact per-seed baseline metrics from `<dataset>_detailed_results.csv`."""
    scalar_map: dict[str, dict[str, list[float]]] = {}
    per_class_map: dict[str, dict[str, list[float]]] = {}
    if not baseline_csv_path.exists():
        logger.warning("Baseline CSV not found at %s; skipping paired baseline stats.", baseline_csv_path)
        return scalar_map, per_class_map

    bdf = pd.read_csv(baseline_csv_path)
    sub = bdf[(bdf["dataset"] == dataset) & (bdf["task"] == task)]
    if sub.empty:
        return scalar_map, per_class_map

    tracked_cols = BINARY_TRACKED_METRICS if task == "binary" else MULTICLASS_TRACKED_METRICS
    per_class_cols = [c for c in sub.columns if c.startswith("per_class_f1_")]

    for base_model in CALIBRATED_MODELS:
        msub = sub[sub["model"] == base_model]
        if msub.empty:
            continue
        scalar_map[base_model] = {}
        for col in tracked_cols:
            if col not in msub.columns:
                continue
            vals: list[float] = []
            valid = True
            for s in seeds:
                row = msub[msub["seed"] == s]
                if row.empty or pd.isna(row.iloc[0][col]):
                    valid = False
                    break
                vals.append(float(row.iloc[0][col]))
            if valid and len(vals) == len(seeds):
                scalar_map[base_model][col] = vals

        if task == "multiclass":
            per_class_map[base_model] = {}
            for pcol in per_class_cols:
                cls_name = pcol[len("per_class_f1_") :]
                vals = []
                valid = True
                for s in seeds:
                    row = msub[msub["seed"] == s]
                    if row.empty or pd.isna(row.iloc[0][pcol]):
                        valid = False
                        break
                    vals.append(float(row.iloc[0][pcol]))
                if valid and len(vals) == len(seeds):
                    per_class_map[base_model][cls_name] = vals

    return scalar_map, per_class_map


def _build_record(
    dataset: str,
    task: str,
    seed: int,
    variant_name: str,
    base_model: str,
    method: str,
    metrics: dict[str, Any],
    val_score: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a flat CSV-friendly detailed result record."""
    rec: dict[str, Any] = {
        "dataset": dataset,
        "task": task,
        "seed": int(seed),
        "model": variant_name,
        "base_model": base_model,
        "method": method,
        "val_selection_score": float(val_score),
    }
    if extra:
        rec.update(extra)
    for k, v in metrics.items():
        if not isinstance(v, (dict, list)):
            rec[k] = float(v)
    if task == "multiclass" and "per_class_f1" in metrics:
        for cls_name, cls_f1 in metrics["per_class_f1"].items():
            rec[f"per_class_f1_{cls_name}"] = float(cls_f1)
    return rec


def run_phase4_main_experiment(
    datasets: list[str] | None = None,
    tasks: list[str] | None = None,
    seeds: list[int] | None = None,
    device: str = "cuda",
    save_dir: Path | str = Path("saved_models/phase4_improved"),
    output_dir: Path | str = Path("results/tables"),
    data_dir: Path | str = Path("data/processed"),
    baseline_models_dir: Path | str = Path("saved_models"),
    baseline_tables_dir: Path | str = Path("results/tables"),
    exclude_destination_port: bool = True,
    sample_size: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Phase 4 main experiments (calibrated + ratio_calibrated + val-selected best)."""
    dataset_list = ["nf_unsw", "cicids2017"] if datasets is None else list(datasets)
    task_list = ["binary", "multiclass"] if tasks is None else list(tasks)
    seed_list = [0, 1, 2] if seeds is None else [int(s) for s in seeds]

    save_dir_path = Path(save_dir)
    output_dir_path = Path(output_dir)
    data_dir_path = Path(data_dir)
    baseline_models_path = Path(baseline_models_dir)
    baseline_tables_path = Path(baseline_tables_dir)
    extra_model_kwargs = dict(model_kwargs) if model_kwargs else {}

    save_dir_path.mkdir(parents=True, exist_ok=True)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    comparison_summary: dict[str, Any] = {
        "seeds": seed_list,
        "datasets": {},
    }

    for dataset in dataset_list:
        data_path = _resolve_parquet_path(dataset, data_dir_path)
        logger.info("Loading dataset %s from %s...", dataset, data_path)
        df = pd.read_parquet(data_path) if data_path.suffix == ".parquet" else pd.read_csv(data_path)

        if sample_size is not None and len(df) > sample_size:
            logger.info("Downsampling %s to %d rows", dataset, sample_size)
            lcol = (
                "label_multiclass"
                if "label_multiclass" in df.columns
                else ("Label" if "Label" in df.columns else "label")
            )
            parts = [
                grp.sample(
                    min(len(grp), max(6, int(sample_size * len(grp) / len(df)))),
                    random_state=42,
                )
                for _, grp in df.groupby(lcol, sort=False)
            ]
            df = pd.concat(parts, ignore_index=True)

        comparison_summary["datasets"][dataset] = {}

        for task in task_list:
            logger.info("=== Phase 4 Main | dataset=%s | task=%s ===", dataset, task)

            # All evaluated variant keys for this task
            variant_to_base: dict[str, str] = {}
            for m in CALIBRATED_MODELS:
                variant_to_base[f"{m}_calibrated"] = m
            for m in RATIO_MODELS:
                variant_to_base[f"{m}_ratio_calibrated"] = m
            for m in CALIBRATED_MODELS:
                variant_to_base[f"phase4_best_{m}"] = m

            variant_seed_metrics: dict[str, dict[str, list[float]]] = {
                v: {} for v in variant_to_base
            }
            variant_seed_per_class: dict[str, dict[str, list[float]]] = {
                v: {} for v in variant_to_base
            }
            variant_selected_sources: dict[str, list[str]] = {
                f"phase4_best_{m}": [] for m in CALIBRATED_MODELS
            }
            class_order: list[str] | None = None

            for seed in seed_list:
                logger.info(
                    "--- Running dataset=%s | task=%s | seed=%d ---",
                    dataset,
                    task,
                    seed,
                )

                # Inspect first saved baseline bundle's metadata to guarantee exact
                # feature_names and label_mapping parity with Stage 1–3 baselines
                first_bundle = load_model_bundle(
                    models_dir=baseline_models_path,
                    dataset=dataset,
                    task=task,
                    seed=seed,
                    model_name=CALIBRATED_MODELS[0],
                )
                meta = first_bundle["metadata"]
                base_feature_names = list(meta["feature_names"])
                label_mapping_meta = dict(meta["label_mapping"])
                inv_label_mapping_meta = {
                    int(k): str(v) for k, v in meta["inverse_label_mapping"].items()
                }
                base_preprocessor = first_bundle["preprocessor"]

                y_multi_all, y_bin_all, label_mapping, inverse_label_mapping, _ = (
                    _extract_labels_and_mappings(
                        df,
                        label_mapping_override=label_mapping_meta,
                        inverse_label_mapping_override=inv_label_mapping_meta,
                    )
                )

                train_idx, val_idx, test_idx = _reconstruct_split_indices(
                    y_multi_all, seed=seed
                )

                if task == "binary":
                    y_train = y_bin_all[train_idx]
                    y_val = y_bin_all[val_idx]
                    y_test = y_bin_all[test_idx]
                    all_classes: list[int] | None = None
                else:
                    y_train = y_multi_all[train_idx]
                    y_val = y_multi_all[val_idx]
                    y_test = y_multi_all[test_idx]
                    all_classes = sorted(inverse_label_mapping.keys())
                    if class_order is None:
                        class_order = [inverse_label_mapping[c] for c in all_classes]

                # Extract raw float32 split arrays once for this seed
                X_val_raw = df[base_feature_names].iloc[val_idx].to_numpy(dtype=np.float32)
                X_test_raw = df[base_feature_names].iloc[test_idx].to_numpy(dtype=np.float32)

                # Transform val and test strictly using the loaded train-fitted baseline preprocessor
                X_val_base = base_preprocessor.transform(X_val_raw)
                X_test_base = base_preprocessor.transform(X_test_raw)

                # Cache results per base model to perform validation-only selection safeguard
                cal_results_by_model: dict[str, dict[str, Any]] = {}
                ratio_results_by_model: dict[str, dict[str, Any]] = {}

                # ================================================================
                # Method 1: Post-hoc Calibration on Saved Baseline Models
                # ================================================================
                for idx_m, model_name in enumerate(CALIBRATED_MODELS):
                    if idx_m == 0:
                        bundle = first_bundle
                    else:
                        bundle = load_model_bundle(
                            models_dir=baseline_models_path,
                            dataset=dataset,
                            task=task,
                            seed=seed,
                            model_name=model_name,
                        )
                    model = bundle["model"]

                    _, prob_val = predict_baseline_model(model, X_val_base, task=task)
                    _, prob_test = predict_baseline_model(model, X_test_base, task=task)

                    if task == "binary":
                        # Tune threshold STRICTLY on (y_val, prob_val)
                        cal_info = tune_binary_threshold_on_val(y_val, prob_val)
                        best_thresh = float(cal_info["best_threshold"])
                        _ = predict_binary_with_threshold(prob_test, best_thresh)
                        metrics = evaluate_binary_metrics(
                            y_test, prob_test, threshold=best_thresh
                        )
                        val_primary = float(cal_info["val_f1_at_best"])
                        extra_fields: dict[str, Any] = {
                            "decision_threshold": best_thresh,
                            "val_f1_at_0.5": float(cal_info["val_f1_at_0.5"]),
                            "val_f1_at_best": val_primary,
                        }
                    else:
                        # Tune per-class priors STRICTLY on (y_val, prob_val)
                        cal_info = tune_multiclass_priors_on_val(y_val, prob_val)
                        gamma = np.asarray(cal_info["gamma"], dtype=np.float64)
                        y_pred_test, prob_test_cal = apply_multiclass_priors(
                            prob_test, gamma
                        )
                        metrics = evaluate_multiclass_metrics(
                            y_test,
                            y_pred_test,
                            y_prob=prob_test_cal,
                            all_classes=all_classes,
                            class_names=inverse_label_mapping,
                        )
                        val_primary = float(cal_info["val_macro_f1_after"])
                        extra_fields = {
                            "val_macro_f1_before": float(cal_info["val_macro_f1_before"]),
                            "val_macro_f1_after": val_primary,
                        }

                    calib_payload = {
                        "dataset": dataset,
                        "task": task,
                        "seed": int(seed),
                        "model": model_name,
                        "method": "calibrated",
                        "calibration": _to_json_serializable(cal_info),
                    }
                    calib_path = save_calibration_artifact(
                        save_dir=save_dir_path,
                        dataset=dataset,
                        task=f"{task}_calibrated",
                        seed=seed,
                        model_name=model_name,
                        calibration=calib_payload,
                    )
                    logger.info(
                        "[%s/%s/seed=%d] %s_calibrated val_score=%.4f | saved %s",
                        dataset,
                        task,
                        seed,
                        model_name,
                        val_primary,
                        calib_path,
                    )

                    variant_key = f"{model_name}_calibrated"
                    rec = _build_record(
                        dataset=dataset,
                        task=task,
                        seed=seed,
                        variant_name=variant_key,
                        base_model=model_name,
                        method="calibrated",
                        metrics=metrics,
                        val_score=val_primary,
                        extra=extra_fields,
                    )
                    all_records.append(rec)
                    cal_results_by_model[model_name] = {
                        "val_score": val_primary,
                        "metrics": metrics,
                        "extra": extra_fields,
                        "variant": variant_key,
                    }

                    for mk, mv in metrics.items():
                        if isinstance(mv, (int, float)):
                            variant_seed_metrics[variant_key].setdefault(mk, []).append(
                                float(mv)
                            )
                    if task == "multiclass" and "per_class_f1" in metrics:
                        for cname, cf1 in metrics["per_class_f1"].items():
                            variant_seed_per_class[variant_key].setdefault(
                                cname, []
                            ).append(float(cf1))

                    del model, bundle, prob_val, prob_test
                    gc.collect()

                # Free baseline preprocessed val/test before ratio feature engineering
                del first_bundle, base_preprocessor, X_val_base, X_test_base
                gc.collect()

                # ================================================================
                # Method 2: Stateless Behavioral Ratios + Tempered Weights + Calibration
                # ================================================================
                X_train_raw = df[base_feature_names].iloc[train_idx].to_numpy(dtype=np.float32)

                # Append 12 stateless behavioral ratios BEFORE preprocessing
                X_train_aug_raw, aug_feature_names = append_behavioral_ratios(
                    X_train_raw, base_feature_names, dataset=dataset
                )
                X_val_aug_raw, _ = append_behavioral_ratios(
                    X_val_raw, base_feature_names, dataset=dataset
                )
                X_test_aug_raw, _ = append_behavioral_ratios(
                    X_test_raw, base_feature_names, dataset=dataset
                )
                del X_train_raw, X_val_raw, X_test_raw
                gc.collect()

                # Fit fresh LeakageFreePreprocessor STRICTLY on X_train_aug_raw
                ratio_prep = LeakageFreePreprocessor(clip_range=(-10.0, 10.0))
                X_train_aug = ratio_prep.fit_transform(X_train_aug_raw, y_train=y_train)
                X_val_aug = ratio_prep.transform(X_val_aug_raw)
                X_test_aug = ratio_prep.transform(X_test_aug_raw)
                del X_train_aug_raw, X_val_aug_raw, X_test_aug_raw
                gc.collect()

                # Compute tempered class weights STRICTLY on y_train
                if task == "binary":
                    sw_train, cw_dict = compute_tempered_weights(
                        y_train,
                        alpha=BINARY_TEMPERED_ALPHA,
                        max_weight=BINARY_TEMPERED_MAX_WEIGHT,
                    )
                else:
                    sw_train, cw_dict = compute_tempered_weights(
                        y_train,
                        alpha=MULTICLASS_TEMPERED_ALPHA,
                        max_weight=MULTICLASS_TEMPERED_MAX_WEIGHT,
                    )
                sw_val = map_sample_weights(y_val, cw_dict)

                for model_name in RATIO_MODELS:
                    logger.info(
                        "Training %s_ratio_calibrated (%s/%s/seed=%d, n_features=%d)...",
                        model_name,
                        dataset,
                        task,
                        seed,
                        X_train_aug.shape[1],
                    )
                    train_kwargs: dict[str, Any] = dict(extra_model_kwargs)
                    if model_name == "xgboost":
                        train_kwargs.setdefault("device", device)

                    model = train_baseline_model(
                        model_name=model_name,
                        X_train=X_train_aug,
                        y_train=y_train,
                        X_val=X_val_aug,
                        y_val=y_val,
                        task=task,
                        seed=seed,
                        sample_weight_train=sw_train,
                        sample_weight_val=sw_val,
                        **train_kwargs,
                    )

                    _, prob_val = predict_baseline_model(model, X_val_aug, task=task)
                    _, prob_test = predict_baseline_model(model, X_test_aug, task=task)

                    if task == "binary":
                        cal_info = tune_binary_threshold_on_val(y_val, prob_val)
                        best_thresh = float(cal_info["best_threshold"])
                        _ = predict_binary_with_threshold(prob_test, best_thresh)
                        metrics = evaluate_binary_metrics(
                            y_test, prob_test, threshold=best_thresh
                        )
                        val_primary = float(cal_info["val_f1_at_best"])
                        extra_fields = {
                            "decision_threshold": best_thresh,
                            "val_f1_at_0.5": float(cal_info["val_f1_at_0.5"]),
                            "val_f1_at_best": val_primary,
                        }
                    else:
                        cal_info = tune_multiclass_priors_on_val(y_val, prob_val)
                        gamma = np.asarray(cal_info["gamma"], dtype=np.float64)
                        y_pred_test, prob_test_cal = apply_multiclass_priors(
                            prob_test, gamma
                        )
                        metrics = evaluate_multiclass_metrics(
                            y_test,
                            y_pred_test,
                            y_prob=prob_test_cal,
                            all_classes=all_classes,
                            class_names=inverse_label_mapping,
                        )
                        val_primary = float(cal_info["val_macro_f1_after"])
                        extra_fields = {
                            "val_macro_f1_before": float(cal_info["val_macro_f1_before"]),
                            "val_macro_f1_after": val_primary,
                        }

                    calib_payload = {
                        "dataset": dataset,
                        "task": task,
                        "seed": int(seed),
                        "model": model_name,
                        "method": "ratio_calibrated",
                        "tempered_alpha": (
                            BINARY_TEMPERED_ALPHA
                            if task == "binary"
                            else MULTICLASS_TEMPERED_ALPHA
                        ),
                        "tempered_max_weight": (
                            BINARY_TEMPERED_MAX_WEIGHT
                            if task == "binary"
                            else MULTICLASS_TEMPERED_MAX_WEIGHT
                        ),
                        "class_weights_train": _to_json_serializable(cw_dict),
                        "calibration": _to_json_serializable(cal_info),
                    }

                    # MANDATORY: Persist model + ratio preprocessor + calibration BEFORE del model
                    saved_model_path = save_model_bundle(
                        save_dir=save_dir_path,
                        dataset=dataset,
                        task=f"{task}_ratio",
                        seed=seed,
                        model_name=model_name,
                        model=model,
                        preprocessor=ratio_prep,
                        feature_names=aug_feature_names,
                        label_mapping=label_mapping,
                        inverse_label_mapping=inverse_label_mapping,
                        calibration=calib_payload,
                        exclude_destination_port=exclude_destination_port,
                        compress=3,
                    )
                    logger.info(
                        "[%s/%s/seed=%d] %s_ratio_calibrated val_score=%.4f | saved %s",
                        dataset,
                        task,
                        seed,
                        model_name,
                        val_primary,
                        saved_model_path,
                    )

                    variant_key = f"{model_name}_ratio_calibrated"
                    rec = _build_record(
                        dataset=dataset,
                        task=task,
                        seed=seed,
                        variant_name=variant_key,
                        base_model=model_name,
                        method="ratio_calibrated",
                        metrics=metrics,
                        val_score=val_primary,
                        extra=extra_fields,
                    )
                    all_records.append(rec)
                    ratio_results_by_model[model_name] = {
                        "val_score": val_primary,
                        "metrics": metrics,
                        "extra": extra_fields,
                        "variant": variant_key,
                    }

                    for mk, mv in metrics.items():
                        if isinstance(mv, (int, float)):
                            variant_seed_metrics[variant_key].setdefault(mk, []).append(
                                float(mv)
                            )
                    if task == "multiclass" and "per_class_f1" in metrics:
                        for cname, cf1 in metrics["per_class_f1"].items():
                            variant_seed_per_class[variant_key].setdefault(
                                cname, []
                            ).append(float(cf1))

                    del model, prob_val, prob_test
                    gc.collect()

                # ================================================================
                # Zero-Leakage Validation Selection Safeguard (`phase4_best_<model>`)
                # ================================================================
                for model_name in CALIBRATED_MODELS:
                    cal_entry = cal_results_by_model[model_name]
                    ratio_entry = ratio_results_by_model.get(model_name)

                    # Strict validation-only comparison: NEVER inspects test scores
                    if (
                        ratio_entry is not None
                        and float(ratio_entry["val_score"]) > float(cal_entry["val_score"])
                    ):
                        winner = ratio_entry
                    else:
                        winner = cal_entry

                    best_key = f"phase4_best_{model_name}"
                    variant_selected_sources[best_key].append(str(winner["variant"]))

                    best_extra = dict(winner["extra"])
                    best_extra["selected_variant"] = str(winner["variant"])
                    rec = _build_record(
                        dataset=dataset,
                        task=task,
                        seed=seed,
                        variant_name=best_key,
                        base_model=model_name,
                        method="val_selected_best",
                        metrics=winner["metrics"],
                        val_score=float(winner["val_score"]),
                        extra=best_extra,
                    )
                    all_records.append(rec)

                    for mk, mv in winner["metrics"].items():
                        if isinstance(mv, (int, float)):
                            variant_seed_metrics[best_key].setdefault(mk, []).append(
                                float(mv)
                            )
                    if task == "multiclass" and "per_class_f1" in winner["metrics"]:
                        for cname, cf1 in winner["metrics"]["per_class_f1"].items():
                            variant_seed_per_class[best_key].setdefault(
                                cname, []
                            ).append(float(cf1))

                # Free per-seed arrays
                del (
                    X_train_aug,
                    X_val_aug,
                    X_test_aug,
                    ratio_prep,
                    sw_train,
                    sw_val,
                    y_train,
                    y_val,
                    y_test,
                    y_multi_all,
                    y_bin_all,
                    train_idx,
                    val_idx,
                    test_idx,
                )
                gc.collect()

            # ====================================================================
            # Aggregate mean ± SD and Paired Statistical Comparisons vs Baseline
            # ====================================================================
            baseline_csv = baseline_tables_path / f"{dataset}_detailed_results.csv"
            base_scalar, base_per_class = _load_baseline_seed_values(
                baseline_csv, dataset=dataset, task=task, seeds=seed_list
            )

            task_summary: dict[str, Any] = {
                "baseline": {},
                "variants": {},
                "paired_vs_baseline": {},
            }
            if task == "multiclass" and class_order is not None:
                task_summary["class_order"] = class_order

            # Summarize baseline metrics for direct plotting/comparison convenience
            for bm, mdict in base_scalar.items():
                task_summary["baseline"][bm] = {}
                for mk, vals in mdict.items():
                    if task == "binary" and mk == "mcc":
                        task_summary["baseline"][bm][mk] = compute_mean_std(
                            vals, as_percentage=False, precision=3
                        )
                    else:
                        task_summary["baseline"][bm][mk] = compute_mean_std(
                            vals, as_percentage=True, precision=2
                        )
                if task == "multiclass" and bm in base_per_class:
                    task_summary["baseline"][bm]["per_class_f1"] = {
                        cname: compute_mean_std(vals, as_percentage=True, precision=2)
                        for cname, vals in base_per_class[bm].items()
                    }

            # Summarize Phase 4 variants
            for variant_key, mdict in variant_seed_metrics.items():
                v_summary: dict[str, Any] = {
                    "base_model": variant_to_base[variant_key],
                }
                if variant_key in variant_selected_sources:
                    v_summary["selected_variants_per_seed"] = variant_selected_sources[
                        variant_key
                    ]
                for mk, vals in mdict.items():
                    if task == "binary" and mk == "mcc":
                        v_summary[mk] = compute_mean_std(
                            vals, as_percentage=False, precision=3
                        )
                    else:
                        v_summary[mk] = compute_mean_std(
                            vals, as_percentage=True, precision=2
                        )
                if task == "multiclass":
                    ordered_names = (
                        class_order
                        if class_order is not None
                        else list(variant_seed_per_class[variant_key].keys())
                    )
                    v_summary["per_class_f1"] = {
                        cname: compute_mean_std(
                            variant_seed_per_class[variant_key][cname],
                            as_percentage=True,
                            precision=2,
                        )
                        for cname in ordered_names
                        if cname in variant_seed_per_class[variant_key]
                    }
                task_summary["variants"][variant_key] = v_summary

            # Paired Student-t tests with Holm-Bonferroni correction (Phase4 - PaperBaseline)
            paired_metrics = (
                ["f1", "pr_auc", "fpr_at_0.5", "recall_at_0.1_fpr", "recall_at_1.0_fpr", "mcc"]
                if task == "binary"
                else ["macro_f1", "weighted_f1", "accuracy"]
            )
            for mk in paired_metrics:
                comparisons: list[dict[str, Any]] = []
                for variant_key, base_model in variant_to_base.items():
                    if (
                        mk in variant_seed_metrics[variant_key]
                        and base_model in base_scalar
                        and mk in base_scalar[base_model]
                        and len(variant_seed_metrics[variant_key][mk])
                        == len(base_scalar[base_model][mk])
                    ):
                        comparisons.append(
                            {
                                "name_a": variant_key,
                                "name_b": f"baseline_{base_model}",
                                "values_a": variant_seed_metrics[variant_key][mk],
                                "values_b": base_scalar[base_model][mk],
                            }
                        )
                if comparisons:
                    is_pct = not (task == "binary" and mk == "mcc")
                    task_summary["paired_vs_baseline"][mk] = paired_t_test_with_holm(
                        comparisons, as_percentage=is_pct
                    )

            # Per-class paired tests for multiclass (e.g., Bot, Rare, Analysis, Backdoor)
            if task == "multiclass" and base_per_class:
                per_class_paired: dict[str, list[dict[str, Any]]] = {}
                target_classes = class_order if class_order is not None else []
                for cname in target_classes:
                    comparisons = []
                    for variant_key, base_model in variant_to_base.items():
                        if (
                            cname in variant_seed_per_class[variant_key]
                            and base_model in base_per_class
                            and cname in base_per_class[base_model]
                            and len(variant_seed_per_class[variant_key][cname])
                            == len(base_per_class[base_model][cname])
                        ):
                            comparisons.append(
                                {
                                    "name_a": variant_key,
                                    "name_b": f"baseline_{base_model}",
                                    "values_a": variant_seed_per_class[variant_key][cname],
                                    "values_b": base_per_class[base_model][cname],
                                }
                            )
                    if comparisons:
                        per_class_paired[cname] = paired_t_test_with_holm(
                            comparisons, as_percentage=True
                        )
                if per_class_paired:
                    task_summary["paired_vs_baseline_per_class_f1"] = per_class_paired

            comparison_summary["datasets"][dataset][task] = task_summary

        del df
        gc.collect()

    # Persist outputs
    csv_path = output_dir_path / "phase4_main_detailed.csv"
    results_df = pd.DataFrame(all_records)
    results_df.to_csv(csv_path, index=False)
    logger.info("Saved Phase 4 main detailed CSV to %s", csv_path)

    json_path = output_dir_path / "phase4_main_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_serializable(comparison_summary), f, indent=2)
    logger.info("Saved Phase 4 main comparison JSON to %s", json_path)

    return {
        "records": all_records,
        "summary": comparison_summary,
        "csv_path": csv_path,
        "json_path": json_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 4 leakage-free calibration and behavioral-ratio experiments."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["nf_unsw", "cicids2017"],
        choices=["nf_unsw", "cicids2017"],
        help="Datasets to evaluate (default: nf_unsw cicids2017)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["binary", "multiclass"],
        choices=["binary", "multiclass"],
        help="Tasks to evaluate (default: binary multiclass)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Random seeds (default: 0 1 2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Execution device for XGBoost (default: cuda)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="saved_models/phase4_improved",
        help="Directory to persist Phase 4 trained models and calibration artifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/tables",
        help="Directory to write Phase 4 comparison JSON and detailed CSV",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed parquet datasets",
    )
    parser.add_argument(
        "--baseline-models-dir",
        type=str,
        default="saved_models",
        help="Root directory of saved Stage 1–3 baseline bundles",
    )
    parser.add_argument(
        "--baseline-tables-dir",
        type=str,
        default="results/tables",
        help="Directory containing baseline *_detailed_results.csv files",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional downsampling size for quick verification",
    )

    args = parser.parse_args()
    run_phase4_main_experiment(
        datasets=args.datasets,
        tasks=args.tasks,
        seeds=args.seeds,
        device=args.device,
        save_dir=Path(args.save_dir),
        output_dir=Path(args.output_dir),
        data_dir=Path(args.data_dir),
        baseline_models_dir=Path(args.baseline_models_dir),
        baseline_tables_dir=Path(args.baseline_tables_dir),
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
