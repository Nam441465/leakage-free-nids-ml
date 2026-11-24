"""Phase 4 Stress Test Runners: Noise-Robust Training & Hybrid Leave-One-Family-Out.

Implements two leakage-free stress-test improvements on CICIDS2017 across seeds {0, 1, 2}:

Part A — Gaussian Feature Noise Robustness (`run_phase4_noise_experiment`):
  - Uses the exact same stratified 100,000-flow test subsample and noise RNG seed
    logic as `src/experiments/run_noise_test.py` for 1:1 paired comparison against
    `results/tables/noise_test_detailed.csv`.
  - Reloads the train-fitted baseline preprocessor from
    `saved_models/cicids2017/binary/seed_<s>/`.
  - Builds noise-augmented training data strictly on `X_train_std` via
    `build_noise_augmented_training_data(..., aug_ratio=0.5, sigmas=(0.1, 0.25, 0.5))`.
  - Uses combined clean + noisy validation (`sigma=0.1, 0.25`) for early stopping
    and threshold calibration (strictly on validation, subject to clean validation
    FPR <= 0.25%).
  - Trains `xgboost_noise_robust` and `lightgbm_noise_robust`, persists them via
    `save_model_bundle` (`joblib.dump(..., compress=3)`) under
    `saved_models/phase4_improved/cicids2017/noise_robust/seed_<s>/` BEFORE `del model`.
  - Evaluates `random_forest_noise_calibrated` by reloading the saved baseline
    Random Forest and calibrating its threshold on noisy validation (`sigma=0.1`)
    subject to clean validation FPR <= 0.25%.
  - Writes `results/tables/phase4_noise_comparison.json` and
    `results/tables/phase4_noise_detailed.csv`.

Part B — Hybrid Leave-One-Family-Out (`run_phase4_lofo_experiment`):
  - Evaluates all 6 withheld families (`DoS`, `DDoS`, `PortScan`, `BruteForce`,
    `Web`, `Bot`) across seeds `{0, 1, 2}` (18 runs).
  - Reloads each saved baseline LOFO bundle (`xgboost` + `preprocessor`) from
    `saved_models/cicids2017/leave_one_out/<f>/seed_<s>/`.
  - Fits `BenignAnomalyScorer` STRICTLY on Benign training flows (`y_train == 0`,
    withheld family excluded).
  - Calibrates hybrid thresholds STRICTLY on validation (`y_val` without withheld
    family) via `calibrate_hybrid_lofo_on_val(..., target_benign_fpr=0.0020)`.
  - Persists `{"anomaly_scorer": scorer, "calib_params": calib_params}` via
    `joblib.dump(..., compress=3)` to
    `saved_models/phase4_improved/cicids2017/leave_one_out/<f>/seed_<s>/hybrid_detector.joblib`.
  - Evaluates on full test split (`held_out_recall`, `seen_attack_recall`,
    `benign_fpr`) and computes paired statistics against
    `results/tables/leave_one_out_detailed.csv`.
  - Writes `results/tables/phase4_lofo_comparison.json` and
    `results/tables/phase4_lofo_detailed.csv`.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.data.split_preprocess import compute_class_weights, map_sample_weights
from src.evaluation.metrics import evaluate_binary_metrics
from src.evaluation.stats import compute_mean_std, paired_t_test_with_holm
from src.experiments.run_leave_one_out import HELD_OUT_FAMILIES
from src.experiments.run_noise_test import (
    SIGMA_VALUES,
    TRACKED_METRICS as NOISE_TRACKED_METRICS,
    _subsample_test_flows,
)
from src.experiments.run_phase4_main import (
    _extract_labels_and_mappings,
    _reconstruct_split_indices,
    _to_json_serializable,
    save_calibration_artifact,
    save_model_bundle,
)
from src.models.calibration import (
    build_noise_augmented_training_data,
    predict_binary_with_threshold,
    tune_binary_threshold_on_val,
)
from src.models.hybrid_lofo import (
    BenignAnomalyScorer,
    calibrate_hybrid_lofo_on_val,
    predict_hybrid_lofo,
)
from src.models.persistence import load_model_bundle
from src.models.tree_baselines import predict_baseline_model, train_baseline_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_phase4_stress")

DEFAULT_SEEDS: list[int] = [0, 1, 2]
NOISE_MODEL_TO_BASE: dict[str, str] = {
    "xgboost_noise_robust": "xgboost",
    "lightgbm_noise_robust": "lightgbm",
    "random_forest_noise_calibrated": "random_forest",
}
CLEAN_VAL_FPR_CAP: float = 0.0025  # 0.25%


def _min_threshold_for_clean_fpr(
    y_val_clean: np.ndarray,
    prob_val_clean: np.ndarray,
    max_clean_fpr: float = CLEAN_VAL_FPR_CAP,
) -> float:
    """Compute the smallest probability threshold achieving clean validation FPR <= max_clean_fpr."""
    y_clean = np.asarray(y_val_clean, dtype=np.int64)
    p_clean = np.asarray(prob_val_clean, dtype=np.float64)
    benign_probs = p_clean[y_clean == 0]
    if len(benign_probs) == 0:
        return 0.05

    sorted_desc = np.sort(benign_probs)[::-1]
    n_neg = len(sorted_desc)
    max_fp = int(np.floor(n_neg * max_clean_fpr))

    if max_fp >= n_neg:
        return 0.01
    if max_fp <= 0:
        return float(min(0.95, float(sorted_desc[0]) + 1e-6))

    # Threshold strictly above sorted_desc[max_fp] allows at most max_fp false positives
    cutoff_val = float(sorted_desc[max_fp])
    # Step up slightly above cutoff_val so ties at cutoff_val do not exceed max_fp
    t_min = float(np.nextafter(cutoff_val, 1.0))
    # Verify empirically on benign_probs (guard against exact ties pushing count above max_fp)
    if float(np.mean(benign_probs >= t_min)) > max_clean_fpr:
        strictly_higher = sorted_desc[sorted_desc > cutoff_val]
        if len(strictly_higher) > 0:
            t_min = float(strictly_higher[-1])
        else:
            t_min = float(min(0.95, cutoff_val + 1e-5))
    return float(np.clip(t_min, 0.01, 0.95))


def _tune_noise_threshold_on_val(
    y_val_clean: np.ndarray,
    prob_val_clean: np.ndarray,
    y_val_noisy: np.ndarray,
    prob_val_noisy: np.ndarray,
    max_clean_fpr: float = CLEAN_VAL_FPR_CAP,
) -> dict[str, float]:
    """Tune binary decision threshold strictly on validation subject to clean val FPR <= max_clean_fpr."""
    y_clean = np.asarray(y_val_clean, dtype=np.int64)
    p_clean = np.asarray(prob_val_clean, dtype=np.float64)
    y_noisy = np.asarray(y_val_noisy, dtype=np.int64)
    p_noisy = np.asarray(prob_val_noisy, dtype=np.float64)

    benign_mask = y_clean == 0
    benign_probs = p_clean[benign_mask]

    min_thresh = _min_threshold_for_clean_fpr(
        y_clean, p_clean, max_clean_fpr=max_clean_fpr
    )

    # First invoke standard validation threshold tuner over [min_thresh, 0.95]
    raw_cal = tune_binary_threshold_on_val(
        y_noisy,
        p_noisy,
        grid_size=181,
        min_thresh=min_thresh,
        max_thresh=0.95,
    )
    candidate_t = float(raw_cal["best_threshold"])
    clean_fpr_at_candidate = (
        float(np.mean(benign_probs >= candidate_t)) if len(benign_probs) > 0 else 0.0
    )

    if clean_fpr_at_candidate <= max_clean_fpr:
        best_t = candidate_t
        best_noisy_f1 = float(raw_cal["val_f1_at_best"])
        best_clean_fpr = clean_fpr_at_candidate
    else:
        # Enforce strict clean validation FPR cap when lower quantile/0.5 candidate exceeded cap
        grid = np.linspace(min_thresh, 0.95, 181)
        quantiles = np.quantile(p_noisy, np.linspace(0.05, 0.95, 37))
        extra = np.array([min_thresh, 0.5, 0.95], dtype=np.float64)
        candidates = np.unique(np.concatenate([grid, quantiles, extra]))
        candidates = candidates[(candidates >= min_thresh) & (candidates <= 0.95)]
        if len(candidates) == 0:
            candidates = np.array([min_thresh], dtype=np.float64)

        best_t = float(candidates[0])
        best_noisy_f1 = -1.0
        best_clean_fpr = (
            float(np.mean(benign_probs >= best_t)) if len(benign_probs) > 0 else 0.0
        )
        for t in candidates:
            t_f = float(t)
            c_fpr = (
                float(np.mean(benign_probs >= t_f)) if len(benign_probs) > 0 else 0.0
            )
            if c_fpr > max_clean_fpr:
                continue
            y_pred_n = predict_binary_with_threshold(p_noisy, t_f)
            f1_n = float(f1_score(y_noisy, y_pred_n, zero_division=0.0))
            if f1_n > best_noisy_f1:
                best_noisy_f1 = f1_n
                best_t = t_f
                best_clean_fpr = c_fpr

        if best_noisy_f1 < 0.0:
            # Fallback to tightest threshold min_thresh
            best_t = float(min_thresh)
            y_pred_n = predict_binary_with_threshold(p_noisy, best_t)
            best_noisy_f1 = float(f1_score(y_noisy, y_pred_n, zero_division=0.0))
            best_clean_fpr = (
                float(np.mean(benign_probs >= best_t)) if len(benign_probs) > 0 else 0.0
            )

    clean_f1_at_best = float(
        f1_score(y_clean, predict_binary_with_threshold(p_clean, best_t), zero_division=0.0)
    )
    return {
        "best_threshold": float(best_t),
        "min_thresh_for_clean_fpr": float(min_thresh),
        "max_clean_fpr_cap": float(max_clean_fpr),
        "val_noisy_f1_at_0.5": float(raw_cal["val_f1_at_0.5"]),
        "val_noisy_f1_at_best": float(best_noisy_f1),
        "val_clean_f1_at_best": float(clean_f1_at_best),
        "val_clean_fpr_at_best": float(best_clean_fpr),
    }


def _make_noisy_copy(
    X_std: np.ndarray,
    sigma: float,
    rng_seed: int,
) -> np.ndarray:
    """Add Gaussian noise N(0, sigma^2) and clip to [-10, 10] in float32."""
    if sigma <= 0.0:
        return X_std
    rng = np.random.RandomState(seed=rng_seed)
    noise = rng.normal(0.0, sigma, size=X_std.shape).astype(np.float32)
    X_noisy = np.clip(X_std + noise, -10.0, 10.0).astype(np.float32)
    del noise
    return X_noisy


def _evaluate_phase4_model_under_noise(
    model: Any,
    model_name: str,
    seed: int,
    X_test_sub: np.ndarray,
    y_test_sub: np.ndarray,
    sigmas: list[float],
    threshold: float,
    results_records: list[dict[str, Any]],
    summary_tracking: dict[str, dict[float, dict[str, list[float]]]],
) -> None:
    """Evaluate a Phase 4 binary model across Gaussian noise levels using exact run_noise_test RNG."""
    for sigma in sigmas:
        logger.info(
            "Evaluating %s under Gaussian noise sigma=%.2f (seed=%d, thresh=%.4f)...",
            model_name,
            sigma,
            seed,
            threshold,
        )
        if sigma > 0.0:
            # EXACT RNG seed parity with src/experiments/run_noise_test.py
            rng = np.random.RandomState(seed=1000 + int(round(sigma * 100)) + seed)
            noise = rng.normal(0.0, sigma, size=X_test_sub.shape).astype(np.float32)
            X_noisy = np.clip(X_test_sub + noise, -10.0, 10.0).astype(np.float32)
        else:
            X_noisy = X_test_sub

        _, y_prob = predict_baseline_model(
            model, X_noisy, task="binary", threshold=threshold
        )
        metrics = evaluate_binary_metrics(y_test_sub, y_prob, threshold=threshold)

        record = {
            "model": model_name,
            "base_model": NOISE_MODEL_TO_BASE.get(model_name, model_name),
            "seed": int(seed),
            "sigma": float(sigma),
            "decision_threshold": float(threshold),
            "f1": float(metrics["f1"]),
            "precision": float(metrics["precision"]),
            "pr_auc": float(metrics["pr_auc"]),
            "recall": float(metrics["recall"]),
            "fpr_at_0.5": float(metrics["fpr_at_0.5"]),
            "mcc": float(metrics["mcc"]),
            "accuracy": float(metrics["accuracy"]),
        }
        results_records.append(record)

        for metric_key in NOISE_TRACKED_METRICS:
            summary_tracking[model_name][sigma][metric_key].append(
                float(metrics[metric_key])
            )

        if sigma > 0.0:
            del noise, X_noisy
            gc.collect()


def _load_baseline_noise_values(
    baseline_noise_csv: Path,
    seeds: list[int],
    sigmas: list[float],
) -> dict[str, dict[float, dict[str, list[float]]]]:
    """Load exact per-seed baseline values from `noise_test_detailed.csv`."""
    out: dict[str, dict[float, dict[str, list[float]]]] = {}
    if not baseline_noise_csv.exists():
        logger.warning(
            "Baseline noise CSV not found at %s; skipping paired baseline noise stats.",
            baseline_noise_csv,
        )
        return out

    bdf = pd.read_csv(baseline_noise_csv)
    for base_model in ["xgboost", "lightgbm", "random_forest"]:
        msub = bdf[bdf["model"] == base_model]
        if msub.empty:
            continue
        out[base_model] = {}
        for sigma in sigmas:
            ssub = msub[np.isclose(msub["sigma"].astype(float), float(sigma), atol=1e-6)]
            out[base_model][sigma] = {}
            for mk in NOISE_TRACKED_METRICS:
                if mk not in ssub.columns:
                    continue
                vals: list[float] = []
                valid = True
                for s in seeds:
                    row = ssub[ssub["seed"] == s]
                    if row.empty or pd.isna(row.iloc[0][mk]):
                        valid = False
                        break
                    vals.append(float(row.iloc[0][mk]))
                if valid and len(vals) == len(seeds):
                    out[base_model][sigma][mk] = vals
    return out


def run_phase4_noise_experiment(
    data_path: Path | str = Path("data/processed/cicids2017.parquet"),
    output_dir: Path | str = Path("results/tables"),
    save_dir: Path | str = Path("saved_models/phase4_improved"),
    baseline_models_dir: Path | str = Path("saved_models"),
    baseline_tables_dir: Path | str = Path("results/tables"),
    seeds: list[int] | None = None,
    sigmas: list[float] | None = None,
    n_test_samples: int = 100_000,
    device: str = "cuda",
    sample_size: int | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute Phase 4 Part A: Noise-Robust Training & Calibration on CICIDS2017 Binary."""
    seed_list = list(DEFAULT_SEEDS) if seeds is None else [int(s) for s in seeds]
    sigma_list = list(SIGMA_VALUES) if sigmas is None else [float(s) for s in sigmas]

    data_path_obj = Path(data_path)
    output_dir_path = Path(output_dir)
    save_dir_path = Path(save_dir)
    baseline_models_path = Path(baseline_models_dir)
    baseline_tables_path = Path(baseline_tables_dir)
    extra_model_kwargs = dict(model_kwargs) if model_kwargs else {}

    output_dir_path.mkdir(parents=True, exist_ok=True)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading CICIDS2017 for Phase 4 Noise Robustness from %s...", data_path_obj)
    if not data_path_obj.exists():
        raise FileNotFoundError(f"Data file not found at {data_path_obj}")
    df = (
        pd.read_parquet(data_path_obj)
        if data_path_obj.suffix == ".parquet"
        else pd.read_csv(data_path_obj)
    )

    if sample_size is not None and len(df) > sample_size:
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

    phase4_models = list(NOISE_MODEL_TO_BASE.keys())
    results_records: list[dict[str, Any]] = []
    summary_tracking: dict[str, dict[float, dict[str, list[float]]]] = {
        m: {s: {mk: [] for mk in NOISE_TRACKED_METRICS} for s in sigma_list}
        for m in phase4_models
    }

    for seed in seed_list:
        logger.info("=== Phase 4 Noise Robustness | seed=%d ===", seed)

        # Load saved baseline preprocessor + metadata (+ saved random_forest model)
        rf_bundle = load_model_bundle(
            models_dir=baseline_models_path,
            dataset="cicids2017",
            task="binary",
            seed=seed,
            model_name="random_forest",
        )
        preprocessor = rf_bundle["preprocessor"]
        metadata = rf_bundle["metadata"]
        rf_model = rf_bundle["model"]

        feature_names = list(metadata["feature_names"])
        label_mapping_meta = dict(metadata["label_mapping"])
        inv_label_mapping_meta = {
            int(k): str(v) for k, v in metadata["inverse_label_mapping"].items()
        }
        exclude_dst_port = bool(metadata.get("exclude_destination_port", True))

        y_multi_all, y_bin_all, label_mapping, inverse_label_mapping, _ = (
            _extract_labels_and_mappings(
                df,
                label_mapping_override=label_mapping_meta,
                inverse_label_mapping_override=inv_label_mapping_meta,
            )
        )
        train_idx, val_idx, test_idx = _reconstruct_split_indices(y_multi_all, seed=seed)

        y_bin_train = y_bin_all[train_idx]
        y_bin_val = y_bin_all[val_idx]
        y_bin_test = y_bin_all[test_idx]

        # Transform train/val/test strictly with the loaded baseline train-fitted preprocessor
        X_train_std = preprocessor.transform(
            df[feature_names].iloc[train_idx].to_numpy(dtype=np.float32)
        )
        X_val_std = preprocessor.transform(
            df[feature_names].iloc[val_idx].to_numpy(dtype=np.float32)
        )
        X_test_std = preprocessor.transform(
            df[feature_names].iloc[test_idx].to_numpy(dtype=np.float32)
        )

        # Exact stratified test subsample matching src/experiments/run_noise_test.py
        X_test_sub, y_test_sub = _subsample_test_flows(
            X_test_std, y_bin_test, n_test_samples=n_test_samples, seed=seed
        )
        del X_test_std, y_bin_test, train_idx, val_idx, test_idx, y_multi_all, y_bin_all
        gc.collect()

        # Build noisy validation copies (validation-only RNG seeds; ZERO test leakage)
        X_val_noisy_01 = _make_noisy_copy(X_val_std, sigma=0.1, rng_seed=2100 + seed)
        X_val_noisy_025 = _make_noisy_copy(X_val_std, sigma=0.25, rng_seed=2250 + seed)

        # Evaluate random_forest_noise_calibrated first so we can free rf_model promptly
        _, rf_prob_val_clean = predict_baseline_model(rf_model, X_val_std, task="binary")
        _, rf_prob_val_noisy01 = predict_baseline_model(
            rf_model, X_val_noisy_01, task="binary"
        )
        rf_val_y_target = np.concatenate([y_bin_val, y_bin_val], axis=0)
        rf_val_p_target = np.concatenate([rf_prob_val_clean, rf_prob_val_noisy01], axis=0)
        rf_cal = _tune_noise_threshold_on_val(
            y_val_clean=y_bin_val,
            prob_val_clean=rf_prob_val_clean,
            y_val_noisy=rf_val_y_target,
            prob_val_noisy=rf_val_p_target,
            max_clean_fpr=CLEAN_VAL_FPR_CAP,
        )
        rf_thresh = float(rf_cal["best_threshold"])
        save_calibration_artifact(
            save_dir=save_dir_path,
            dataset="cicids2017",
            task="noise_robust",
            seed=seed,
            model_name="random_forest_noise_calibrated",
            calibration={
                "dataset": "cicids2017",
                "task": "noise_robust",
                "seed": int(seed),
                "model": "random_forest_noise_calibrated",
                "source_baseline_model": "saved_models/cicids2017/binary",
                "calibration": rf_cal,
            },
        )
        _evaluate_phase4_model_under_noise(
            model=rf_model,
            model_name="random_forest_noise_calibrated",
            seed=seed,
            X_test_sub=X_test_sub,
            y_test_sub=y_test_sub,
            sigmas=sigma_list,
            threshold=rf_thresh,
            results_records=results_records,
            summary_tracking=summary_tracking,
        )
        del rf_model, rf_bundle, rf_prob_val_clean, rf_prob_val_noisy01, rf_val_y_target, rf_val_p_target
        gc.collect()

        # Build noise-augmented training data strictly from train split
        bin_cw = compute_class_weights(y_bin_train)
        w_bin_train = map_sample_weights(y_bin_train, bin_cw)
        w_bin_val = map_sample_weights(y_bin_val, bin_cw)

        X_train_aug, y_train_aug, w_train_aug = build_noise_augmented_training_data(
            X_train_std,
            y_bin_train,
            w_bin_train,
            seed=seed,
            aug_ratio=0.5,
            sigmas=(0.1, 0.25, 0.5),
        )
        del X_train_std, y_bin_train, w_bin_train
        gc.collect()

        # Combined clean + noisy validation (sigma=0.1, 0.25) for early stopping & threshold tuning
        # Stratified 50% of each noisy copy keeps 50% clean / 50% noisy balance and low RAM footprint
        val_indices = np.arange(len(y_bin_val))
        if len(y_bin_val) >= 20 and len(np.unique(y_bin_val)) >= 2:
            val_sub_01, _ = train_test_split(
                val_indices, train_size=0.5, random_state=3100 + seed, stratify=y_bin_val
            )
            val_sub_025, _ = train_test_split(
                val_indices, train_size=0.5, random_state=3250 + seed, stratify=y_bin_val
            )
        else:
            val_sub_01 = val_indices
            val_sub_025 = val_indices

        X_val_combined = np.concatenate(
            [X_val_std, X_val_noisy_01[val_sub_01], X_val_noisy_025[val_sub_025]],
            axis=0,
        ).astype(np.float32)
        y_val_combined = np.concatenate(
            [y_bin_val, y_bin_val[val_sub_01], y_bin_val[val_sub_025]], axis=0
        ).astype(np.int64)
        w_val_combined = np.concatenate(
            [w_bin_val, w_bin_val[val_sub_01], w_bin_val[val_sub_025]], axis=0
        ).astype(np.float32)

        X_val_noisy_eval = np.concatenate(
            [X_val_noisy_01[val_sub_01], X_val_noisy_025[val_sub_025]], axis=0
        ).astype(np.float32)
        y_val_noisy_eval = np.concatenate(
            [y_bin_val[val_sub_01], y_bin_val[val_sub_025]], axis=0
        ).astype(np.int64)

        del X_val_noisy_01, X_val_noisy_025, val_indices, val_sub_01, val_sub_025, w_bin_val
        gc.collect()

        # Train xgboost_noise_robust and lightgbm_noise_robust sequentially
        for base_model_name, robust_name in [
            ("xgboost", "xgboost_noise_robust"),
            ("lightgbm", "lightgbm_noise_robust"),
        ]:
            logger.info("Training %s (seed=%d)...", robust_name, seed)
            train_kwargs: dict[str, Any] = dict(extra_model_kwargs)
            if base_model_name == "xgboost":
                train_kwargs.setdefault("device", device)

            model = train_baseline_model(
                model_name=base_model_name,
                X_train=X_train_aug,
                y_train=y_train_aug,
                X_val=X_val_combined,
                y_val=y_val_combined,
                task="binary",
                seed=seed,
                sample_weight_train=w_train_aug,
                sample_weight_val=w_val_combined,
                **train_kwargs,
            )

            # Tune decision threshold strictly on validation (noisy val F1, clean val FPR <= 0.25%)
            _, prob_val_clean = predict_baseline_model(model, X_val_std, task="binary")
            _, prob_val_noisy = predict_baseline_model(
                model, X_val_noisy_eval, task="binary"
            )
            cal_info = _tune_noise_threshold_on_val(
                y_val_clean=y_bin_val,
                prob_val_clean=prob_val_clean,
                y_val_noisy=y_val_noisy_eval,
                prob_val_noisy=prob_val_noisy,
                max_clean_fpr=CLEAN_VAL_FPR_CAP,
            )
            best_thresh = float(cal_info["best_threshold"])

            # MANDATORY: Save model bundle BEFORE del model and gc.collect()
            saved_path = save_model_bundle(
                save_dir=save_dir_path,
                dataset="cicids2017",
                task="noise_robust",
                seed=seed,
                model_name=robust_name,
                model=model,
                preprocessor=preprocessor,
                feature_names=feature_names,
                label_mapping=label_mapping,
                inverse_label_mapping=inverse_label_mapping,
                calibration={
                    "dataset": "cicids2017",
                    "task": "noise_robust",
                    "seed": int(seed),
                    "model": robust_name,
                    "aug_ratio": 0.5,
                    "aug_sigmas": [0.1, 0.25, 0.5],
                    "calibration": cal_info,
                },
                exclude_destination_port=exclude_dst_port,
                compress=3,
            )
            logger.info(
                "Saved %s (seed=%d, thresh=%.4f, clean_val_fpr=%.5f) to %s",
                robust_name,
                seed,
                best_thresh,
                cal_info["val_clean_fpr_at_best"],
                saved_path,
            )

            _evaluate_phase4_model_under_noise(
                model=model,
                model_name=robust_name,
                seed=seed,
                X_test_sub=X_test_sub,
                y_test_sub=y_test_sub,
                sigmas=sigma_list,
                threshold=best_thresh,
                results_records=results_records,
                summary_tracking=summary_tracking,
            )

            del model, prob_val_clean, prob_val_noisy
            gc.collect()

        del (
            preprocessor,
            X_train_aug,
            y_train_aug,
            w_train_aug,
            X_val_std,
            y_bin_val,
            X_val_combined,
            y_val_combined,
            w_val_combined,
            X_val_noisy_eval,
            y_val_noisy_eval,
            X_test_sub,
            y_test_sub,
        )
        gc.collect()

    del df
    gc.collect()

    # Load baseline noise detailed CSV for paired statistics
    baseline_noise_csv = baseline_tables_path / "noise_test_detailed.csv"
    base_noise_vals = _load_baseline_noise_values(
        baseline_noise_csv, seeds=seed_list, sigmas=sigma_list
    )

    summary: dict[str, Any] = {
        "seeds": seed_list,
        "sigmas": sigma_list,
        "baseline": {},
        "models": {},
        "paired_vs_baseline": {},
    }

    for bm, sdict in base_noise_vals.items():
        summary["baseline"][bm] = {}
        for sigma in sigma_list:
            s_key = f"sigma_{sigma}"
            summary["baseline"][bm][s_key] = {
                mk: compute_mean_std(sdict[sigma][mk], as_percentage=True, precision=2)
                for mk in NOISE_TRACKED_METRICS
                if sigma in sdict and mk in sdict[sigma]
            }

    for m in phase4_models:
        summary["models"][m] = {}
        for sigma in sigma_list:
            s_key = f"sigma_{sigma}"
            summary["models"][m][s_key] = {
                mk: compute_mean_std(
                    summary_tracking[m][sigma][mk], as_percentage=True, precision=2
                )
                for mk in NOISE_TRACKED_METRICS
            }

    # Paired Student-t tests with Holm correction per sigma and metric
    for sigma in sigma_list:
        s_key = f"sigma_{sigma}"
        summary["paired_vs_baseline"][s_key] = {}
        for mk in ["f1", "pr_auc", "recall", "fpr_at_0.5"]:
            comparisons: list[dict[str, Any]] = []
            for p4_model, base_model in NOISE_MODEL_TO_BASE.items():
                if (
                    base_model in base_noise_vals
                    and sigma in base_noise_vals[base_model]
                    and mk in base_noise_vals[base_model][sigma]
                    and len(summary_tracking[p4_model][sigma][mk])
                    == len(base_noise_vals[base_model][sigma][mk])
                ):
                    comparisons.append(
                        {
                            "name_a": p4_model,
                            "name_b": f"baseline_{base_model}",
                            "values_a": summary_tracking[p4_model][sigma][mk],
                            "values_b": base_noise_vals[base_model][sigma][mk],
                        }
                    )
            if comparisons:
                summary["paired_vs_baseline"][s_key][mk] = paired_t_test_with_holm(
                    comparisons, as_percentage=True
                )

    csv_path = output_dir_path / "phase4_noise_detailed.csv"
    pd.DataFrame(results_records).to_csv(csv_path, index=False)
    logger.info("Saved Phase 4 noise detailed CSV to %s", csv_path)

    json_path = output_dir_path / "phase4_noise_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_serializable(summary), f, indent=2)
    logger.info("Saved Phase 4 noise comparison JSON to %s", json_path)

    return {
        "records": results_records,
        "summary": summary,
        "csv_path": csv_path,
        "json_path": json_path,
    }


def _load_baseline_lofo_values(
    baseline_lofo_csv: Path,
    families: list[str],
    seeds: list[int],
    model_name: str = "xgboost",
) -> dict[str, dict[str, list[float]]]:
    """Load exact per-seed baseline LOFO metrics from `leave_one_out_detailed.csv`."""
    out: dict[str, dict[str, list[float]]] = {}
    if not baseline_lofo_csv.exists():
        logger.warning(
            "Baseline LOFO CSV not found at %s; skipping paired baseline LOFO stats.",
            baseline_lofo_csv,
        )
        return out

    bdf = pd.read_csv(baseline_lofo_csv)
    msub = bdf[bdf["model"] == model_name]
    for fam in families:
        fsub = msub[msub["held_out_family"].str.lower() == fam.lower()]
        if fsub.empty:
            continue
        out[fam] = {}
        for mk in ["held_out_recall", "seen_attack_recall", "benign_fpr"]:
            vals: list[float] = []
            valid = True
            for s in seeds:
                row = fsub[fsub["seed"] == s]
                if row.empty or pd.isna(row.iloc[0][mk]):
                    valid = False
                    break
                vals.append(float(row.iloc[0][mk]))
            if valid and len(vals) == len(seeds):
                out[fam][mk] = vals
    return out


def run_phase4_lofo_experiment(
    data_path: Path | str = Path("data/processed/cicids2017.parquet"),
    output_dir: Path | str = Path("results/tables"),
    save_dir: Path | str = Path("saved_models/phase4_improved"),
    baseline_models_dir: Path | str = Path("saved_models"),
    baseline_tables_dir: Path | str = Path("results/tables"),
    seeds: list[int] | None = None,
    held_out_families: list[str] | None = None,
    target_benign_fpr: float = 0.0020,
    sample_size: int | None = None,
) -> dict[str, Any]:
    """Execute Phase 4 Part B: Hybrid XGBoost + Benign Anomaly Detector LOFO."""
    seed_list = list(DEFAULT_SEEDS) if seeds is None else [int(s) for s in seeds]
    family_list = (
        list(HELD_OUT_FAMILIES)
        if held_out_families is None
        else list(held_out_families)
    )

    data_path_obj = Path(data_path)
    output_dir_path = Path(output_dir)
    save_dir_path = Path(save_dir)
    baseline_models_path = Path(baseline_models_dir)
    baseline_tables_path = Path(baseline_tables_dir)

    output_dir_path.mkdir(parents=True, exist_ok=True)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading CICIDS2017 for Phase 4 Hybrid LOFO from %s...", data_path_obj)
    if not data_path_obj.exists():
        raise FileNotFoundError(f"Data file not found at {data_path_obj}")
    df = (
        pd.read_parquet(data_path_obj)
        if data_path_obj.suffix == ".parquet"
        else pd.read_csv(data_path_obj)
    )

    if sample_size is not None and len(df) > sample_size:
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

    # Extract canonical labels matching run_leave_one_out.py
    y_multi_all, _, label_mapping_df, inverse_label_mapping_df, _ = (
        _extract_labels_and_mappings(df)
    )
    benign_matches = [
        cid for cid, name in inverse_label_mapping_df.items() if name.lower() == "benign"
    ]
    benign_class_id = benign_matches[0] if benign_matches else 0

    results_records: list[dict[str, Any]] = []
    evaluated_families: list[str] = []
    hybrid_tracking: dict[str, dict[str, list[Any]]] = {
        fam: {
            "held_out_recall": [],
            "seen_attack_recall": [],
            "benign_fpr": [],
            "n_held_test": [],
        }
        for fam in family_list
    }

    for family in family_list:
        matching_cids = [
            cid
            for cid, name in inverse_label_mapping_df.items()
            if name.lower() == family.lower()
        ]
        if not matching_cids:
            logger.warning("Family %s not found in dataset labels; skipping.", family)
            continue
        held_out_cid = matching_cids[0]
        evaluated_families.append(family)
        logger.info("=== Phase 4 Hybrid LOFO | family=%s (id=%d) ===", family, held_out_cid)

        for seed in seed_list:
            # Load saved baseline LOFO bundle (xgboost + train-fitted preprocessor + metadata)
            lofo_task = f"leave_one_out/{family}"
            bundle = load_model_bundle(
                models_dir=baseline_models_path,
                dataset="cicids2017",
                task=lofo_task,
                seed=seed,
                model_name="xgboost",
            )
            xgb_model = bundle["model"]
            prep = bundle["preprocessor"]
            meta = bundle["metadata"]
            feature_names = list(meta["feature_names"])

            # Reconstruct exact 60/20/20 split matching run_leave_one_out.py
            train_idx, val_idx, test_idx = _reconstruct_split_indices(
                y_multi_all, seed=seed
            )

            # Filter withheld family out of train and val ONLY (keep test full!)
            final_train_idx = train_idx[y_multi_all[train_idx] != held_out_cid]
            final_val_idx = val_idx[y_multi_all[val_idx] != held_out_cid]
            final_test_idx = test_idx

            y_train_bin_filtered = (
                y_multi_all[final_train_idx] != benign_class_id
            ).astype(np.int64)
            y_val_bin_filtered = (
                y_multi_all[final_val_idx] != benign_class_id
            ).astype(np.int64)
            y_test_multi = y_multi_all[final_test_idx]

            # Transform filtered train/val and full test using the loaded baseline preprocessor
            X_train_filtered = prep.transform(
                df[feature_names].iloc[final_train_idx].to_numpy(dtype=np.float32)
            )
            X_val_filtered = prep.transform(
                df[feature_names].iloc[final_val_idx].to_numpy(dtype=np.float32)
            )
            X_test_full = prep.transform(
                df[feature_names].iloc[final_test_idx].to_numpy(dtype=np.float32)
            )

            # Fit BenignAnomalyScorer STRICTLY on Benign training flows (y == 0)
            scorer = BenignAnomalyScorer().fit(
                X_train_filtered, y_train_bin_filtered, seed=seed
            )

            # Calibrate hybrid thresholds STRICTLY on validation (withheld family excluded!)
            _, xgb_prob_val = predict_baseline_model(
                xgb_model, X_val_filtered, task="binary"
            )
            anom_raw_val = scorer.score_raw(X_val_filtered)
            calib_params = calibrate_hybrid_lofo_on_val(
                y_val_bin_filtered,
                xgb_prob_val,
                anom_raw_val,
                target_benign_fpr=target_benign_fpr,
            )

            # Persist hybrid detector BEFORE freeing memory (compress=3)
            hybrid_dir = (
                save_dir_path
                / "cicids2017"
                / "leave_one_out"
                / family
                / f"seed_{seed}"
            )
            hybrid_dir.mkdir(parents=True, exist_ok=True)
            hybrid_path = hybrid_dir / "hybrid_detector.joblib"
            joblib.dump(
                {"anomaly_scorer": scorer, "calib_params": calib_params},
                hybrid_path,
                compress=3,
            )
            with open(hybrid_dir / "calibration_hybrid.json", "w", encoding="utf-8") as f:
                json.dump(_to_json_serializable(calib_params), f, indent=2)

            # Evaluate once on full X_test_full
            _, xgb_prob_test = predict_baseline_model(
                xgb_model, X_test_full, task="binary"
            )
            anom_raw_test = scorer.score_raw(X_test_full)
            y_pred_hybrid, _ = predict_hybrid_lofo(
                xgb_prob_test, anom_raw_test, calib_params
            )

            held_mask = y_test_multi == held_out_cid
            n_held = int(np.sum(held_mask))
            held_recall = (
                float(np.sum((y_pred_hybrid == 1) & held_mask) / n_held)
                if n_held > 0
                else 0.0
            )

            seen_mask = (y_test_multi != held_out_cid) & (
                y_test_multi != benign_class_id
            )
            n_seen = int(np.sum(seen_mask))
            seen_recall = (
                float(np.sum((y_pred_hybrid == 1) & seen_mask) / n_seen)
                if n_seen > 0
                else 0.0
            )

            benign_mask = y_test_multi == benign_class_id
            n_benign = int(np.sum(benign_mask))
            benign_fpr = (
                float(np.sum((y_pred_hybrid == 1) & benign_mask) / n_benign)
                if n_benign > 0
                else 0.0
            )

            logger.info(
                "[LOFO %s/seed=%d] held_out_recall=%.2f%% | seen_recall=%.2f%% | benign_fpr=%.4f%% | saved %s",
                family,
                seed,
                held_recall * 100.0,
                seen_recall * 100.0,
                benign_fpr * 100.0,
                hybrid_path,
            )

            record = {
                "held_out_family": family,
                "model": "xgboost_hybrid_lofo",
                "base_model": "xgboost",
                "seed": int(seed),
                "held_out_recall": held_recall,
                "seen_attack_recall": seen_recall,
                "benign_fpr": benign_fpr,
                "n_held_test": n_held,
                "n_seen_test": n_seen,
                "n_benign_test": n_benign,
                "xgb_thresh": float(calib_params["xgb_thresh"]),
                "anom_thresh": float(calib_params["anom_thresh"]),
                "val_benign_fpr_hybrid": float(calib_params["val_benign_fpr_hybrid"]),
            }
            results_records.append(record)

            hybrid_tracking[family]["held_out_recall"].append(held_recall)
            hybrid_tracking[family]["seen_attack_recall"].append(seen_recall)
            hybrid_tracking[family]["benign_fpr"].append(benign_fpr)
            hybrid_tracking[family]["n_held_test"].append(n_held)

            del (
                xgb_model,
                prep,
                bundle,
                scorer,
                X_train_filtered,
                X_val_filtered,
                X_test_full,
                xgb_prob_val,
                anom_raw_val,
                xgb_prob_test,
                anom_raw_test,
                y_pred_hybrid,
                train_idx,
                val_idx,
                test_idx,
                final_train_idx,
                final_val_idx,
            )
            gc.collect()

    del df, y_multi_all
    gc.collect()

    # Load baseline LOFO detailed CSV for paired statistics
    baseline_lofo_csv = baseline_tables_path / "leave_one_out_detailed.csv"
    base_lofo = _load_baseline_lofo_values(
        baseline_lofo_csv,
        families=evaluated_families,
        seeds=seed_list,
        model_name="xgboost",
    )

    summary: dict[str, Any] = {
        "seeds": seed_list,
        "families": evaluated_families,
        "baseline": {"xgboost": {}},
        "models": {"xgboost_hybrid_lofo": {}},
        "paired_vs_baseline": {},
    }

    for fam in evaluated_families:
        d = hybrid_tracking[fam]
        summary["models"]["xgboost_hybrid_lofo"][fam] = {
            "test_flows": int(round(float(np.mean(d["n_held_test"])))),
            "test_flows_per_seed": [int(x) for x in d["n_held_test"]],
            "held_out_recall": compute_mean_std(
                d["held_out_recall"], as_percentage=True, precision=2
            ),
            "seen_attack_recall": compute_mean_std(
                d["seen_attack_recall"], as_percentage=True, precision=2
            ),
            "benign_fpr": compute_mean_std(
                d["benign_fpr"], as_percentage=True, precision=2
            ),
        }
        if fam in base_lofo:
            summary["baseline"]["xgboost"][fam] = {
                mk: compute_mean_std(base_lofo[fam][mk], as_percentage=True, precision=2)
                for mk in ["held_out_recall", "seen_attack_recall", "benign_fpr"]
                if mk in base_lofo[fam]
            }

    # Overall rows (pooled across all 18 runs + per-seed mean across families, matching Table 11)
    if evaluated_families and seed_list:
        pooled_seen = [
            v
            for fam in evaluated_families
            for v in hybrid_tracking[fam]["seen_attack_recall"]
        ]
        pooled_fpr = [
            v for fam in evaluated_families for v in hybrid_tracking[fam]["benign_fpr"]
        ]
        pooled_held = [
            v
            for fam in evaluated_families
            for v in hybrid_tracking[fam]["held_out_recall"]
        ]
        per_seed_seen = [
            float(
                np.mean(
                    [
                        hybrid_tracking[fam]["seen_attack_recall"][si]
                        for fam in evaluated_families
                    ]
                )
            )
            for si in range(len(seed_list))
        ]
        per_seed_fpr = [
            float(
                np.mean(
                    [
                        hybrid_tracking[fam]["benign_fpr"][si]
                        for fam in evaluated_families
                    ]
                )
            )
            for si in range(len(seed_list))
        ]
        per_seed_held = [
            float(
                np.mean(
                    [
                        hybrid_tracking[fam]["held_out_recall"][si]
                        for fam in evaluated_families
                    ]
                )
            )
            for si in range(len(seed_list))
        ]

        overall_seen = compute_mean_std(pooled_seen, as_percentage=True, precision=2)
        overall_seen["n_runs"] = len(pooled_seen)
        overall_seen["per_seed_mean"] = compute_mean_std(
            per_seed_seen, as_percentage=True, precision=2
        )
        overall_fpr = compute_mean_std(pooled_fpr, as_percentage=True, precision=2)
        overall_fpr["n_runs"] = len(pooled_fpr)
        overall_fpr["per_seed_mean"] = compute_mean_std(
            per_seed_fpr, as_percentage=True, precision=2
        )
        overall_held = compute_mean_std(pooled_held, as_percentage=True, precision=2)
        overall_held["n_runs"] = len(pooled_held)
        overall_held["per_seed_mean"] = compute_mean_std(
            per_seed_held, as_percentage=True, precision=2
        )

        summary["models"]["xgboost_hybrid_lofo"][
            "overall_seen_attack_recall"
        ] = overall_seen
        summary["models"]["xgboost_hybrid_lofo"]["overall_benign_fpr"] = overall_fpr
        summary["models"]["xgboost_hybrid_lofo"][
            "overall_held_out_recall"
        ] = overall_held

        # Baseline overall rows if available
        if all(fam in base_lofo for fam in evaluated_families):
            b_pooled_seen = [
                v
                for fam in evaluated_families
                for v in base_lofo[fam].get("seen_attack_recall", [])
            ]
            b_pooled_fpr = [
                v
                for fam in evaluated_families
                for v in base_lofo[fam].get("benign_fpr", [])
            ]
            b_pooled_held = [
                v
                for fam in evaluated_families
                for v in base_lofo[fam].get("held_out_recall", [])
            ]
            if b_pooled_seen:
                summary["baseline"]["xgboost"][
                    "overall_seen_attack_recall"
                ] = compute_mean_std(b_pooled_seen, as_percentage=True, precision=2)
            if b_pooled_fpr:
                summary["baseline"]["xgboost"]["overall_benign_fpr"] = compute_mean_std(
                    b_pooled_fpr, as_percentage=True, precision=2
                )
            if b_pooled_held:
                summary["baseline"]["xgboost"][
                    "overall_held_out_recall"
                ] = compute_mean_std(b_pooled_held, as_percentage=True, precision=2)

    # Paired Student-t tests with Holm-Bonferroni correction across families
    for mk in ["held_out_recall", "seen_attack_recall", "benign_fpr"]:
        comparisons: list[dict[str, Any]] = []
        for fam in evaluated_families:
            if (
                fam in base_lofo
                and mk in base_lofo[fam]
                and len(hybrid_tracking[fam][mk]) == len(base_lofo[fam][mk])
            ):
                comparisons.append(
                    {
                        "name_a": f"hybrid_{fam}",
                        "name_b": f"baseline_{fam}",
                        "values_a": hybrid_tracking[fam][mk],
                        "values_b": base_lofo[fam][mk],
                    }
                )
        if comparisons:
            summary["paired_vs_baseline"][mk] = paired_t_test_with_holm(
                comparisons, as_percentage=True
            )

    csv_path = output_dir_path / "phase4_lofo_detailed.csv"
    pd.DataFrame(results_records).to_csv(csv_path, index=False)
    logger.info("Saved Phase 4 LOFO detailed CSV to %s", csv_path)

    json_path = output_dir_path / "phase4_lofo_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_json_serializable(summary), f, indent=2)
    logger.info("Saved Phase 4 LOFO comparison JSON to %s", json_path)

    return {
        "records": results_records,
        "summary": summary,
        "csv_path": csv_path,
        "json_path": json_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 4 stress tests (Noise Robustness & Hybrid LOFO) on CICIDS2017."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
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
        help="Directory to persist Phase 4 models and hybrid detectors",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/tables",
        help="Directory to write comparison JSONs and detailed CSVs",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "noise", "lofo"],
        help="Which stress test(s) to run: all, noise, or lofo (default: all)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/cicids2017.parquet",
        help="Path to cleaned CICIDS2017 parquet",
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
        help="Directory containing baseline noise_test_detailed.csv and leave_one_out_detailed.csv",
    )
    parser.add_argument(
        "--n-test-samples",
        type=int,
        default=100_000,
        help="Maximum stratified test flows for Gaussian noise evaluation (default: 100000)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional downsampling size for quick verification",
    )

    args = parser.parse_args()

    if args.mode in ("all", "noise"):
        run_phase4_noise_experiment(
            data_path=Path(args.data_path),
            output_dir=Path(args.output_dir),
            save_dir=Path(args.save_dir),
            baseline_models_dir=Path(args.baseline_models_dir),
            baseline_tables_dir=Path(args.baseline_tables_dir),
            seeds=args.seeds,
            n_test_samples=args.n_test_samples,
            device=args.device,
            sample_size=args.sample_size,
        )

    if args.mode in ("all", "lofo"):
        run_phase4_lofo_experiment(
            data_path=Path(args.data_path),
            output_dir=Path(args.output_dir),
            save_dir=Path(args.save_dir),
            baseline_models_dir=Path(args.baseline_models_dir),
            baseline_tables_dir=Path(args.baseline_tables_dir),
            seeds=args.seeds,
            sample_size=args.sample_size,
        )


if __name__ == "__main__":
    main()
