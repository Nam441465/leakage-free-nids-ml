"""Gaussian feature noise stress test on CICIDS2017.

Protocol from Section 4 and Table 12 of 2611.04660v1.pdf:
1. Reuse saved binary baseline models + train-fitted preprocessors from
   `saved_models/cicids2017/binary/seed_<seed>/` when available (zero refitting,
   zero retraining), or fall back cleanly to leakage-free training.
2. Reconstruct the exact deterministic 60/20/20 test split matching
   `prepare_splits`, transform ONLY test rows using the saved preprocessor, and
   subsample up to 100,000 standardized test flows (stratified on binary label).
3. For sigma in {0.0, 0.1, 0.25, 0.5}:
   - Add N(0, sigma^2) noise to standardized test features.
   - Re-clip to [-10, 10] in float32.
   - Evaluate Binary F1, Precision, PR-AUC, Recall, FPR at threshold 0.5.
4. Repeat across seeds {0, 1, 2}, aggregate mean ± SD, and save results.
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

from src.data.split_preprocess import prepare_splits
from src.evaluation.metrics import evaluate_binary_metrics
from src.evaluation.stats import compute_mean_std
from src.experiments.demo_evaluate import _extract_labels_and_test_indices
from src.models.persistence import get_seed_dir, load_model_bundle
from src.models.tree_baselines import predict_baseline_model, train_baseline_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_noise_test")

SIGMA_VALUES = [0.0, 0.1, 0.25, 0.5]
DEFAULT_MODELS = ["xgboost", "lightgbm", "random_forest"]
DEFAULT_SEEDS = [0, 1, 2]
TRACKED_METRICS = ("f1", "precision", "pr_auc", "recall", "fpr_at_0.5")


def _has_saved_bundle(
    models_dir: Path,
    seed: int,
    model_name: str,
    exclude_destination_port: bool,
) -> bool:
    """Check whether a compatible saved binary CICIDS2017 bundle exists on disk."""
    seed_dir = get_seed_dir(models_dir, "cicids2017", "binary", seed)
    model_path = seed_dir / f"{model_name}.joblib"
    preprocessor_path = seed_dir / "preprocessor.joblib"
    metadata_path = seed_dir / "metadata.json"

    if not (model_path.exists() and preprocessor_path.exists() and metadata_path.exists()):
        return False

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception:
        return False

    saved_exclude_port = metadata.get("exclude_destination_port")
    if saved_exclude_port is not None and bool(saved_exclude_port) != bool(
        exclude_destination_port
    ):
        logger.warning(
            "Saved bundle at %s has exclude_destination_port=%s (requested %s); "
            "falling back to fresh training.",
            seed_dir,
            saved_exclude_port,
            exclude_destination_port,
        )
        return False

    return True


def _subsample_test_flows(
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_test_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified subsample of standardized test flows up to n_test_samples."""
    n_test = len(y_test)
    if n_test > n_test_samples:
        indices = np.arange(n_test)
        sub_idx, _ = train_test_split(
            indices,
            train_size=n_test_samples,
            random_state=42 + seed,
            stratify=y_test,
        )
        return X_test[sub_idx], y_test[sub_idx]
    return X_test, y_test


def _evaluate_model_under_noise(
    model: Any,
    model_name: str,
    seed: int,
    X_test_sub: np.ndarray,
    y_test_sub: np.ndarray,
    sigmas: list[float],
    results_records: list[dict[str, Any]],
    summary_tracking: dict[str, dict[float, dict[str, list[float]]]],
) -> None:
    """Evaluate a binary model across Gaussian noise levels sigma."""
    for sigma in sigmas:
        logger.info(
            "Evaluating %s under Gaussian noise sigma=%.2f (seed=%d)...",
            model_name,
            sigma,
            seed,
        )
        if sigma > 0.0:
            rng = np.random.RandomState(seed=1000 + int(round(sigma * 100)) + seed)
            noise = rng.normal(0.0, sigma, size=X_test_sub.shape).astype(np.float32)
            X_noisy = np.clip(X_test_sub + noise, -10.0, 10.0).astype(np.float32)
        else:
            X_noisy = X_test_sub

        _, y_prob = predict_baseline_model(model, X_noisy, task="binary", threshold=0.5)
        metrics = evaluate_binary_metrics(y_test_sub, y_prob, threshold=0.5)

        record = {
            "model": model_name,
            "seed": seed,
            "sigma": sigma,
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "pr_auc": metrics["pr_auc"],
            "recall": metrics["recall"],
            "fpr_at_0.5": metrics["fpr_at_0.5"],
            "mcc": metrics["mcc"],
            "accuracy": metrics["accuracy"],
        }
        results_records.append(record)

        for metric_key in TRACKED_METRICS:
            summary_tracking[model_name][sigma][metric_key].append(metrics[metric_key])

        if sigma > 0.0:
            del noise, X_noisy
            gc.collect()


def run_gaussian_noise_experiment(
    data_path: Path,
    output_dir: Path,
    models: list[str] | None = None,
    seeds: list[int] | None = None,
    sigmas: list[float] | None = None,
    n_test_samples: int = 100_000,
    sample_size: int | None = None,
    exclude_destination_port: bool = True,
    models_dir: Path | None = Path("saved_models"),
    use_saved_models: bool = True,
) -> dict[str, Any]:
    """Execute Gaussian feature noise stress test across noise levels and seeds."""
    model_list = list(DEFAULT_MODELS) if models is None else list(models)
    seed_list = list(DEFAULT_SEEDS) if seeds is None else list(seeds)
    sigma_list = list(SIGMA_VALUES) if sigmas is None else list(sigmas)
    effective_models_dir = Path(models_dir) if models_dir is not None else None

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
        logger.info("Downsampling for quick verification to %d rows", sample_size)
        sampled_parts = [
            grp.sample(
                min(len(grp), max(2, int(sample_size * len(grp) / len(df)))),
                random_state=42,
            )
            for _, grp in df.groupby(label_col, sort=False)
        ]
        df = pd.concat(sampled_parts, ignore_index=True)

    results_records: list[dict[str, Any]] = []
    summary_tracking: dict[str, dict[float, dict[str, list[float]]]] = {
        m: {
            s: {metric_key: [] for metric_key in TRACKED_METRICS}
            for s in sigma_list
        }
        for m in model_list
    }

    for seed in seed_list:
        saved_models_for_seed: list[str] = []
        fallback_models_for_seed: list[str] = []

        for model_name in model_list:
            if (
                use_saved_models
                and effective_models_dir is not None
                and _has_saved_bundle(
                    effective_models_dir,
                    seed=seed,
                    model_name=model_name,
                    exclude_destination_port=exclude_destination_port,
                )
            ):
                saved_models_for_seed.append(model_name)
            else:
                fallback_models_for_seed.append(model_name)

        # Path 1: Load saved bundles (zero refitting, zero retraining)
        if saved_models_for_seed and effective_models_dir is not None:
            X_test_sub_saved: np.ndarray | None = None
            y_test_sub_saved: np.ndarray | None = None

            for model_name in saved_models_for_seed:
                logger.info(
                    "Loading saved bundle for %s (cicids2017/binary/seed_%d) from %s...",
                    model_name,
                    seed,
                    effective_models_dir,
                )
                bundle = load_model_bundle(
                    models_dir=effective_models_dir,
                    dataset="cicids2017",
                    task="binary",
                    seed=seed,
                    model_name=model_name,
                )
                model = bundle["model"]

                if X_test_sub_saved is None or y_test_sub_saved is None:
                    preprocessor = bundle["preprocessor"]
                    metadata = bundle["metadata"]
                    feature_names = list(metadata["feature_names"])
                    label_mapping = dict(metadata["label_mapping"])
                    inverse_label_mapping = {
                        int(k): str(v)
                        for k, v in metadata["inverse_label_mapping"].items()
                    }

                    test_idx, y_test_bin, _ = _extract_labels_and_test_indices(
                        df=df,
                        seed=seed,
                        label_mapping=label_mapping,
                        inverse_label_mapping=inverse_label_mapping,
                    )
                    X_test_raw = (
                        df[feature_names].iloc[test_idx].to_numpy(dtype=np.float32)
                    )
                    X_test = preprocessor.transform(X_test_raw)
                    del test_idx, X_test_raw, preprocessor
                    gc.collect()

                    X_test_sub_saved, y_test_sub_saved = _subsample_test_flows(
                        X_test, y_test_bin, n_test_samples=n_test_samples, seed=seed
                    )
                    del X_test, y_test_bin
                    gc.collect()

                _evaluate_model_under_noise(
                    model=model,
                    model_name=model_name,
                    seed=seed,
                    X_test_sub=X_test_sub_saved,
                    y_test_sub=y_test_sub_saved,
                    sigmas=sigma_list,
                    results_records=results_records,
                    summary_tracking=summary_tracking,
                )

                del model, bundle
                gc.collect()

            del X_test_sub_saved, y_test_sub_saved
            gc.collect()

        # Path 2: Fallback training if use_saved_models=False or saved bundle missing
        if fallback_models_for_seed:
            logger.info(
                "Preparing splits for fallback training of %s (seed=%d)...",
                fallback_models_for_seed,
                seed,
            )
            splits = prepare_splits(
                df,
                seed=seed,
                exclude_destination_port=exclude_destination_port,
            )

            X_test_sub_fb, y_test_sub_fb = _subsample_test_flows(
                splits.X_test,
                splits.y_test_binary,
                n_test_samples=n_test_samples,
                seed=seed,
            )

            for model_name in fallback_models_for_seed:
                logger.info("Training %s on clean data (seed=%d)...", model_name, seed)
                model = train_baseline_model(
                    model_name=model_name,
                    X_train=splits.X_train,
                    y_train=splits.y_train_binary,
                    X_val=splits.X_val,
                    y_val=splits.y_val_binary,
                    task="binary",
                    seed=seed,
                    sample_weight_train=splits.sample_weight_train_binary,
                    sample_weight_val=splits.sample_weight_val_binary,
                )

                _evaluate_model_under_noise(
                    model=model,
                    model_name=model_name,
                    seed=seed,
                    X_test_sub=X_test_sub_fb,
                    y_test_sub=y_test_sub_fb,
                    sigmas=sigma_list,
                    results_records=results_records,
                    summary_tracking=summary_tracking,
                )

                del model
                gc.collect()

            del splits, X_test_sub_fb, y_test_sub_fb
            gc.collect()

    del df
    gc.collect()

    output_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results_records)
    csv_path = output_dir / "noise_test_detailed.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Saved detailed results to %s", csv_path)

    # Compute mean ± SD (%) for each sigma
    summary: dict[str, Any] = {"models": {}}
    for m in model_list:
        summary["models"][m] = {}
        for s in sigma_list:
            s_key = f"sigma_{s}"
            summary["models"][m][s_key] = {
                metric_key: compute_mean_std(
                    summary_tracking[m][s][metric_key], as_percentage=True
                )
                for metric_key in TRACKED_METRICS
            }

    summary_path = output_dir / "noise_test_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary to %s", summary_path)

    return {"records": results_records, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Gaussian feature noise stress test on CICIDS2017."
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
        help="Models to evaluate (default: xgboost lightgbm random_forest)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="Random seeds",
    )
    parser.add_argument(
        "--n-test-samples",
        type=int,
        default=100_000,
        help="Maximum number of stratified test flows to evaluate (default: 100000)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional downsampling size for quick verification",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="saved_models",
        help="Root directory containing saved model bundles",
    )
    parser.add_argument(
        "--use-saved-models",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse saved binary models + preprocessors when available (use --no-use-saved-models to retrain)",
    )
    parser.add_argument(
        "--include-destination-port",
        action="store_true",
        default=False,
        help="Include Destination Port in features (for ablation only)",
    )

    args = parser.parse_args()
    run_gaussian_noise_experiment(
        data_path=Path(args.data_path),
        output_dir=Path(args.output_dir),
        models=args.models,
        seeds=args.seeds,
        n_test_samples=args.n_test_samples,
        sample_size=args.sample_size,
        exclude_destination_port=not args.include_destination_port,
        models_dir=Path(args.models_dir),
        use_saved_models=args.use_saved_models,
    )


if __name__ == "__main__":
    main()
