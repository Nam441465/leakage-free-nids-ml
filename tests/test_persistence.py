"""Adversarial unit tests for model, preprocessor, and metadata persistence.

Verifies:
1. Bit-for-bit round-trip fidelity (save -> del -> load) for XGBoost, LightGBM,
   and Random Forest on both binary and multiclass tasks.
2. Preprocessor round-trip fidelity (exact float32 transform equivalence).
3. Metadata schema, int-key restoration for inverse_label_mapping, preservation
   and deduplication of models_saved across repeated calls.
4. Error handling when required artifact files are missing.
5. End-to-end integration with run_benchmark_for_dataset(..., save_models=True)
   and backward-compatible default (save_models=False).
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.split_preprocess import prepare_splits
from src.experiments.run_main import run_benchmark_for_dataset
from src.models.persistence import (
    get_seed_dir,
    load_model_bundle,
    save_model_artifact,
    save_seed_artifacts,
)
from src.models.tree_baselines import predict_baseline_model, train_baseline_model


def _make_synthetic_df(seed: int = 123, n_benign: int = 90, n_dos: int = 45, n_web: int = 45) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    n_total = n_benign + n_dos + n_web
    # Include both non-negative high-p99 columns (log-transformed) and signed columns
    X_benign = np.column_stack(
        [
            rng.exponential(scale=50.0, size=n_benign),
            rng.normal(loc=0.0, scale=0.8, size=(n_benign, 3)),
        ]
    ).astype(np.float32)
    X_dos = np.column_stack(
        [
            rng.exponential(scale=120.0, size=n_dos),
            rng.normal(loc=3.0, scale=0.8, size=(n_dos, 3)),
        ]
    ).astype(np.float32)
    X_web = np.column_stack(
        [
            rng.exponential(scale=10.0, size=n_web),
            rng.normal(loc=-3.0, scale=0.8, size=(n_web, 3)),
        ]
    ).astype(np.float32)
    X = np.vstack([X_benign, X_dos, X_web])

    return pd.DataFrame(
        {
            "f_pos_skew": X[:, 0],
            "f_signed_1": X[:, 1],
            "f_signed_2": X[:, 2],
            "f_signed_3": X[:, 3],
            "Destination Port": rng.randint(1, 65535, size=n_total).astype(np.float32),
            "label_multiclass": ["Benign"] * n_benign + ["DoS"] * n_dos + ["Web"] * n_web,
        }
    )


def test_get_seed_dir_layout(tmp_path: Path) -> None:
    """Verify canonical seed directory layout for both str and Path roots."""
    expected = tmp_path / "cicids2017" / "binary" / "seed_2"
    assert get_seed_dir(tmp_path, "cicids2017", "binary", 2) == expected
    assert get_seed_dir(str(tmp_path), "cicids2017", "binary", 2) == expected


@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_roundtrip_all_models_bit_for_bit(tmp_path: Path, task: str) -> None:
    """Save -> del -> load all 3 models on binary & multiclass; assert bit-for-bit outputs."""
    df = _make_synthetic_df(seed=42)
    splits = prepare_splits(df, seed=0, exclude_destination_port=True)

    y_train = splits.y_train_binary if task == "binary" else splits.y_train_multiclass
    y_val = splits.y_val_binary if task == "binary" else splits.y_val_multiclass
    sw_train = (
        splits.sample_weight_train_binary
        if task == "binary"
        else splits.sample_weight_train
    )
    sw_val = (
        splits.sample_weight_val_binary if task == "binary" else splits.sample_weight_val
    )

    models_root = tmp_path / "saved_models"
    seed_dir = save_seed_artifacts(
        models_dir=models_root,
        dataset="synth_ds",
        task=task,
        seed=0,
        preprocessor=splits.preprocessor,
        feature_names=splits.feature_names,
        label_mapping=splits.label_mapping,
        inverse_label_mapping=splits.inverse_label_mapping,
        exclude_destination_port=True,
    )
    assert (seed_dir / "preprocessor.joblib").exists()
    assert (seed_dir / "metadata.json").exists()

    # Raw feature matrix for preprocessor equivalence check
    X_raw_check = df[splits.feature_names].to_numpy(dtype=np.float32)
    orig_transformed = splits.preprocessor.transform(X_raw_check)

    expected_models = ["xgboost", "lightgbm", "random_forest"]
    for idx, model_name in enumerate(expected_models):
        model = train_baseline_model(
            model_name=model_name,
            X_train=splits.X_train,
            y_train=y_train,
            X_val=splits.X_val,
            y_val=y_val,
            task=task,
            seed=0,
            sample_weight_train=sw_train,
            sample_weight_val=sw_val,
            n_estimators=15,
        )
        orig_direct_proba = model.predict_proba(splits.X_test)
        orig_direct_pred = model.predict(splits.X_test)
        orig_pred, orig_prob = predict_baseline_model(model, splits.X_test, task=task)

        model_path = save_model_artifact(
            models_dir=models_root,
            dataset="synth_ds",
            task=task,
            seed=0,
            model_name=model_name,
            model=model,
        )
        assert model_path == seed_dir / f"{model_name}.joblib"
        assert model_path.exists()

        # Destroy in-memory model before reloading
        del model
        gc.collect()

        bundle = load_model_bundle(
            models_dir=models_root,
            dataset="synth_ds",
            task=task,
            seed=0,
            model_name=model_name,
        )
        assert bundle["model_path"] == model_path
        assert bundle["preprocessor_path"] == seed_dir / "preprocessor.joblib"

        loaded_model = bundle["model"]
        loaded_prep = bundle["preprocessor"]
        loaded_meta = bundle["metadata"]

        # Bit-for-bit identical model predictions and probabilities
        loaded_direct_proba = loaded_model.predict_proba(splits.X_test)
        loaded_direct_pred = loaded_model.predict(splits.X_test)
        loaded_pred, loaded_prob = predict_baseline_model(
            loaded_model, splits.X_test, task=task
        )

        np.testing.assert_allclose(loaded_direct_proba, orig_direct_proba, atol=0, rtol=0)
        np.testing.assert_array_equal(loaded_direct_pred, orig_direct_pred)
        np.testing.assert_allclose(loaded_prob, orig_prob, atol=0, rtol=0)
        np.testing.assert_array_equal(loaded_pred, orig_pred)

        # Bit-for-bit identical preprocessor output in float32
        loaded_transformed = loaded_prep.transform(X_raw_check)
        assert loaded_transformed.dtype == np.float32
        np.testing.assert_array_equal(loaded_transformed, orig_transformed)

        # Metadata integrity & int-key restoration
        assert loaded_meta["dataset"] == "synth_ds"
        assert loaded_meta["task"] == task
        assert loaded_meta["seed"] == 0
        assert loaded_meta["exclude_destination_port"] is True
        assert loaded_meta["feature_names"] == splits.feature_names
        assert loaded_meta["label_mapping"] == splits.label_mapping
        assert loaded_meta["inverse_label_mapping"] == splits.inverse_label_mapping
        assert all(isinstance(k, int) for k in loaded_meta["inverse_label_mapping"])
        assert loaded_meta["models_saved"] == expected_models[: idx + 1]


def test_metadata_preserves_and_deduplicates_models_saved(tmp_path: Path) -> None:
    """Re-saving seed artifacts or an existing model preserves order without duplicates."""
    df = _make_synthetic_df(seed=7)
    splits = prepare_splits(df, seed=1, exclude_destination_port=True)
    models_root = tmp_path / "saved_models"

    seed_dir = save_seed_artifacts(
        models_dir=models_root,
        dataset="ds_test",
        task="binary",
        seed=1,
        preprocessor=splits.preprocessor,
        feature_names=splits.feature_names,
        label_mapping=splits.label_mapping,
        inverse_label_mapping=splits.inverse_label_mapping,
    )

    dummy_model = {"weights": [1, 2, 3]}
    save_model_artifact(models_root, "ds_test", "binary", 1, "xgboost", dummy_model)
    save_model_artifact(models_root, "ds_test", "binary", 1, "lightgbm", dummy_model)
    # Re-save xgboost -> must remain deduplicated and keep original order
    save_model_artifact(models_root, "ds_test", "binary", 1, "xgboost", dummy_model)

    # Re-save seed artifacts -> must preserve existing models_saved list
    save_seed_artifacts(
        models_dir=models_root,
        dataset="ds_test",
        task="binary",
        seed=1,
        preprocessor=splits.preprocessor,
        feature_names=splits.feature_names,
        label_mapping=splits.label_mapping,
        inverse_label_mapping=splits.inverse_label_mapping,
    )

    with open(seed_dir / "metadata.json", "r", encoding="utf-8") as f:
        raw_meta = json.load(f)

    assert raw_meta["models_saved"] == ["xgboost", "lightgbm"]
    # Raw JSON must serialize inverse_label_mapping keys as strings
    assert all(isinstance(k, str) for k in raw_meta["inverse_label_mapping"])

    bundle = load_model_bundle(models_root, "ds_test", "binary", 1, "xgboost")
    assert all(isinstance(k, int) for k in bundle["metadata"]["inverse_label_mapping"])
    assert bundle["metadata"]["models_saved"] == ["xgboost", "lightgbm"]


def test_load_model_bundle_missing_artifacts_raises(tmp_path: Path) -> None:
    """load_model_bundle raises FileNotFoundError when any required file is missing."""
    df = _make_synthetic_df(seed=11)
    splits = prepare_splits(df, seed=0)
    models_root = tmp_path / "saved_models"

    # 1. Nothing saved -> missing model
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        load_model_bundle(models_root, "ds", "binary", 0, "xgboost")

    # 2. Only model saved -> missing preprocessor
    save_model_artifact(models_root, "ds", "binary", 0, "xgboost", {"ok": True})
    with pytest.raises(FileNotFoundError, match="Preprocessor artifact not found"):
        load_model_bundle(models_root, "ds", "binary", 0, "xgboost")

    # 3. Preprocessor and model saved, metadata removed -> missing metadata
    seed_dir = save_seed_artifacts(
        models_dir=models_root,
        dataset="ds",
        task="binary",
        seed=0,
        preprocessor=splits.preprocessor,
        feature_names=splits.feature_names,
        label_mapping=splits.label_mapping,
        inverse_label_mapping=splits.inverse_label_mapping,
    )
    (seed_dir / "metadata.json").unlink()
    with pytest.raises(FileNotFoundError, match="Metadata file not found"):
        load_model_bundle(models_root, "ds", "binary", 0, "xgboost")


def test_run_benchmark_for_dataset_save_models(tmp_path: Path) -> None:
    """run_benchmark_for_dataset(..., save_models=True) persists all artifacts cleanly."""
    df = _make_synthetic_df(seed=2026)
    parquet_path = tmp_path / "tiny_persist.parquet"
    df.to_parquet(parquet_path, index=False)

    out_dir = tmp_path / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = tmp_path / "saved_models"

    tasks = ["binary", "multiclass"]
    models = ["xgboost", "lightgbm", "random_forest"]
    seeds = [0, 1]

    run_benchmark_for_dataset(
        dataset_name="tiny_persist",
        data_path=parquet_path,
        tasks=tasks,
        models=models,
        seeds=seeds,
        output_dir=out_dir,
        exclude_destination_port=True,
        save_models=True,
        models_dir=models_dir,
    )

    for task in tasks:
        for seed in seeds:
            seed_dir = get_seed_dir(models_dir, "tiny_persist", task, seed)
            assert (seed_dir / "preprocessor.joblib").exists()
            assert (seed_dir / "metadata.json").exists()
            for model_name in models:
                assert (seed_dir / f"{model_name}.joblib").exists()

                bundle = load_model_bundle(
                    models_dir=models_dir,
                    dataset="tiny_persist",
                    task=task,
                    seed=seed,
                    model_name=model_name,
                )
                meta = bundle["metadata"]
                assert meta["dataset"] == "tiny_persist"
                assert meta["task"] == task
                assert meta["seed"] == seed
                assert meta["exclude_destination_port"] is True
                assert "Destination Port" not in meta["feature_names"]
                assert meta["models_saved"] == models
                assert all(isinstance(k, int) for k in meta["inverse_label_mapping"])

                # Confirm loaded preprocessor + model run end-to-end on raw features
                X_raw = df[meta["feature_names"]].to_numpy(dtype=np.float32)[:10]
                X_trans = bundle["preprocessor"].transform(X_raw)
                y_pred, y_prob = predict_baseline_model(
                    bundle["model"], X_trans, task=task
                )
                assert len(y_pred) == 10
                if task == "binary":
                    assert y_prob.shape == (10,)
                else:
                    assert y_prob.shape == (10, 3)


def test_run_benchmark_for_dataset_default_no_save(tmp_path: Path) -> None:
    """Default save_models=False preserves backward compatibility and writes no model files."""
    df = _make_synthetic_df(seed=99)
    parquet_path = tmp_path / "tiny_nosave.parquet"
    df.to_parquet(parquet_path, index=False)

    out_dir = tmp_path / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    unused_models_dir = tmp_path / "unused_saved_models"

    run_benchmark_for_dataset(
        dataset_name="tiny_nosave",
        data_path=parquet_path,
        tasks=["binary"],
        models=["xgboost"],
        seeds=[0],
        output_dir=out_dir,
        exclude_destination_port=True,
        models_dir=unused_models_dir,
    )

    assert not unused_models_dir.exists()
