"""Adversarial unit tests for Stage 3 stress test runners (noise & leave-one-out).

Verifies:
1. `run_gaussian_noise_experiment(..., use_saved_models=True)` loads saved
   model + preprocessor bundles with ZERO retraining and ZERO preprocessor
   refitting (enforced via monkeypatched AssertionError traps), reconstructs
   the exact test split, clips noisy features to [-10, 10] in float32, and
   saves `precision` alongside `f1`, `pr_auc`, `recall`, `fpr_at_0.5` in both
   CSV and JSON.
2. Clean fallback when `use_saved_models=False` or saved bundles are absent.
3. `run_leave_one_out_experiment(..., save_models=True)` strictly withholds
   the target family from train & val (leaving test intact), fits preprocessor
   & binary C=2 weights strictly on filtered train, passes `device` ONLY to
   XGBoost, persists `preprocessor.joblib`, `metadata.json`, and
   `<model>.joblib` under `cicids2017/leave_one_out/<family>/seed_<seed>/`
   reloadable via `load_model_bundle`, and computes per-family `test_flows`
   plus Table 11's `overall_seen_attack_recall` and `overall_benign_fpr`
   (both pooled and `per_seed_mean`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.data.split_preprocess import LeakageFreePreprocessor, prepare_splits
from src.evaluation.metrics import evaluate_binary_metrics
from src.experiments import run_leave_one_out, run_noise_test
from src.models.persistence import (
    load_model_bundle,
    save_model_artifact,
    save_seed_artifacts,
)
from src.models.tree_baselines import predict_baseline_model, train_baseline_model


def _make_synthetic_cicids_df(seed: int = 2026) -> pd.DataFrame:
    """Create a separable synthetic CICIDS2017-like DataFrame with Destination Port."""
    rng = np.random.RandomState(seed)
    counts = {
        "Benign": 120,
        "DoS": 60,
        "PortScan": 40,
        "Web": 30,
    }
    centers = {
        "Benign": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "DoS": np.array([4.0, 3.0, 0.0, 0.0], dtype=np.float32),
        "PortScan": np.array([0.0, 0.0, 4.0, 3.0], dtype=np.float32),
        "Web": np.array([-3.5, 3.5, -3.5, 3.5], dtype=np.float32),
    }

    blocks = []
    labels = []
    for cls_name, n in counts.items():
        # Include one positive heavy-tailed feature to trigger signed-log selection
        f_pos = rng.exponential(scale=30.0, size=(n, 1)).astype(np.float32)
        if cls_name != "Benign":
            f_pos = f_pos + 100.0
        f_rest = (
            rng.normal(loc=0.0, scale=0.6, size=(n, 4)).astype(np.float32)
            + centers[cls_name]
        )
        blocks.append(np.hstack([f_pos, f_rest]))
        labels.extend([cls_name] * n)

    X = np.vstack(blocks)
    n_total = len(labels)
    return pd.DataFrame(
        {
            "Flow Duration": X[:, 0],
            "feat_1": X[:, 1],
            "feat_2": X[:, 2],
            "feat_3": X[:, 3],
            "feat_4": X[:, 4],
            "Destination Port": rng.randint(1, 65535, size=n_total).astype(np.float32),
            "label_multiclass": labels,
        }
    )


def test_noise_test_saved_bundle_zero_retraining_zero_refitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saved-model noise test must NEVER call train_baseline_model or Preprocessor.fit."""
    df = _make_synthetic_cicids_df(seed=101)
    parquet_path = tmp_path / "cicids2017_synth.parquet"
    df.to_parquet(parquet_path, index=False)

    models_dir = tmp_path / "saved_models"
    seeds = [0, 1]
    models = ["xgboost", "lightgbm"]
    sigmas = [0.0, 0.1, 0.25, 0.5]

    # Pre-train and persist bundles for both seeds and models
    expected_clean_metrics: dict[tuple[str, int], dict[str, float]] = {}
    for seed in seeds:
        splits = prepare_splits(df, seed=seed, exclude_destination_port=True)
        save_seed_artifacts(
            models_dir=models_dir,
            dataset="cicids2017",
            task="binary",
            seed=seed,
            preprocessor=splits.preprocessor,
            feature_names=splits.feature_names,
            label_mapping=splits.label_mapping,
            inverse_label_mapping=splits.inverse_label_mapping,
            exclude_destination_port=True,
        )
        for model_name in models:
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
                n_estimators=15,
            )
            save_model_artifact(
                models_dir=models_dir,
                dataset="cicids2017",
                task="binary",
                seed=seed,
                model_name=model_name,
                model=model,
            )
            _, y_prob = predict_baseline_model(
                model, splits.X_test, task="binary", threshold=0.5
            )
            expected_clean_metrics[(model_name, seed)] = evaluate_binary_metrics(
                splits.y_test_binary, y_prob, threshold=0.5
            )

    # Plant strict traps: ANY retraining or refitting MUST fail immediately
    def _forbid_train(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("train_baseline_model must NOT be called with saved bundles!")

    def _forbid_fit(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "LeakageFreePreprocessor.fit must NOT be called with saved bundles!"
        )

    def _forbid_prepare_splits(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("prepare_splits must NOT be called with saved bundles!")

    monkeypatch.setattr(run_noise_test, "train_baseline_model", _forbid_train)
    monkeypatch.setattr(LeakageFreePreprocessor, "fit", _forbid_fit)
    monkeypatch.setattr(run_noise_test, "prepare_splits", _forbid_prepare_splits)

    out_dir = tmp_path / "tables"
    res = run_noise_test.run_gaussian_noise_experiment(
        data_path=parquet_path,
        output_dir=out_dir,
        models=models,
        seeds=seeds,
        sigmas=sigmas,
        n_test_samples=100_000,
        exclude_destination_port=True,
        models_dir=models_dir,
        use_saved_models=True,
    )

    csv_path = out_dir / "noise_test_detailed.csv"
    json_path = out_dir / "noise_test_summary.json"
    assert csv_path.exists()
    assert json_path.exists()

    detailed_df = pd.read_csv(csv_path)
    assert len(detailed_df) == len(models) * len(seeds) * len(sigmas)
    for col in ("model", "seed", "sigma", "f1", "precision", "pr_auc", "recall", "fpr_at_0.5"):
        assert col in detailed_df.columns

    # Exact bit-level match on sigma=0.0 proves test_idx and saved preprocessor match prepare_splits
    for (model_name, seed), exp_m in expected_clean_metrics.items():
        row = detailed_df[
            (detailed_df["model"] == model_name)
            & (detailed_df["seed"] == seed)
            & (detailed_df["sigma"] == 0.0)
        ].iloc[0]
        assert abs(float(row["f1"]) - exp_m["f1"]) < 1e-6
        assert abs(float(row["precision"]) - exp_m["precision"]) < 1e-6
        assert abs(float(row["pr_auc"]) - exp_m["pr_auc"]) < 1e-6
        assert abs(float(row["recall"]) - exp_m["recall"]) < 1e-6

    with open(json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary == res["summary"]
    for m in models:
        for s in sigmas:
            entry = summary["models"][m][f"sigma_{s}"]
            for metric_key in ("f1", "precision", "pr_auc", "recall", "fpr_at_0.5"):
                assert metric_key in entry
                assert 0.0 <= entry[metric_key]["mean"] <= 100.0
                assert " ± " in entry[metric_key]["formatted"]


def test_noise_test_fallback_when_bundle_missing_or_disabled(tmp_path: Path) -> None:
    """Noise test falls back cleanly to prepare_splits + training when bundle is missing."""
    df = _make_synthetic_cicids_df(seed=202)
    parquet_path = tmp_path / "cicids2017_fb.parquet"
    df.to_parquet(parquet_path, index=False)

    out_dir = tmp_path / "tables_fb"
    res = run_noise_test.run_gaussian_noise_experiment(
        data_path=parquet_path,
        output_dir=out_dir,
        models=["xgboost"],
        seeds=[0],
        sigmas=[0.0, 0.1],
        models_dir=tmp_path / "nonexistent_saved_models",
        use_saved_models=True,
    )

    assert len(res["records"]) == 2
    assert "precision" in res["summary"]["models"]["xgboost"]["sigma_0.1"]
    assert run_noise_test.DEFAULT_MODELS == ["xgboost", "lightgbm", "random_forest"]


def test_leave_one_out_isolation_persistence_and_overall_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify LOFO family isolation, C=2 weights, device routing, persistence, and Table 11 summary."""
    df = _make_synthetic_cicids_df(seed=303)
    parquet_path = tmp_path / "cicids2017_lofo.parquet"
    df.to_parquet(parquet_path, index=False)

    models_dir = tmp_path / "saved_models"
    out_dir = tmp_path / "tables_lofo"
    held_out_families = ["DoS", "PortScan"]
    models = ["xgboost", "lightgbm"]
    seeds = [0, 1]

    # Spy on train_baseline_model to verify isolation, C=2 sample weights, and device routing
    train_calls: list[dict[str, Any]] = []
    real_train = train_baseline_model

    def _spy_train(*args: Any, **kwargs: Any) -> Any:
        train_calls.append(
            {
                "model_name": kwargs.get("model_name"),
                "n_train": len(kwargs["y_train"]),
                "n_val": len(kwargs["y_val"]),
                "has_device_kwarg": "device" in kwargs,
                "device": kwargs.get("device"),
                "unique_y_train": set(np.unique(kwargs["y_train"]).tolist()),
                "unique_weights": sorted(
                    set(np.round(kwargs["sample_weight_train"], 6).tolist())
                ),
            }
        )
        kwargs["n_estimators"] = 12
        return real_train(*args, **kwargs)

    monkeypatch.setattr(run_leave_one_out, "train_baseline_model", _spy_train)

    res = run_leave_one_out.run_leave_one_out_experiment(
        data_path=parquet_path,
        output_dir=out_dir,
        models=models,
        seeds=seeds,
        held_out_families=held_out_families,
        exclude_destination_port=True,
        save_models=True,
        models_dir=models_dir,
        device="cpu",
    )

    # 2 families * 2 seeds * 2 models = 8 training calls
    assert len(train_calls) == 8

    # Verify device routing: passed ONLY to xgboost, NEVER to lightgbm
    for call in train_calls:
        assert call["unique_y_train"] == {0, 1}
        # Binary C=2 yields exactly 2 distinct sample weights
        assert len(call["unique_weights"]) == 2
        if call["model_name"] == "xgboost":
            assert call["has_device_kwarg"] is True
            assert call["device"] == "cpu"
        else:
            assert call["has_device_kwarg"] is False

    # Verify strict held-out isolation on train & val counts
    # Total N=250: 60% train=150, 20% val=50, 20% test=50.
    # DoS has 60 rows -> 36 in train, 12 in val, 12 in test.
    # Filtered DoS train = 150 - 36 = 114; filtered val = 50 - 12 = 38.
    dos_calls = train_calls[:4]
    for call in dos_calls:
        assert call["n_train"] == 114
        assert call["n_val"] == 38

    # PortScan has 40 rows -> 24 in train, 8 in val, 8 in test.
    # Filtered PortScan train = 150 - 24 = 126; filtered val = 50 - 8 = 42.
    ps_calls = train_calls[4:]
    for call in ps_calls:
        assert call["n_train"] == 126
        assert call["n_val"] == 42

    # Verify saved model artifacts and reloadability via load_model_bundle
    for fam in held_out_families:
        for seed in seeds:
            seed_dir = (
                models_dir / "cicids2017" / "leave_one_out" / fam / f"seed_{seed}"
            )
            assert (seed_dir / "preprocessor.joblib").exists()
            assert (seed_dir / "metadata.json").exists()
            for model_name in models:
                assert (seed_dir / f"{model_name}.joblib").exists()

            bundle = load_model_bundle(
                models_dir=models_dir,
                dataset="cicids2017",
                task=f"leave_one_out/{fam}",
                seed=seed,
                model_name="xgboost",
            )
            meta = bundle["metadata"]
            assert meta["exclude_destination_port"] is True
            assert "Destination Port" not in meta["feature_names"]
            assert meta["models_saved"] == ["xgboost", "lightgbm"]

            # Reloaded preprocessor & model predict cleanly on float32 raw features
            X_sample = df[meta["feature_names"]].iloc[:10].to_numpy(dtype=np.float32)
            X_scaled = bundle["preprocessor"].transform(X_sample)
            assert X_scaled.dtype == np.float32
            y_pred, y_prob = predict_baseline_model(
                bundle["model"], X_scaled, task="binary", threshold=0.5
            )
            assert y_pred.shape == (10,)
            assert y_prob.shape == (10,)

    # Verify CSV and JSON outputs including test_flows and Table 11 overall rows
    csv_path = out_dir / "leave_one_out_detailed.csv"
    json_path = out_dir / "leave_one_out_summary.json"
    assert csv_path.exists()
    assert json_path.exists()

    detailed_df = pd.read_csv(csv_path)
    assert len(detailed_df) == 8

    with open(json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary == res["summary"]

    expected_test_flows = {"DoS": 12, "PortScan": 8}
    for m in models:
        m_summary = summary["models"][m]
        for fam, exp_flows in expected_test_flows.items():
            assert m_summary[fam]["test_flows"] == exp_flows
            for k in ("held_out_recall", "seen_attack_recall", "benign_fpr"):
                assert 0.0 <= m_summary[fam][k]["mean"] <= 100.0
                assert " ± " in m_summary[fam][k]["formatted"]

        # Verify Table 11 overall summary rows (pooled + per_seed_mean)
        m_df = detailed_df[detailed_df["model"] == m]
        for overall_key, csv_col in (
            ("overall_seen_attack_recall", "seen_attack_recall"),
            ("overall_benign_fpr", "benign_fpr"),
        ):
            assert overall_key in m_summary
            overall_entry = m_summary[overall_key]
            assert overall_entry["n_runs"] == len(held_out_families) * len(seeds)

            expected_pooled_mean = float(m_df[csv_col].mean() * 100.0)
            expected_pooled_std = float(m_df[csv_col].std(ddof=1) * 100.0)
            assert overall_entry["mean"] == pytest.approx(expected_pooled_mean, abs=1e-6)
            assert overall_entry["std"] == pytest.approx(expected_pooled_std, abs=1e-6)

            per_seed = overall_entry["per_seed_mean"]
            assert len(per_seed["values"]) == len(seeds)
            seed_means = [
                float(m_df[m_df["seed"] == s][csv_col].mean() * 100.0) for s in seeds
            ]
            assert per_seed["mean"] == pytest.approx(float(np.mean(seed_means)), abs=1e-6)
            assert per_seed["std"] == pytest.approx(
                float(np.std(seed_means, ddof=1)), abs=1e-6
            )
