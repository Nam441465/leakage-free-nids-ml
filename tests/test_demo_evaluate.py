"""Unit and adversarial tests for live demo evaluation (`src/experiments/demo_evaluate.py`).

Tests:
1. End-to-end training with `save_models=True` -> live `evaluate_saved_model` for
   BOTH `binary` and `multiclass` across all 3 models (`xgboost`, `lightgbm`,
   `random_forest`), asserting:
   - Exact match (`abs_diff < 1e-6`) against `*_detailed_results.csv`.
   - Zero preprocessor refitting (`fit` / `fit_transform` are forbidden).
2. Adversarial test: missing saved model raises `FileNotFoundError` cleanly.
3. Adversarial test: `results_dir=None` handles absent CSV gracefully.
4. CLI test: `--model all` evaluates all 3 models sequentially.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.split_preprocess import LeakageFreePreprocessor
from src.experiments.demo_evaluate import evaluate_saved_model, main as demo_main
from src.experiments.run_main import run_benchmark_for_dataset


def _make_synthetic_nids_df(n_benign: int = 90, n_dos: int = 45, n_web: int = 45) -> pd.DataFrame:
    """Create a small, separable synthetic NIDS DataFrame matching processed parquet schema."""
    rng = np.random.RandomState(2026)
    n_total = n_benign + n_dos + n_web

    X_benign = rng.normal(loc=0.0, scale=0.8, size=(n_benign, 4)).astype(np.float32)
    X_dos = rng.normal(loc=3.0, scale=0.8, size=(n_dos, 4)).astype(np.float32)
    X_web = rng.normal(loc=-3.0, scale=0.8, size=(n_web, 4)).astype(np.float32)
    X = np.vstack([X_benign, X_dos, X_web])

    # Add one non-negative heavy-tailed feature to exercise signed-log selection
    heavy_tail = rng.exponential(scale=25.0, size=n_total).astype(np.float32)

    return pd.DataFrame(
        {
            "f0": X[:, 0],
            "f1": X[:, 1],
            "f2": X[:, 2],
            "f3": X[:, 3],
            "f_heavy": heavy_tail,
            "Destination Port": rng.randint(1, 65535, size=n_total).astype(np.float32),
            "label_multiclass": ["Benign"] * n_benign + ["DoS"] * n_dos + ["Web"] * n_web,
        }
    )


def test_evaluate_saved_model_end_to_end_binary_and_multiclass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end test: train+save -> load+evaluate matches detailed_results.csv to < 1e-6."""
    data_dir = tmp_path / "processed"
    results_dir = tmp_path / "tables"
    models_dir = tmp_path / "saved_models"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = _make_synthetic_nids_df()
    parquet_path = data_dir / "nf_unsw_nb15_v2.parquet"
    df.to_parquet(parquet_path, index=False)

    models = ["xgboost", "lightgbm", "random_forest"]
    run_benchmark_for_dataset(
        dataset_name="nf_unsw",
        data_path=parquet_path,
        tasks=["binary", "multiclass"],
        models=models,
        seeds=[0],
        output_dir=results_dir,
        exclude_destination_port=True,
        save_models=True,
        models_dir=models_dir,
    )

    # Strictly forbid any preprocessor refitting during live demo evaluation
    def _forbid_fit(*args, **kwargs):
        raise AssertionError("LeakageFreePreprocessor.fit must NEVER be called during demo evaluation!")

    monkeypatch.setattr(LeakageFreePreprocessor, "fit", _forbid_fit)
    monkeypatch.setattr(LeakageFreePreprocessor, "fit_transform", _forbid_fit)

    expected_test_samples = int(round(len(df) * 0.2))

    for task in ["binary", "multiclass"]:
        for model_name in models:
            res = evaluate_saved_model(
                dataset="nf_unsw",
                task=task,
                model_name=model_name,
                seed=0,
                models_dir=models_dir,
                data_dir=data_dir,
                results_dir=results_dir,
            )

            assert res["dataset"] == "nf_unsw"
            assert res["task"] == task
            assert res["model"] == model_name
            assert res["seed"] == 0
            assert res["n_test_samples"] == expected_test_samples
            assert res["model_size_mb"] > 0.0
            assert res["inference_time_sec"] >= 0.0
            assert Path(res["model_path"]).exists()
            assert Path(res["preprocessor_path"]).exists()

            comp = res["reported_comparison"]
            assert comp is not None
            assert comp["metric"] == ("f1" if task == "binary" else "macro_f1")
            assert comp["abs_diff"] < 1e-6
            assert comp["matched"] is True

            captured = capsys.readouterr()
            assert "MATCH (< 1e-6)" in captured.out


def test_evaluate_saved_model_missing_bundle_raises_file_not_found(tmp_path: Path) -> None:
    """Adversarial test: missing saved model must raise FileNotFoundError cleanly."""
    with pytest.raises(FileNotFoundError):
        evaluate_saved_model(
            dataset="cicids2017",
            task="binary",
            model_name="xgboost",
            seed=0,
            models_dir=tmp_path / "nonexistent_models",
            data_dir=tmp_path / "nonexistent_data",
            results_dir=None,
        )


def test_evaluate_saved_model_without_results_csv_and_cli_all(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify results_dir=None works cleanly and CLI `--model all` runs sequentially."""
    data_dir = tmp_path / "processed"
    results_dir = tmp_path / "tables"
    models_dir = tmp_path / "saved_models"
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = _make_synthetic_nids_df(n_benign=60, n_dos=30, n_web=30)
    parquet_path = data_dir / "cicids2017.parquet"
    df.to_parquet(parquet_path, index=False)

    run_benchmark_for_dataset(
        dataset_name="cicids2017",
        data_path=parquet_path,
        tasks=["binary"],
        models=["xgboost", "lightgbm", "random_forest"],
        seeds=[0],
        output_dir=results_dir,
        exclude_destination_port=True,
        save_models=True,
        models_dir=models_dir,
    )

    # Without results_dir, reported_comparison is None
    res = evaluate_saved_model(
        dataset="cicids2017",
        task="binary",
        model_name="xgboost",
        seed=0,
        models_dir=models_dir,
        data_dir=data_dir,
        results_dir=None,
    )
    assert res["reported_comparison"] is None
    assert "f1" in res["metrics"]
    capsys.readouterr()

    # CLI `--model all` evaluates all 3 models sequentially
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo_evaluate.py",
            "--dataset",
            "cicids2017",
            "--task",
            "binary",
            "--model",
            "all",
            "--seed",
            "0",
            "--models-dir",
            str(models_dir),
            "--data-dir",
            str(data_dir),
            "--results-dir",
            str(results_dir),
        ],
    )
    demo_main()
    out = capsys.readouterr().out
    assert "model=xgboost" in out
    assert "model=lightgbm" in out
    assert "model=random_forest" in out
    assert out.count("MATCH (< 1e-6)") == 3
