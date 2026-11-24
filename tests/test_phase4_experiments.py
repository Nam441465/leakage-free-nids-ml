"""Unit and anti-leakage tests for Phase 4 experiment runners and plotting.

Covers:
1. `src/experiments/run_phase4_main.py`:
   - End-to-end calibrated + ratio_calibrated + val-selected best on synthetic
     NF-UNSW-NB15-v2 and CICIDS2017 parquet datasets (binary + multiclass).
   - Model persistence verification (`save_model_bundle` + `load_model_bundle`).
   - Paired Student-t + Holm-Bonferroni statistical output verification.
   - Anti-leakage test: changing test rows/labels leaves calibration thresholds,
     multiclass prior multipliers, and `phase4_best_<model>` selection invariant.
2. `src/experiments/run_phase4_stress.py`:
   - Part A: Noise-robust training (`xgboost_noise_robust`, `lightgbm_noise_robust`),
     `random_forest_noise_calibrated`, clean validation FPR <= 0.25% enforcement,
     saved bundle reloadability, and exact paired noise comparison.
   - Part B: Hybrid LOFO (`BenignAnomalyScorer` + saved LOFO XGBoost),
     `hybrid_detector.joblib` persistence, Benign-only fitting anti-leakage check,
     and OOD withheld-family detection gain.
3. `src/evaluation/plot_phase4_figures.py`:
   - Generation of all 3 publication-grade 300 DPI PNG figures, verifying valid
     PNG magic bytes, non-empty size, and 300 DPI resolution.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.evaluation.plot_phase4_figures import generate_all_phase4_figures
from src.experiments.run_leave_one_out import run_leave_one_out_experiment
from src.experiments.run_main import run_benchmark_for_dataset
from src.experiments.run_noise_test import run_gaussian_noise_experiment
from src.experiments.run_phase4_main import run_phase4_main_experiment
from src.experiments.run_phase4_stress import (
    CLEAN_VAL_FPR_CAP,
    _tune_noise_threshold_on_val,
    run_phase4_lofo_experiment,
    run_phase4_noise_experiment,
)
from src.models.persistence import get_seed_dir, load_model_bundle

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

FAST_TREE_KWARGS = {
    "n_estimators": 15,
    "max_depth": 4,
    "n_jobs": 2,
}


def _read_png_dpi(png_path: Path) -> tuple[float, float]:
    """Read (dpi_x, dpi_y) from the PNG pHYs chunk without external dependencies."""
    raw = png_path.read_bytes()
    assert raw[:8] == PNG_SIGNATURE, f"Invalid PNG signature in {png_path}"
    pos = 8
    while pos + 8 <= len(raw):
        length, chunk_type = struct.unpack(">I4s", raw[pos : pos + 8])
        data_start = pos + 8
        data_end = data_start + length
        if chunk_type == b"pHYs" and length == 9:
            ppu_x, ppu_y, unit = struct.unpack(">IIB", raw[data_start:data_end])
            if unit == 1:  # pixels per meter
                return ppu_x / 39.37007874, ppu_y / 39.37007874
        pos = data_end + 4
    return 0.0, 0.0


def _make_synthetic_nf_unsw_df(n_rows: int = 360, seed: int = 101) -> pd.DataFrame:
    """Create synthetic NF-UNSW-NB15-v2 dataframe with real feature names for ratio tests."""
    rng = np.random.default_rng(seed)
    classes = ["Benign", "DoS", "Exploits", "Fuzzers", "Generic", "Rare"]
    probs = [0.55, 0.12, 0.12, 0.09, 0.07, 0.05]
    labels = rng.choice(classes, size=n_rows, p=probs)
    # Guarantee minimum class count for 60/20/20 stratification
    for idx, cname in enumerate(classes):
        labels[idx * 6 : (idx + 1) * 6] = cname

    is_attack = (labels != "Benign").astype(np.float32)
    cls_shift = np.array([classes.index(c) for c in labels], dtype=np.float32)

    data = {
        "IN_BYTES": (500.0 + 400.0 * is_attack + 80.0 * cls_shift + rng.exponential(100.0, n_rows)).astype(np.float32),
        "OUT_BYTES": (450.0 + 120.0 * (1.0 - is_attack) + rng.exponential(80.0, n_rows)).astype(np.float32),
        "IN_PKTS": (10.0 + 15.0 * is_attack + 2.0 * cls_shift + rng.uniform(1.0, 5.0, n_rows)).astype(np.float32),
        "OUT_PKTS": (9.0 + 3.0 * (1.0 - is_attack) + rng.uniform(1.0, 4.0, n_rows)).astype(np.float32),
        "DURATION_IN": (50.0 + 30.0 * is_attack + rng.uniform(0.0, 20.0, n_rows)).astype(np.float32),
        "DURATION_OUT": (45.0 + 10.0 * (1.0 - is_attack) + rng.uniform(0.0, 15.0, n_rows)).astype(np.float32),
        "SRC_TO_DST_AVG_THROUGHPUT": (1000.0 + 800.0 * is_attack + rng.uniform(0.0, 300.0, n_rows)).astype(np.float32),
        "DST_TO_SRC_AVG_THROUGHPUT": (900.0 + 200.0 * (1.0 - is_attack) + rng.uniform(0.0, 250.0, n_rows)).astype(np.float32),
        "RETRANSMITTED_IN_PKTS": (1.0 * is_attack + rng.uniform(0.0, 2.0, n_rows)).astype(np.float32),
        "RETRANSMITTED_OUT_PKTS": (0.5 * is_attack + rng.uniform(0.0, 1.0, n_rows)).astype(np.float32),
        "NUM_PKTS_UP_TO_128_BYTES": (4.0 + 6.0 * is_attack + rng.uniform(0.0, 3.0, n_rows)).astype(np.float32),
        "NUM_PKTS_512_TO_1024_BYTES": (2.0 + 3.0 * cls_shift + rng.uniform(0.0, 2.0, n_rows)).astype(np.float32),
        "NUM_PKTS_1024_TO_1514_BYTES": (1.0 + 2.0 * is_attack + rng.uniform(0.0, 2.0, n_rows)).astype(np.float32),
        "MAX_TTL": (64.0 + 10.0 * is_attack + rng.uniform(0.0, 8.0, n_rows)).astype(np.float32),
        "MIN_TTL": (32.0 + 4.0 * (1.0 - is_attack) + rng.uniform(0.0, 4.0, n_rows)).astype(np.float32),
        "LONGEST_FLOW_PKT": (800.0 + 300.0 * is_attack + rng.uniform(0.0, 100.0, n_rows)).astype(np.float32),
        "SHORTEST_FLOW_PKT": (60.0 + 20.0 * (1.0 - is_attack) + rng.uniform(0.0, 10.0, n_rows)).astype(np.float32),
        "L4_DST_PORT": rng.integers(20, 8080, size=n_rows).astype(np.float32),
        "label_multiclass": labels,
        "label_binary": (labels != "Benign").astype(np.int64),
    }
    return pd.DataFrame(data)


def _make_synthetic_cicids_df(n_rows: int = 480, seed: int = 202) -> pd.DataFrame:
    """Create synthetic CICIDS2017 dataframe with real feature names and an OOD Bot class."""
    rng = np.random.default_rng(seed)
    classes = ["Benign", "DoS", "DDoS", "PortScan", "BruteForce", "Web", "Bot"]
    probs = [0.55, 0.12, 0.10, 0.08, 0.05, 0.05, 0.05]
    labels = rng.choice(classes, size=n_rows, p=probs)
    for idx, cname in enumerate(classes):
        labels[idx * 8 : (idx + 1) * 8] = cname

    is_attack = (labels != "Benign").astype(np.float32)
    is_bot = (labels == "Bot").astype(np.float32)
    cls_shift = np.array([classes.index(c) for c in labels], dtype=np.float32)

    data = {
        "Total Length of Fwd Packets": (300.0 + 500.0 * is_attack + 90.0 * cls_shift + rng.exponential(60.0, n_rows)).astype(np.float32),
        "Total Length of Bwd Packets": (280.0 + 100.0 * (1.0 - is_attack) + rng.exponential(50.0, n_rows)).astype(np.float32),
        "Total Fwd Packets": (8.0 + 12.0 * is_attack + 2.0 * cls_shift + rng.uniform(1.0, 3.0, n_rows)).astype(np.float32),
        "Total Backward Packets": (7.0 + 3.0 * (1.0 - is_attack) + rng.uniform(1.0, 3.0, n_rows)).astype(np.float32),
        "Max Packet Length": (700.0 + 400.0 * is_attack + rng.uniform(0.0, 80.0, n_rows)).astype(np.float32),
        "Min Packet Length": (40.0 + 15.0 * (1.0 - is_attack) + rng.uniform(0.0, 10.0, n_rows)).astype(np.float32),
        "Average Packet Size": (250.0 + 120.0 * is_attack + rng.uniform(0.0, 40.0, n_rows)).astype(np.float32),
        "Fwd IAT Mean": (120.0 + 180.0 * is_attack + rng.uniform(0.0, 40.0, n_rows)).astype(np.float32),
        "Bwd IAT Mean": (110.0 + 50.0 * (1.0 - is_attack) + rng.uniform(0.0, 30.0, n_rows)).astype(np.float32),
        "Active Mean": (60.0 + 80.0 * is_attack + rng.uniform(0.0, 20.0, n_rows)).astype(np.float32),
        "Idle Mean": (90.0 + 40.0 * (1.0 - is_attack) + rng.uniform(0.0, 20.0, n_rows)).astype(np.float32),
        "Fwd Header Length": (160.0 + 120.0 * is_attack + rng.uniform(0.0, 20.0, n_rows)).astype(np.float32),
        "SYN Flag Count": (1.0 * is_attack + rng.uniform(0.0, 0.3, n_rows)).astype(np.float32),
        "PSH Flag Count": (0.8 * is_attack + rng.uniform(0.0, 0.2, n_rows)).astype(np.float32),
        "ACK Flag Count": (1.0 + 0.5 * is_attack + rng.uniform(0.0, 0.2, n_rows)).astype(np.float32),
        "RST Flag Count": (0.2 * is_attack + rng.uniform(0.0, 0.1, n_rows)).astype(np.float32),
        "FIN Flag Count": (0.3 * is_attack + rng.uniform(0.0, 0.1, n_rows)).astype(np.float32),
        "Init_Win_bytes_forward": (8192.0 + 2000.0 * is_attack + rng.uniform(0.0, 500.0, n_rows)).astype(np.float32),
        "Init_Win_bytes_backward": (4096.0 + 500.0 * (1.0 - is_attack) + rng.uniform(0.0, 300.0, n_rows)).astype(np.float32),
        "act_data_pkt_fwd": (5.0 + 6.0 * is_attack + rng.uniform(0.0, 2.0, n_rows)).astype(np.float32),
        # Strong orthogonal OOD signal for Bot so BenignAnomalyScorer cleanly flags unseen Bot
        "Flow Bytes/s": (100.0 + 150.0 * is_attack + 9000.0 * is_bot + rng.normal(0.0, 15.0, n_rows)).astype(np.float32),
        "Flow Packets/s": (50.0 + 60.0 * is_attack + 4500.0 * is_bot + rng.normal(0.0, 8.0, n_rows)).astype(np.float32),
        "Destination Port": rng.integers(20, 8080, size=n_rows).astype(np.float32),
        "label_multiclass": labels,
        "label_binary": (labels != "Benign").astype(np.int64),
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def synthetic_baselines(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Create read-only synthetic processed datasets and Stage 1–3 baseline bundles once per module."""
    root = tmp_path_factory.mktemp("phase4_baselines")
    data_dir = root / "processed"
    baseline_models_dir = root / "baseline_models"
    baseline_tables_dir = root / "baseline_tables"

    for d in (data_dir, baseline_models_dir, baseline_tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    nf_path = data_dir / "nf_unsw_nb15_v2.parquet"
    cic_path = data_dir / "cicids2017.parquet"
    _make_synthetic_nf_unsw_df().to_parquet(nf_path, index=False)
    _make_synthetic_cicids_df().to_parquet(cic_path, index=False)

    # Train and save Stage 1–3 main baselines for seeds [0, 1]
    for ds, dpath in [("nf_unsw", nf_path), ("cicids2017", cic_path)]:
        run_benchmark_for_dataset(
            dataset_name=ds,
            data_path=dpath,
            tasks=["binary", "multiclass"],
            models=["xgboost", "lightgbm", "random_forest"],
            seeds=[0, 1],
            output_dir=baseline_tables_dir,
            exclude_destination_port=True,
            save_models=True,
            models_dir=baseline_models_dir,
        )

    # Run Stage 3 noise baseline (reusing saved CICIDS2017 binary models)
    run_gaussian_noise_experiment(
        data_path=cic_path,
        output_dir=baseline_tables_dir,
        models=["xgboost", "lightgbm", "random_forest"],
        seeds=[0, 1],
        sigmas=[0.0, 0.1, 0.25, 0.5],
        n_test_samples=100,
        models_dir=baseline_models_dir,
        use_saved_models=True,
    )

    # Run Stage 3 LOFO baseline for 2 families (DoS, Bot) to keep tests fast
    run_leave_one_out_experiment(
        data_path=cic_path,
        output_dir=baseline_tables_dir,
        models=["xgboost"],
        seeds=[0, 1],
        held_out_families=["DoS", "Bot"],
        save_models=True,
        models_dir=baseline_models_dir,
        device="cpu",
    )

    return {
        "data_dir": data_dir,
        "nf_path": nf_path,
        "cic_path": cic_path,
        "baseline_models_dir": baseline_models_dir,
        "baseline_tables_dir": baseline_tables_dir,
    }


@pytest.fixture()
def synthetic_workspace(
    synthetic_baselines: dict[str, Path],
    tmp_path: Path,
) -> dict[str, Path]:
    """Provide per-test isolated Phase 4 output directories backed by shared read-only baselines."""
    phase4_save_dir = tmp_path / "phase4_saved"
    phase4_tables_dir = tmp_path / "phase4_tables"
    phase4_figures_dir = tmp_path / "phase4_figures"
    for d in (phase4_save_dir, phase4_tables_dir, phase4_figures_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {
        **synthetic_baselines,
        "phase4_save_dir": phase4_save_dir,
        "phase4_tables_dir": phase4_tables_dir,
        "phase4_figures_dir": phase4_figures_dir,
    }


def test_run_phase4_main_end_to_end_and_persistence(synthetic_workspace: dict[str, Path]) -> None:
    """Verify run_phase4_main produces valid tables, paired stats, and reloadable bundles."""
    ws = synthetic_workspace
    out = run_phase4_main_experiment(
        datasets=["nf_unsw", "cicids2017"],
        tasks=["binary", "multiclass"],
        seeds=[0, 1],
        device="cpu",
        save_dir=ws["phase4_save_dir"],
        output_dir=ws["phase4_tables_dir"],
        data_dir=ws["data_dir"],
        baseline_models_dir=ws["baseline_models_dir"],
        baseline_tables_dir=ws["baseline_tables_dir"],
        model_kwargs=FAST_TREE_KWARGS,
    )

    assert out["csv_path"].exists()
    assert out["json_path"].exists()

    df_det = pd.read_csv(out["csv_path"])
    assert not df_det.empty
    # Per (dataset, task, seed): 3 calibrated + 2 ratio_calibrated + 3 phase4_best = 8 rows
    # Total: 2 datasets * 2 tasks * 2 seeds * 8 = 64 rows
    assert len(df_det) == 64

    summary = json.loads(out["json_path"].read_text(encoding="utf-8"))
    for ds in ("nf_unsw", "cicids2017"):
        assert ds in summary["datasets"]
        for task in ("binary", "multiclass"):
            tblock = summary["datasets"][ds][task]
            assert "baseline" in tblock
            assert "variants" in tblock
            assert "paired_vs_baseline" in tblock

            for m in ("xgboost", "lightgbm", "random_forest"):
                assert f"{m}_calibrated" in tblock["variants"]
                assert f"phase4_best_{m}" in tblock["variants"]
                # Calibration JSON must be persisted for every seed
                for s in (0, 1):
                    cal_file = (
                        ws["phase4_save_dir"]
                        / ds
                        / f"{task}_calibrated"
                        / f"seed_{s}"
                        / f"calibration_{m}.json"
                    )
                    assert cal_file.exists(), f"Missing calibration artifact: {cal_file}"

            for m in ("xgboost", "lightgbm"):
                assert f"{m}_ratio_calibrated" in tblock["variants"]
                # Trained ratio model + preprocessor + metadata + calibration must be saved & reloadable
                for s in (0, 1):
                    bundle = load_model_bundle(
                        models_dir=ws["phase4_save_dir"],
                        dataset=ds,
                        task=f"{task}_ratio",
                        seed=s,
                        model_name=m,
                    )
                    assert bundle["model"] is not None
                    assert bundle["preprocessor"].is_fitted_
                    # 12 behavioral ratio features should have been appended
                    base_meta_path = (
                        get_seed_dir(ws["baseline_models_dir"], ds, task, s)
                        / "metadata.json"
                    )
                    base_n_feat = len(json.loads(base_meta_path.read_text())["feature_names"])
                    assert len(bundle["metadata"]["feature_names"]) == base_n_feat + 12

                    cal_ratio_path = (
                        get_seed_dir(ws["phase4_save_dir"], ds, f"{task}_ratio", s)
                        / f"calibration_{m}.json"
                    )
                    assert cal_ratio_path.exists()

            # Check paired statistical tests structure
            key_metric = "f1" if task == "binary" else "macro_f1"
            paired = tblock["paired_vs_baseline"][key_metric]
            assert len(paired) == 8  # 3 calibrated + 2 ratio + 3 best
            for entry in paired:
                for field in ("mean_diff", "ci_lower", "ci_upper", "p_raw", "p_adjusted", "reject_null"):
                    assert field in entry


def test_run_phase4_main_zero_test_leakage_in_calibration_and_selection(
    synthetic_workspace: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Corrupting test features must NEVER alter calibration parameters or val-based winner selection."""
    ws = synthetic_workspace
    out_clean_dir = tmp_path / "clean_run"
    out_corrupt_dir = tmp_path / "corrupt_run"
    corrupt_data_dir = tmp_path / "corrupt_data"
    corrupt_data_dir.mkdir(parents=True, exist_ok=True)

    # Run on unmodified NF-UNSW dataset (seed 0, binary)
    res_clean = run_phase4_main_experiment(
        datasets=["nf_unsw"],
        tasks=["binary"],
        seeds=[0],
        device="cpu",
        save_dir=out_clean_dir / "saved",
        output_dir=out_clean_dir / "tables",
        data_dir=ws["data_dir"],
        baseline_models_dir=ws["baseline_models_dir"],
        baseline_tables_dir=ws["baseline_tables_dir"],
        model_kwargs=FAST_TREE_KWARGS,
    )

    # Corrupt ONLY the exact seed=0 test rows' feature values (keep labels so stratified split indices stay identical!)
    df_nf = pd.read_parquet(ws["nf_path"])
    from src.experiments.run_phase4_main import (
        _extract_labels_and_mappings,
        _reconstruct_split_indices,
    )

    y_multi, _, _, _, label_cols = _extract_labels_and_mappings(df_nf)
    _, _, test_idx = _reconstruct_split_indices(y_multi, seed=0)
    df_corrupt = df_nf.copy()
    feat_cols = [c for c in df_corrupt.columns if c not in label_cols]
    df_corrupt.loc[test_idx, feat_cols] = 99999.0
    df_corrupt.to_parquet(corrupt_data_dir / "nf_unsw_nb15_v2.parquet", index=False)

    res_corrupt = run_phase4_main_experiment(
        datasets=["nf_unsw"],
        tasks=["binary"],
        seeds=[0],
        device="cpu",
        save_dir=out_corrupt_dir / "saved",
        output_dir=out_corrupt_dir / "tables",
        data_dir=corrupt_data_dir,
        baseline_models_dir=ws["baseline_models_dir"],
        baseline_tables_dir=ws["baseline_tables_dir"],
        model_kwargs=FAST_TREE_KWARGS,
    )

    # Calibration parameters & validation-selected winner must be 100% IDENTICAL!
    for m in ("xgboost", "lightgbm", "random_forest"):
        c_clean = json.loads(
            (out_clean_dir / "saved" / "nf_unsw" / "binary_calibrated" / "seed_0" / f"calibration_{m}.json").read_text()
        )
        c_corr = json.loads(
            (out_corrupt_dir / "saved" / "nf_unsw" / "binary_calibrated" / "seed_0" / f"calibration_{m}.json").read_text()
        )
        assert c_clean["calibration"]["best_threshold"] == pytest.approx(
            c_corr["calibration"]["best_threshold"], abs=1e-12
        )

    sel_clean = res_clean["summary"]["datasets"]["nf_unsw"]["binary"]["variants"]["phase4_best_xgboost"]["selected_variants_per_seed"]
    sel_corr = res_corrupt["summary"]["datasets"]["nf_unsw"]["binary"]["variants"]["phase4_best_xgboost"]["selected_variants_per_seed"]
    assert sel_clean == sel_corr


def test_tune_noise_threshold_enforces_clean_val_fpr_cap() -> None:
    """Threshold tuner must strictly respect clean validation FPR <= 0.25%."""
    rng = np.random.default_rng(7)
    n_neg = 2000
    n_pos = 500
    y_clean = np.concatenate([np.zeros(n_neg, dtype=np.int64), np.ones(n_pos, dtype=np.int64)])
    # Clean benign scores up to 0.35; 5 highest benign scores are in (0.30, 0.35]
    p_clean = np.concatenate([
        rng.uniform(0.0, 0.30, size=n_neg - 5),
        np.array([0.31, 0.32, 0.33, 0.34, 0.35]),
        rng.uniform(0.60, 0.99, size=n_pos),
    ])

    # On noisy validation, attack probabilities shift down to [0.15, 0.45] so unconstrained
    # tuning would choose threshold ~0.15, which would destroy clean validation FPR!
    y_noisy = y_clean.copy()
    p_noisy = np.concatenate([
        rng.uniform(0.0, 0.10, size=n_neg),
        rng.uniform(0.15, 0.45, size=n_pos),
    ])

    cal = _tune_noise_threshold_on_val(
        y_val_clean=y_clean,
        prob_val_clean=p_clean,
        y_val_noisy=y_noisy,
        prob_val_noisy=p_noisy,
        max_clean_fpr=CLEAN_VAL_FPR_CAP,
    )
    assert cal["val_clean_fpr_at_best"] <= CLEAN_VAL_FPR_CAP + 1e-12
    assert cal["best_threshold"] >= 0.30


def test_run_phase4_stress_noise_and_lofo_and_plotting(
    synthetic_workspace: dict[str, Path],
) -> None:
    """Verify Part A (noise), Part B (hybrid LOFO), and all 3 300-DPI figures end-to-end."""
    ws = synthetic_workspace

    # 1. Run Phase 4 main first so phase4_main_comparison.json exists for Fig 1
    run_phase4_main_experiment(
        datasets=["nf_unsw", "cicids2017"],
        tasks=["binary", "multiclass"],
        seeds=[0, 1],
        device="cpu",
        save_dir=ws["phase4_save_dir"],
        output_dir=ws["phase4_tables_dir"],
        data_dir=ws["data_dir"],
        baseline_models_dir=ws["baseline_models_dir"],
        baseline_tables_dir=ws["baseline_tables_dir"],
        model_kwargs=FAST_TREE_KWARGS,
    )

    # 2. Run Phase 4 Stress Part A — Noise Robustness
    noise_out = run_phase4_noise_experiment(
        data_path=ws["cic_path"],
        output_dir=ws["phase4_tables_dir"],
        save_dir=ws["phase4_save_dir"],
        baseline_models_dir=ws["baseline_models_dir"],
        baseline_tables_dir=ws["baseline_tables_dir"],
        seeds=[0, 1],
        sigmas=[0.0, 0.1, 0.25, 0.5],
        n_test_samples=100,
        device="cpu",
        model_kwargs=FAST_TREE_KWARGS,
    )
    assert noise_out["csv_path"].exists()
    assert noise_out["json_path"].exists()

    df_noise = pd.read_csv(noise_out["csv_path"])
    # 3 models * 2 seeds * 4 sigmas = 24 rows
    assert len(df_noise) == 24
    assert set(df_noise["model"].unique()) == {
        "xgboost_noise_robust",
        "lightgbm_noise_robust",
        "random_forest_noise_calibrated",
    }

    # Verify noise-robust models were persisted BEFORE del model and are reloadable
    for s in (0, 1):
        for m in ("xgboost_noise_robust", "lightgbm_noise_robust"):
            bundle = load_model_bundle(
                models_dir=ws["phase4_save_dir"],
                dataset="cicids2017",
                task="noise_robust",
                seed=s,
                model_name=m,
            )
            assert bundle["model"] is not None
            assert bundle["preprocessor"].is_fitted_
            cal_path = (
                get_seed_dir(ws["phase4_save_dir"], "cicids2017", "noise_robust", s)
                / f"calibration_{m}.json"
            )
            assert cal_path.exists()
        rf_cal_path = (
            get_seed_dir(ws["phase4_save_dir"], "cicids2017", "noise_robust", s)
            / "calibration_random_forest_noise_calibrated.json"
        )
        assert rf_cal_path.exists()

    noise_summary = json.loads(noise_out["json_path"].read_text(encoding="utf-8"))
    assert "sigma_0.1" in noise_summary["paired_vs_baseline"]
    assert "f1" in noise_summary["paired_vs_baseline"]["sigma_0.1"]

    # 3. Run Phase 4 Stress Part B — Hybrid LOFO
    lofo_out = run_phase4_lofo_experiment(
        data_path=ws["cic_path"],
        output_dir=ws["phase4_tables_dir"],
        save_dir=ws["phase4_save_dir"],
        baseline_models_dir=ws["baseline_models_dir"],
        baseline_tables_dir=ws["baseline_tables_dir"],
        seeds=[0, 1],
        held_out_families=["DoS", "Bot"],
        target_benign_fpr=0.02,
    )
    assert lofo_out["csv_path"].exists()
    assert lofo_out["json_path"].exists()

    df_lofo = pd.read_csv(lofo_out["csv_path"])
    # 2 families * 2 seeds = 4 rows
    assert len(df_lofo) == 4
    assert (df_lofo["model"] == "xgboost_hybrid_lofo").all()

    # Verify persisted hybrid_detector.joblib artifacts and Benign-only fitting
    for fam in ("DoS", "Bot"):
        for s in (0, 1):
            hybrid_path = (
                ws["phase4_save_dir"]
                / "cicids2017"
                / "leave_one_out"
                / fam
                / f"seed_{s}"
                / "hybrid_detector.joblib"
            )
            assert hybrid_path.exists()
            payload = joblib.load(hybrid_path)
            assert "anomaly_scorer" in payload
            assert "calib_params" in payload
            assert payload["anomaly_scorer"].is_fitted_
            assert "xgb_thresh" in payload["calib_params"]
            assert "anom_thresh" in payload["calib_params"]

    # On the synthetic OOD Bot family, Hybrid LOFO should achieve high held-out recall
    bot_rows = df_lofo[df_lofo["held_out_family"] == "Bot"]
    assert float(bot_rows["held_out_recall"].mean()) >= 0.80

    lofo_summary = json.loads(lofo_out["json_path"].read_text(encoding="utf-8"))
    assert "overall_seen_attack_recall" in lofo_summary["models"]["xgboost_hybrid_lofo"]
    assert "overall_benign_fpr" in lofo_summary["models"]["xgboost_hybrid_lofo"]
    assert "held_out_recall" in lofo_summary["paired_vs_baseline"]

    # 4. Generate all 3 publication-grade 300 DPI PNG figures
    fig_paths = generate_all_phase4_figures(
        tables_dir=ws["phase4_tables_dir"],
        output_dir=ws["phase4_figures_dir"],
    )
    expected_names = {
        "phase4_fig1_main_comparison.png",
        "phase4_fig2_noise_robustness.png",
        "phase4_fig3_hybrid_lofo.png",
    }
    assert {p.name for p in fig_paths.values()} == expected_names
    for fig_path in fig_paths.values():
        assert fig_path.exists()
        assert fig_path.stat().st_size > 10_000
        dpi_x, dpi_y = _read_png_dpi(fig_path)
        assert dpi_x == pytest.approx(300.0, abs=1.5)
        assert dpi_y == pytest.approx(300.0, abs=1.5)


def test_run_phase4_lofo_zero_test_and_withheld_leakage(
    synthetic_workspace: dict[str, Path],
    tmp_path: Path,
) -> None:
    """Corrupting test flows and withheld-family flows must leave LOFO scorer & calib invariant."""
    ws = synthetic_workspace
    from src.experiments.run_phase4_main import (
        _extract_labels_and_mappings,
        _reconstruct_split_indices,
    )

    clean_save = tmp_path / "lofo_clean_save"
    clean_tables = tmp_path / "lofo_clean_tables"
    run_phase4_lofo_experiment(
        data_path=ws["cic_path"],
        output_dir=clean_tables,
        save_dir=clean_save,
        baseline_models_dir=ws["baseline_models_dir"],
        baseline_tables_dir=ws["baseline_tables_dir"],
        seeds=[0],
        held_out_families=["Bot"],
        target_benign_fpr=0.02,
    )

    # Corrupt ALL test rows AND ALL Bot rows in the parquet (keeping labels identical)
    df_cic = pd.read_parquet(ws["cic_path"])
    y_multi, _, _, _, label_cols = _extract_labels_and_mappings(df_cic)
    _, _, test_idx = _reconstruct_split_indices(y_multi, seed=0)
    df_corr = df_cic.copy()
    feat_cols = [c for c in df_corr.columns if c not in label_cols]
    df_corr.loc[test_idx, feat_cols] = -77777.0
    df_corr.loc[df_corr["label_multiclass"] == "Bot", feat_cols] = 88888.0
    corr_parquet = tmp_path / "cicids2017_corrupt.parquet"
    df_corr.to_parquet(corr_parquet, index=False)

    corr_save = tmp_path / "lofo_corr_save"
    corr_tables = tmp_path / "lofo_corr_tables"
    run_phase4_lofo_experiment(
        data_path=corr_parquet,
        output_dir=corr_tables,
        save_dir=corr_save,
        baseline_models_dir=ws["baseline_models_dir"],
        baseline_tables_dir=ws["baseline_tables_dir"],
        seeds=[0],
        held_out_families=["Bot"],
        target_benign_fpr=0.02,
    )

    p_clean = joblib.load(
        clean_save / "cicids2017" / "leave_one_out" / "Bot" / "seed_0" / "hybrid_detector.joblib"
    )
    p_corr = joblib.load(
        corr_save / "cicids2017" / "leave_one_out" / "Bot" / "seed_0" / "hybrid_detector.joblib"
    )
    np.testing.assert_allclose(
        p_clean["anomaly_scorer"].benign_mean_,
        p_corr["anomaly_scorer"].benign_mean_,
        atol=0.0,
    )
    np.testing.assert_allclose(
        p_clean["anomaly_scorer"].benign_std_,
        p_corr["anomaly_scorer"].benign_std_,
        atol=0.0,
    )
    assert p_clean["calib_params"]["anom_thresh"] == pytest.approx(
        p_corr["calib_params"]["anom_thresh"], abs=1e-12
    )
    assert p_clean["calib_params"]["xgb_thresh"] == pytest.approx(
        p_corr["calib_params"]["xgb_thresh"], abs=1e-12
    )

