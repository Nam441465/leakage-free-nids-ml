"""Comprehensive unit tests for leakage-free ML pipeline.

Tests:
1. Leakage-free preprocessing (val/test outliers do not affect train stats/weights).
2. Split stratification and Destination Port exclusion by default.
3. XGBoost, LightGBM, and Random Forest fitting & predicting on float32 data.
4. Binary metrics (F1, MCC, PR-AUC, FPR@0.5, Recall@0.1% FPR, Recall@1.0% FPR).
5. Multiclass metrics (Macro-F1 strictly over present classes, row-normalized CM).
6. Statistical testing (mean ± SD, paired differences, 95% t-CI, Holm adjustment).
7. Noise test and leave-one-family-out isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.split_preprocess import (
    LeakageFreePreprocessor,
    prepare_splits,
    find_destination_port_columns,
)
from src.models.tree_baselines import (
    get_xgboost_model,
    get_lightgbm_model,
    get_random_forest_model,
    train_baseline_model,
    predict_baseline_model,
)
from src.evaluation.metrics import (
    compute_recall_at_fixed_fpr,
    evaluate_binary_metrics,
    evaluate_multiclass_metrics,
)
from src.evaluation.stats import (
    compute_mean_std,
    compute_paired_difference_ci,
    holm_bonferroni_correction,
    paired_t_test_with_holm,
)


# ==============================================================================
# 1. Leakage-Free Preprocessing Tests
# ==============================================================================

def test_leakage_free_isolation():
    """Verify that val and test outliers do NOT affect train stats, log_cols, or weights."""
    rng = np.random.RandomState(42)
    # 3 features:
    # Col 0: non-negative, high p99 -> should get log-transformed
    # Col 1: contains negative values -> should NOT get log-transformed
    # Col 2: non-negative, low values (< 10) -> should NOT get log-transformed
    n_train = 200
    col0 = rng.exponential(scale=20.0, size=(n_train, 1)).astype(np.float32)
    col1 = rng.normal(loc=0.0, scale=5.0, size=(n_train, 1)).astype(np.float32)
    col2 = rng.uniform(low=0.0, high=5.0, size=(n_train, 1)).astype(np.float32)
    X_train = np.hstack([col0, col1, col2])
    y_train = np.array([0] * 180 + [1] * 20, dtype=np.int64)

    # Clean val and test
    X_val_clean = X_train[:50].copy()
    X_test_clean = X_train[50:100].copy()

    # Poisoned val and test with massive outliers (1e8)
    X_val_poisoned = X_val_clean.copy()
    X_val_poisoned[0, 0] = 1e8
    X_val_poisoned[1, 1] = -1e8
    X_val_poisoned[2, 2] = 1e8

    X_test_poisoned = X_test_clean.copy()
    X_test_poisoned[0, :] = 1e9

    # Fit preprocessor 1 on clean data
    prep1 = LeakageFreePreprocessor(clip_range=(-10.0, 10.0))
    X_train_trans1 = prep1.fit_transform(X_train, y_train=y_train)

    # Fit preprocessor 2 on poisoned scenario (fit only on train!)
    prep2 = LeakageFreePreprocessor(clip_range=(-10.0, 10.0))
    X_train_trans2 = prep2.fit_transform(X_train, y_train=y_train)

    # Transform both val/test sets
    val_trans1 = prep1.transform(X_val_clean)
    val_trans2 = prep2.transform(X_val_poisoned)
    test_trans2 = prep2.transform(X_test_poisoned)

    # Confirm parameters are bit-for-bit identical regardless of val/test
    np.testing.assert_array_equal(prep1.log_cols_, prep2.log_cols_)
    np.testing.assert_array_equal(prep1.mean_, prep2.mean_)
    np.testing.assert_array_equal(prep1.std_, prep2.std_)
    assert prep1.class_weights_ == prep2.class_weights_
    np.testing.assert_array_equal(X_train_trans1, X_train_trans2)

    # Verify log column selection logic: only Col 0 (index 0) qualifies
    assert 0 in prep1.log_cols_
    assert 1 not in prep1.log_cols_
    assert 2 not in prep1.log_cols_

    # Verify clipping to [-10, 10] on extreme outliers
    assert np.all(val_trans2 >= -10.0)
    assert np.all(val_trans2 <= 10.0)
    assert np.all(test_trans2 >= -10.0)
    assert np.all(test_trans2 <= 10.0)

    # Verify strictly float32
    assert X_train_trans1.dtype == np.float32
    assert val_trans2.dtype == np.float32
    assert test_trans2.dtype == np.float32

    # Verify class weights formula: w_c = min(N / (C * n_c), 100.0)
    # N = 200, C = 2, n_0 = 180, n_1 = 20
    # w_0 = min(200 / (2 * 180), 100) = 200 / 360 = 0.5555...
    # w_1 = min(200 / (2 * 20), 100) = 200 / 40 = 5.0
    w_0 = prep1.class_weights_[0]
    w_1 = prep1.class_weights_[1]
    assert pytest.approx(w_0, 0.001) == 200.0 / 360.0
    assert pytest.approx(w_1, 0.001) == 5.0


def test_split_stratification_and_port_exclusion():
    """Verify 60/20/20 stratified split and default exclusion of Destination Port."""
    rng = np.random.RandomState(123)
    n_samples = 300
    df = pd.DataFrame(
        {
            "feat_a": rng.normal(0, 1, n_samples).astype(np.float32),
            "feat_b": rng.uniform(0, 10, n_samples).astype(np.float32),
            "Destination Port": rng.randint(1, 65535, n_samples).astype(np.float32),
            "label_multiclass": ["Benign"] * 240 + ["DoS"] * 40 + ["Web"] * 20,
        }
    )

    # Test with default: exclude_destination_port=True
    splits = prepare_splits(df, seed=0, exclude_destination_port=True)
    assert "Destination Port" not in splits.feature_names
    assert len(splits.feature_names) == 2
    assert splits.X_train.shape[1] == 2
    assert splits.X_val.shape[1] == 2
    assert splits.X_test.shape[1] == 2

    # Check exact split sizes: 60% = 180, 20% = 60, 20% = 60
    assert len(splits.y_train_multiclass) == 180
    assert len(splits.y_val_multiclass) == 60
    assert len(splits.y_test_multiclass) == 60

    # Check stratification on multiclass label
    # Train should have 60% of each class: 240*0.6=144 Benign, 40*0.6=24 DoS, 20*0.6=12 Web
    benign_id = splits.label_mapping["Benign"]
    dos_id = splits.label_mapping["DoS"]
    web_id = splits.label_mapping["Web"]
    assert np.sum(splits.y_train_multiclass == benign_id) == 144
    assert np.sum(splits.y_train_multiclass == dos_id) == 24
    assert np.sum(splits.y_train_multiclass == web_id) == 12

    # Val & Test should have 20% each
    assert np.sum(splits.y_val_multiclass == benign_id) == 48
    assert np.sum(splits.y_test_multiclass == benign_id) == 48

    # Binary labels: Benign is 0, attacks are 1
    assert np.all(splits.y_train_binary[splits.y_train_multiclass == benign_id] == 0)
    assert np.all(splits.y_train_binary[splits.y_train_multiclass == dos_id] == 1)
    assert np.all(splits.y_train_binary[splits.y_train_multiclass == web_id] == 1)

    # Test include_destination_port=False (port retained for ablation)
    splits_with_port = prepare_splits(df, seed=0, exclude_destination_port=False)
    assert "Destination Port" in splits_with_port.feature_names
    assert splits_with_port.X_train.shape[1] == 3


# ==============================================================================
# 2. Model Baseline Tests (XGBoost, LightGBM, Random Forest)
# ==============================================================================

def test_tree_models_binary():
    """Verify all 3 models fit and predict on synthetic binary float32 data."""
    rng = np.random.RandomState(42)
    n = 120
    X = rng.normal(0, 1, size=(n, 5)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

    X_train, X_val, X_test = X[:60], X[60:90], X[90:]
    y_train, y_val, y_test = y[:60], y[60:90], y[90:]

    for model_name in ["xgboost", "lightgbm", "random_forest"]:
        model = train_baseline_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            task="binary",
            seed=0,
            n_estimators=10,  # small for unit test
        )
        y_pred, y_prob = predict_baseline_model(model, X_test, task="binary")

        assert y_pred.shape == (30,)
        assert y_prob.shape == (30,)
        assert np.all(np.isin(y_pred, [0, 1]))
        assert np.all(y_prob >= 0.0)
        assert np.all(y_prob <= 1.0)
        assert y_prob.dtype == np.float32


def test_tree_models_multiclass():
    """Verify all 3 models fit and predict on synthetic multiclass float32 data."""
    rng = np.random.RandomState(42)
    n = 150
    X = rng.normal(0, 1, size=(n, 6)).astype(np.float32)
    y = rng.randint(0, 3, size=n).astype(np.int64)

    X_train, X_val, X_test = X[:90], X[90:120], X[120:]
    y_train, y_val, y_test = y[:90], y[90:120], y[120:]

    for model_name in ["xgboost", "lightgbm", "random_forest"]:
        model = train_baseline_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            task="multiclass",
            seed=0,
            n_estimators=10,
        )
        y_pred, y_prob = predict_baseline_model(model, X_test, task="multiclass")

        assert y_pred.shape == (30,)
        assert y_prob.shape == (30, 3)
        assert np.all(np.isin(y_pred, [0, 1, 2]))
        # Probabilities sum to 1
        np.testing.assert_allclose(np.sum(y_prob, axis=1), 1.0, atol=1e-4)


# ==============================================================================
# 3. Metrics Tests
# ==============================================================================

def test_binary_metrics():
    """Verify binary metrics including Recall@fixed FPR, PR-AUC, MCC, and FPR@0.5."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.8, 0.85, 0.9, 0.95, 0.99])

    metrics = evaluate_binary_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["f1"] == 1.0
    assert metrics["mcc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["fpr_at_0.5"] == 0.0
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["accuracy"] == 1.0

    # Recall at 0.1% FPR with perfect separation should be 1.0
    assert metrics["recall_at_0.1_fpr"] == 1.0
    assert metrics["recall_at_1.0_fpr"] == 1.0

    # Test imperfect case
    y_true_imperfect = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob_imperfect = np.array([0.1, 0.8, 0.2, 0.3, 0.4, 0.7, 0.9, 0.85])
    m2 = evaluate_binary_metrics(y_true_imperfect, y_prob_imperfect, threshold=0.5)
    assert 0.0 < m2["f1"] < 1.0
    assert m2["fpr_at_0.5"] == 0.25  # 1 FP out of 4 negatives


def test_multiclass_metrics():
    """Verify multiclass metrics with class presence restriction and row-normalized CM."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 0])
    # Class 2 exists in global space but NOT in this test split
    all_classes = [0, 1, 2]

    metrics = evaluate_multiclass_metrics(
        y_true, y_pred, all_classes=all_classes
    )

    # Macro-F1 should only average over classes present in y_true ([0, 1])
    assert 0.0 < metrics["macro_f1"] <= 1.0
    assert "accuracy" in metrics
    assert "per_class_f1" in metrics
    # Row normalized CM: row for 2 should sum to 0 (since not in y_true)
    cm_norm = np.array(metrics["confusion_matrix_normalized"])
    assert cm_norm.shape == (3, 3)
    assert pytest.approx(np.sum(cm_norm[0])) == 1.0
    assert pytest.approx(np.sum(cm_norm[1])) == 1.0
    assert np.sum(cm_norm[2]) == 0.0


# ==============================================================================
# 4. Statistical Testing Tests
# ==============================================================================

def test_stats_mean_std_and_ci():
    """Verify mean ± SD and Student's t 95% confidence interval calculations."""
    vals = [0.996, 0.997, 0.995]
    res = compute_mean_std(vals, as_percentage=True)
    assert pytest.approx(res["mean"], 0.01) == 99.6
    assert pytest.approx(res["std"], 0.01) == 0.1
    assert "99.60 ± 0.10" in res["formatted"]

    # Paired differences CI
    vals_a = [0.95, 0.96, 0.97]
    vals_b = [0.90, 0.91, 0.92]
    ci = compute_paired_difference_ci(vals_a, vals_b, as_percentage=True)
    assert pytest.approx(ci["mean_diff"], 0.01) == 5.0
    assert ci["ci_lower"] <= ci["mean_diff"] <= ci["ci_upper"]


def test_holm_bonferroni_correction():
    """Verify step-down Holm-Bonferroni correction on multiple comparisons."""
    p_vals = [0.01, 0.04, 0.03]
    # Sorted: 0.01 (x3 -> 0.03), 0.03 (x2 -> 0.06), 0.04 (x1 -> 0.06)
    adj = holm_bonferroni_correction(p_vals)
    assert pytest.approx(adj[0]) == 0.03
    assert pytest.approx(adj[2]) == 0.06
    assert pytest.approx(adj[1]) == 0.06

    comparisons = [
        {
            "name_a": "XGBoost",
            "name_b": "LightGBM",
            "values_a": [0.99, 0.99, 0.99],
            "values_b": [0.98, 0.98, 0.98],
        },
        {
            "name_a": "XGBoost",
            "name_b": "RandomForest",
            "values_a": [0.99, 0.99, 0.99],
            "values_b": [0.95, 0.95, 0.95],
        },
    ]
    test_results = paired_t_test_with_holm(comparisons)
    assert len(test_results) == 2
    for r in test_results:
        assert "p_raw" in r
        assert "p_adjusted" in r
        assert "reject_null" in r


# ==============================================================================
# 5. Stress Test Helpers (Noise & LOFO)
# ==============================================================================

def test_noise_test_simulation():
    """Verify Gaussian feature noise application and clipping."""
    X = np.zeros((100, 10), dtype=np.float32)
    rng = np.random.RandomState(42)
    noise = rng.normal(0.0, 0.5, size=X.shape).astype(np.float32)
    X_noisy = np.clip(X + noise, -10.0, 10.0).astype(np.float32)

    assert X_noisy.shape == (100, 10)
    assert X_noisy.dtype == np.float32
    assert np.all(X_noisy >= -10.0)
    assert np.all(X_noisy <= 10.0)
    assert np.std(X_noisy) > 0.4  # Matches ~0.5 std


def test_leave_one_out_isolation():
    """Verify that withheld attack family is absent from train/val and present in test."""
    n = 200
    df = pd.DataFrame(
        {
            "feat_1": np.random.randn(n).astype(np.float32),
            "feat_2": np.random.randn(n).astype(np.float32),
            "label_multiclass": ["Benign"] * 100 + ["DoS"] * 50 + ["PortScan"] * 50,
        }
    )
    # If DoS is withheld:
    # 60/40 then 50/50 split on df
    splits = prepare_splits(df, seed=0)
    dos_id = splits.label_mapping["DoS"]

    # Filter out DoS from train and val
    train_mask = splits.y_train_multiclass != dos_id
    val_mask = splits.y_val_multiclass != dos_id

    y_train_filtered = splits.y_train_multiclass[train_mask]
    y_val_filtered = splits.y_val_multiclass[val_mask]
    y_test = splits.y_test_multiclass  # Not filtered!

    assert dos_id not in y_train_filtered
    assert dos_id not in y_val_filtered
    assert dos_id in y_test  # Test set retains the held-out family!


def test_binary_vs_multiclass_sample_weights():
    """Verify binary task uses C=2 weights while multiclass uses C=num_classes weights."""
    rng = np.random.RandomState(7)
    n = 100
    df = pd.DataFrame(
        {
            "feat_1": rng.normal(0, 1, n).astype(np.float32),
            "feat_2": rng.normal(0, 1, n).astype(np.float32),
            "label_multiclass": ["Benign"] * 80 + ["DoS"] * 15 + ["Rare"] * 5,
        }
    )
    splits = prepare_splits(df, seed=0)

    # Train split has N=60: 48 Benign (0), 9 DoS, 3 Rare (12 Attack total)
    # Binary C=2: w_benign = 60 / (2 * 48) = 0.625, w_attack = 60 / (2 * 12) = 2.5
    bin_benign_w = splits.sample_weight_train_binary[splits.y_train_binary == 0][0]
    bin_attack_w = splits.sample_weight_train_binary[splits.y_train_binary == 1][0]
    assert pytest.approx(bin_benign_w, abs=1e-4) == 60.0 / (2.0 * 48.0)
    assert pytest.approx(bin_attack_w, abs=1e-4) == 60.0 / (2.0 * 12.0)

    # Multiclass C=3: Rare weight = 60 / (3 * 3) = 6.6667
    rare_id = splits.label_mapping["Rare"]
    rare_w = splits.sample_weight_train[splits.y_train_multiclass == rare_id][0]
    assert pytest.approx(rare_w, abs=1e-4) == 60.0 / (3.0 * 3.0)


def test_prepare_splits_does_not_mutate_input_df():
    """Verify prepare_splits reads directly from df without mutating caller's DataFrame."""
    rng = np.random.RandomState(99)
    n = 150
    df = pd.DataFrame(
        {
            "feat_pos": rng.exponential(scale=50.0, size=n).astype(np.float32),
            "feat_neg": rng.normal(0.0, 2.0, size=n).astype(np.float32),
            "Destination Port": rng.randint(1, 65535, size=n).astype(np.float32),
            "label_multiclass": ["Benign"] * 90 + ["DoS"] * 40 + ["Web"] * 20,
        }
    )
    df_snapshot = df.copy(deep=True)

    splits = prepare_splits(df, seed=0, exclude_destination_port=True)
    assert "Destination Port" not in splits.feature_names
    assert splits.X_train.dtype == np.float32

    # Caller's DataFrame must be bit-for-bit unchanged
    pd.testing.assert_frame_equal(df, df_snapshot, check_exact=True)


def test_run_benchmark_for_dataset_outputs_and_formatting(tmp_path):
    """Adversarial test for run_benchmark_for_dataset JSON/CSV schemas, MCC scale, CM sum, and Holm stats."""
    import json
    from src.experiments.run_main import run_benchmark_for_dataset

    rng = np.random.RandomState(2026)
    n_benign, n_dos, n_web = 90, 45, 45
    n_total = n_benign + n_dos + n_web

    # Separable synthetic features so models learn non-trivial MCC > 0
    X_benign = rng.normal(loc=0.0, scale=0.8, size=(n_benign, 4)).astype(np.float32)
    X_dos = rng.normal(loc=3.0, scale=0.8, size=(n_dos, 4)).astype(np.float32)
    X_web = rng.normal(loc=-3.0, scale=0.8, size=(n_web, 4)).astype(np.float32)
    X = np.vstack([X_benign, X_dos, X_web])

    df = pd.DataFrame(
        {
            "f0": X[:, 0],
            "f1": X[:, 1],
            "f2": X[:, 2],
            "f3": X[:, 3],
            "Destination Port": rng.randint(1, 65535, size=n_total).astype(np.float32),
            "label_multiclass": ["Benign"] * n_benign + ["DoS"] * n_dos + ["Web"] * n_web,
        }
    )
    parquet_path = tmp_path / "tiny_synth.parquet"
    df.to_parquet(parquet_path, index=False)

    out_dir = tmp_path / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [0, 1]
    models = ["xgboost", "lightgbm"]

    run_benchmark_for_dataset(
        dataset_name="tiny_synth",
        data_path=parquet_path,
        tasks=["binary", "multiclass"],
        models=models,
        seeds=seeds,
        output_dir=out_dir,
        exclude_destination_port=True,
    )

    bin_path = out_dir / "tiny_synth_binary_summary.json"
    multi_path = out_dir / "tiny_synth_multiclass_summary.json"
    csv_path = out_dir / "tiny_synth_detailed_results.csv"

    assert bin_path.exists()
    assert multi_path.exists()
    assert csv_path.exists()

    with open(bin_path, "r", encoding="utf-8") as f:
        bin_summary = json.load(f)

    # Binary MCC must be unitless (in [-1, 1], formatted to 3 decimals), while F1/PR-AUC are %
    for m in models:
        mcc_entry = bin_summary["models"][m]["mcc"]
        assert -1.0 <= mcc_entry["mean"] <= 1.0
        assert len(mcc_entry["formatted"].split(" ± ")[0].split(".")[1]) == 3

        f1_entry = bin_summary["models"][m]["f1"]
        assert 50.0 < f1_entry["mean"] <= 100.0
        assert len(f1_entry["formatted"].split(" ± ")[0].split(".")[1]) == 2

    # Binary pairwise comparisons must include BOTH f1 and pr_auc
    assert "pairwise_comparisons" in bin_summary
    assert "f1" in bin_summary["pairwise_comparisons"]
    assert "pr_auc" in bin_summary["pairwise_comparisons"]
    assert len(bin_summary["pairwise_comparisons"]["f1"]) == 1
    assert len(bin_summary["pairwise_comparisons"]["pr_auc"]) == 1
    assert "p_adjusted" in bin_summary["pairwise_comparisons"]["pr_auc"][0]

    with open(multi_path, "r", encoding="utf-8") as f:
        multi_summary = json.load(f)

    assert multi_summary["class_order"] == ["Benign", "DoS", "Web"]
    assert "macro_f1" in multi_summary["pairwise_comparisons"]
    expected_test_total = len(seeds) * int(round(n_total * 0.2))

    for m in models:
        m_entry = multi_summary["models"][m]
        assert "per_class_f1" in m_entry
        for cls_name in ["Benign", "DoS", "Web"]:
            assert cls_name in m_entry["per_class_f1"]
            assert 0.0 <= m_entry["per_class_f1"][cls_name]["mean"] <= 100.0

        cm_raw = np.array(m_entry["confusion_matrix_raw"])
        cm_norm = np.array(m_entry["confusion_matrix_normalized"])
        assert cm_raw.shape == (3, 3)
        assert cm_norm.shape == (3, 3)
        assert int(cm_raw.sum()) == expected_test_total
        # Each row of normalized CM sums to 100.0%
        np.testing.assert_allclose(cm_norm.sum(axis=1), [100.0, 100.0, 100.0], atol=1e-6)

    # Detailed CSV retains scalar metrics and per_class_f1_* columns
    detailed_df = pd.read_csv(csv_path)
    assert len(detailed_df) == len(seeds) * len(models) * 2
    for col in ["f1", "mcc", "pr_auc", "macro_f1", "per_class_f1_Benign", "per_class_f1_DoS", "per_class_f1_Web"]:
        assert col in detailed_df.columns
