"""Unit and adversarial anti-leakage tests for Phase 4 core modules.

Covers:
- src.data.feature_engineering (append_behavioral_ratios)
- src.models.calibration (compute_tempered_weights, tune_binary_threshold_on_val,
  predict_binary_with_threshold, tune_multiclass_priors_on_val,
  apply_multiclass_priors, build_noise_augmented_training_data)
- src.models.hybrid_lofo (BenignAnomalyScorer, calibrate_hybrid_lofo_on_val,
  predict_hybrid_lofo)
"""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.data.feature_engineering import (
    CICIDS2017_RATIO_SPECS,
    NF_UNSW_RATIO_SPECS,
    append_behavioral_ratios,
)
from src.data.split_preprocess import compute_class_weights
from src.models.calibration import (
    apply_multiclass_priors,
    build_noise_augmented_training_data,
    compute_tempered_weights,
    predict_binary_with_threshold,
    tune_binary_threshold_on_val,
    tune_multiclass_priors_on_val,
)
from src.models.hybrid_lofo import (
    BenignAnomalyScorer,
    calibrate_hybrid_lofo_on_val,
    predict_hybrid_lofo,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================================
# 1. Feature Engineering Tests
# =============================================================================


class TestFeatureEngineering:
    def test_nf_unsw_all_12_ratios_from_real_schema(self):
        meta_path = PROJECT_ROOT / "data" / "processed" / "nf_unsw_nb15_v2_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feature_names = [c for c in meta["feature_columns"] if c != "L4_DST_PORT"]

        rng = np.random.default_rng(42)
        X = rng.uniform(0.0, 500.0, size=(32, len(feature_names))).astype(np.float32)

        for ds_alias in ("nf_unsw", "nf_unsw_nb15_v2"):
            X_aug, names_aug = append_behavioral_ratios(X, feature_names, ds_alias)
            assert X_aug.dtype == np.float32
            assert X_aug.shape == (32, len(feature_names) + 12)
            assert len(names_aug) == len(feature_names) + 12
            expected_new = [spec[0] for spec in NF_UNSW_RATIO_SPECS]
            assert names_aug[len(feature_names) :] == expected_new
            assert np.all(np.isfinite(X_aug))
            assert np.all(X_aug[:, len(feature_names) :] >= 0.0)

        # DataFrame input produces bit-for-bit identical output
        df = pd.DataFrame(X, columns=feature_names)
        X_df_aug, names_df = append_behavioral_ratios(df, feature_names, "nf_unsw")
        assert names_df == names_aug
        assert np.array_equal(X_df_aug, X_aug)

    def test_cicids2017_all_12_ratios_from_real_schema(self):
        meta_path = PROJECT_ROOT / "data" / "processed" / "cicids2017.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feature_names = [c for c in meta["feature_columns"] if c != "Destination Port"]

        rng = np.random.default_rng(7)
        X = rng.uniform(-1.0, 1000.0, size=(24, len(feature_names))).astype(np.float32)

        X_aug, names_aug = append_behavioral_ratios(X, feature_names, "cicids2017")
        assert X_aug.dtype == np.float32
        assert X_aug.shape == (24, len(feature_names) + 12)
        expected_new = [spec[0] for spec in CICIDS2017_RATIO_SPECS]
        assert names_aug[len(feature_names) :] == expected_new
        assert np.all(np.isfinite(X_aug))
        assert np.all(X_aug[:, len(feature_names) :] >= 0.0)

    def test_negative_sentinels_and_non_finite_values_safe(self):
        # Negative -1 sentinels in denominators must NOT produce zero division
        cols = [
            "Init_Win_bytes_forward",
            "Init_Win_bytes_backward",
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
        ]
        X = np.array(
            [
                [100.0, -1.0, 50.0, -1.0],
                [-1.0, -1.0, np.nan, np.inf],
                [256.0, 0.0, -50.0, -999.0],
            ],
            dtype=np.float32,
        )
        X_aug, names_aug = append_behavioral_ratios(X, cols, "cicids2017")
        assert X_aug.dtype == np.float32
        assert names_aug == cols + ["RATIO_FWD_BWD_BYTES", "WIN_BYTES_RATIO"]

        ratio_bytes = X_aug[:, 4]
        ratio_win = X_aug[:, 5]
        assert np.all(np.isfinite(ratio_bytes))
        assert np.all(np.isfinite(ratio_win))
        assert np.all(ratio_bytes >= 0.0)
        assert np.all(ratio_win >= 0.0)

        # Row 0: pos(50) / (pos(-1) + 1) = 50 / 1 = 50; pos(100) / (pos(-1) + 1) = 100
        assert np.isclose(ratio_bytes[0], 50.0, rtol=1e-5)
        assert np.isclose(ratio_win[0], 100.0, rtol=1e-5)
        # Row 1: all sanitized to 0
        assert ratio_bytes[1] == 0.0
        assert ratio_win[1] == 0.0

    def test_stateless_anti_leakage_per_row_invariance(self):
        cols = ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS", "MAX_TTL", "MIN_TTL"]
        rng = np.random.default_rng(123)
        X_batch = rng.uniform(0.0, 1e4, size=(50, len(cols))).astype(np.float32)

        X_batch_aug, _ = append_behavioral_ratios(X_batch, cols, "nf_unsw")
        # Computing row 17 in isolation must be bit-for-bit identical to batch row 17
        X_single_aug, _ = append_behavioral_ratios(X_batch[17:18], cols, "nf_unsw")
        assert np.array_equal(X_single_aug[0], X_batch_aug[17])

    def test_missing_columns_gracefully_skipped_and_unknown_dataset_raises(self):
        X = np.ones((5, 3), dtype=np.float32)
        X_aug, names_aug = append_behavioral_ratios(
            X, ["feat_a", "feat_b", "feat_c"], "cicids2017"
        )
        assert X_aug.shape == (5, 3)
        assert X_aug.dtype == np.float32
        assert names_aug == ["feat_a", "feat_b", "feat_c"]

        with pytest.raises(ValueError, match="Unsupported dataset"):
            append_behavioral_ratios(X, ["a", "b", "c"], "unknown_dataset")


# =============================================================================
# 2. Calibration, Tempered Weights & Noise Augmentation Tests
# =============================================================================


class TestCalibration:
    def test_compute_tempered_weights_exact_formula_and_cap(self):
        # 900 samples of class 0, 99 of class 1, 1 of class 2 -> N=1000, C=3
        y_train = np.array([0] * 900 + [1] * 99 + [2] * 1, dtype=np.int64)

        sw, cw = compute_tempered_weights(y_train, alpha=0.5, max_weight=15.0)
        assert sw.dtype == np.float32
        assert sw.shape == (1000,)
        assert set(cw.keys()) == {0, 1, 2}

        expected_0 = min((1000.0 / (3.0 * 900.0)) ** 0.5, 15.0)
        expected_1 = min((1000.0 / (3.0 * 99.0)) ** 0.5, 15.0)
        # (1000 / 3)^0.5 = 18.257 > 15.0 -> capped at 15.0
        expected_2 = 15.0

        assert np.isclose(cw[0], expected_0, rtol=1e-6)
        assert np.isclose(cw[1], expected_1, rtol=1e-6)
        assert np.isclose(cw[2], expected_2, rtol=1e-6)
        assert np.isclose(sw[0], expected_0, rtol=1e-5)
        assert np.isclose(sw[900], expected_1, rtol=1e-5)
        assert np.isclose(sw[-1], expected_2, rtol=1e-5)

        # alpha=1.0, max_weight=100.0 matches paper's compute_class_weights exactly
        _, cw_paper = compute_tempered_weights(y_train, alpha=1.0, max_weight=100.0)
        ref_paper = compute_class_weights(y_train)
        for k in ref_paper:
            assert np.isclose(cw_paper[k], ref_paper[k], rtol=1e-10)

    def test_tune_binary_threshold_improves_over_05_and_matches_sklearn(self):
        rng = np.random.default_rng(11)
        # Simulated over-weighted model: benign clustered around 0.55, attacks around 0.85
        y_val = np.array([0] * 800 + [1] * 200, dtype=np.int64)
        p_benign = rng.normal(loc=0.55, scale=0.06, size=800).clip(0.01, 0.99)
        p_attack = rng.normal(loc=0.85, scale=0.05, size=200).clip(0.01, 0.99)
        y_prob = np.concatenate([p_benign, p_attack]).astype(np.float32)

        res = tune_binary_threshold_on_val(y_val, y_prob)
        required_keys = {
            "best_threshold",
            "val_f1_at_0.5",
            "val_f1_at_best",
            "val_fpr_at_0.5",
            "val_fpr_at_best",
        }
        assert set(res.keys()) == required_keys
        assert res["best_threshold"] > 0.65
        assert res["val_f1_at_best"] > res["val_f1_at_0.5"] + 0.40
        assert res["val_fpr_at_best"] < res["val_fpr_at_0.5"]

        # Exact match against sklearn f1_score
        y_pred_best = predict_binary_with_threshold(y_prob, res["best_threshold"])
        assert y_pred_best.dtype == np.int64
        ref_f1_best = float(f1_score(y_val, y_pred_best, zero_division=0.0))
        assert np.isclose(res["val_f1_at_best"], ref_f1_best, atol=1e-12)

        y_pred_05 = predict_binary_with_threshold(y_prob, 0.5)
        ref_f1_05 = float(f1_score(y_val, y_pred_05, zero_division=0.0))
        assert np.isclose(res["val_f1_at_0.5"], ref_f1_05, atol=1e-12)

        # 2D probability input works identically
        y_prob_2d = np.column_stack([1.0 - y_prob, y_prob])
        res_2d = tune_binary_threshold_on_val(y_val, y_prob_2d)
        assert np.isclose(res_2d["best_threshold"], res["best_threshold"], atol=1e-12)

    def test_tune_multiclass_priors_improves_macro_f1_respects_weighted_constraint(self):
        rng = np.random.default_rng(29)
        # 4 classes: class 0 (700), class 1 (200), class 2 (80), class 3 (20, rare)
        # Rare class 3 gets systematically suppressed probability (0.28 vs class 0's 0.60)
        n0, n1, n2, n3 = 700, 200, 80, 20
        y_val = np.array([0] * n0 + [1] * n1 + [2] * n2 + [3] * n3, dtype=np.int64)
        N = len(y_val)
        probs = np.zeros((N, 4), dtype=np.float64)

        probs[:n0] = np.array([0.85, 0.08, 0.05, 0.02]) + rng.uniform(0, 0.03, (n0, 4))
        probs[n0 : n0 + n1] = np.array([0.08, 0.82, 0.06, 0.04]) + rng.uniform(0, 0.03, (n1, 4))
        probs[n0 + n1 : n0 + n1 + n2] = np.array([0.10, 0.08, 0.78, 0.04]) + rng.uniform(0, 0.03, (n2, 4))
        # Class 3 has a distinct signal (p3 ~ 0.30 vs benign p3 ~ 0.02), but raw argmax picks class 0!
        probs[n0 + n1 + n2 :] = np.array([0.58, 0.06, 0.06, 0.30]) + rng.uniform(0, 0.02, (n3, 4))
        probs /= probs.sum(axis=1, keepdims=True)

        res = tune_multiclass_priors_on_val(y_val, probs, max_iter=3)
        assert res["gamma"].shape == (4,)
        assert res["val_macro_f1_after"] > res["val_macro_f1_before"] + 0.15
        assert res["val_weighted_f1_after"] >= res["val_weighted_f1_before"] - 0.0015 - 1e-12

        # Verify apply_multiclass_priors exactness and normalization
        y_pred, prob_cal = apply_multiclass_priors(probs, res["gamma"])
        assert y_pred.dtype == np.int64
        assert prob_cal.dtype == np.float32
        assert np.allclose(prob_cal.sum(axis=1), 1.0, atol=1e-5)
        assert np.array_equal(y_pred, np.argmax(prob_cal, axis=1))

        ref_macro = float(f1_score(y_val, y_pred, average="macro", zero_division=0.0))
        ref_weighted = float(f1_score(y_val, y_pred, average="weighted", zero_division=0.0))
        assert np.isclose(res["val_macro_f1_after"], ref_macro, atol=1e-10)
        assert np.isclose(res["val_weighted_f1_after"], ref_weighted, atol=1e-10)

        # Edge case: all-zero probability row normalizes to uniform without NaN
        zero_prob = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        yp_z, pc_z = apply_multiclass_priors(zero_prob, res["gamma"])
        assert np.allclose(pc_z, 0.25)
        assert yp_z[0] == 0

    def test_tune_multiclass_priors_matches_bruteforce_single_pass(self):
        """Adversarial equivalence test: vectorized bucket sweep == brute-force sklearn."""
        rng = np.random.default_rng(77)
        y_val = rng.integers(0, 4, size=150, dtype=np.int64)
        logits = rng.normal(0.0, 1.0, size=(150, 4))
        logits[np.arange(150), y_val] += 0.9
        exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_l / exp_l.sum(axis=1, keepdims=True)

        fast_res = tune_multiclass_priors_on_val(y_val, probs, max_iter=1)

        # Brute-force reference for 1 pass
        grid = np.unique(np.concatenate([np.geomspace(0.05, 20.0, 61), [1.0]]))
        gamma_bf = np.ones(4, dtype=np.float64)
        yp0, _ = apply_multiclass_priors(probs, gamma_bf)
        best_macro = float(f1_score(y_val, yp0, average="macro", zero_division=0.0))
        min_w = float(f1_score(y_val, yp0, average="weighted", zero_division=0.0)) - 0.0015

        for c in range(4):
            cand_best_m = best_macro
            cand_best_w = -1.0
            cand_best_g = gamma_bf[c]
            for g in grid:
                gt = gamma_bf.copy()
                gt[c] = g
                yp, _ = apply_multiclass_priors(probs, gt)
                mf = float(f1_score(y_val, yp, average="macro", zero_division=0.0))
                wf = float(f1_score(y_val, yp, average="weighted", zero_division=0.0))
                if wf >= min_w - 1e-12:
                    better_macro = mf > cand_best_m + 1e-12
                    tied_macro = abs(mf - cand_best_m) <= 1e-12 and mf > best_macro + 1e-12
                    better_tie = tied_macro and (
                        wf > cand_best_w + 1e-12
                        or (
                            abs(wf - cand_best_w) <= 1e-12
                            and abs(np.log(g)) < abs(np.log(cand_best_g))
                        )
                    )
                    if better_macro or better_tie:
                        cand_best_m = mf
                        cand_best_w = wf
                        cand_best_g = float(g)
            if cand_best_m > best_macro + 1e-12:
                gamma_bf[c] = cand_best_g
                best_macro = cand_best_m

        assert np.isclose(fast_res["val_macro_f1_after"], best_macro, atol=1e-10)
        assert np.allclose(fast_res["gamma"], gamma_bf, rtol=1e-10)

    def test_build_noise_augmented_training_data_structure_clipping_and_mask(self):
        rng = np.random.default_rng(5)
        n_clean, n_feat = 1000, 12
        X_train = rng.normal(0.0, 2.0, size=(n_clean, n_feat)).clip(-10.0, 10.0).astype(np.float32)
        # Put extreme values near clipping bounds to test clipping
        X_train[:50] = 9.95
        y_train = np.array([0] * 800 + [1] * 200, dtype=np.int64)
        w_train = np.where(y_train == 1, 2.5, 0.625).astype(np.float32)
        cont_mask = np.array([True] * 6 + [False] * 6, dtype=bool)

        X_aug, y_aug, w_aug = build_noise_augmented_training_data(
            X_train,
            y_train,
            w_train,
            seed=42,
            aug_ratio=0.5,
            sigmas=(0.1, 0.25, 0.5),
            continuous_mask=cont_mask,
        )

        assert X_aug.dtype == np.float32
        assert w_aug.dtype == np.float32
        assert X_aug.shape == (1500, n_feat)
        assert y_aug.shape == (1500,)
        assert w_aug.shape == (1500,)

        # Clean portion is 100% preserved
        assert np.array_equal(X_aug[:n_clean], X_train)
        assert np.array_equal(y_aug[:n_clean], y_train)
        assert np.array_equal(w_aug[:n_clean], w_train)

        # Clipped strictly to [-10, 10]
        assert float(np.max(X_aug)) <= 10.0
        assert float(np.min(X_aug)) >= -10.0

        # Stratified class ratio preserved in augmented tail (80% class 0, 20% class 1)
        aug_pos_ratio = float(np.mean(y_aug[n_clean:] == 1))
        assert np.isclose(aug_pos_ratio, 0.20, atol=0.01)

        # Determinism check
        X_aug2, y_aug2, w_aug2 = build_noise_augmented_training_data(
            X_train,
            y_train,
            w_train,
            seed=42,
            aug_ratio=0.5,
            sigmas=(0.1, 0.25, 0.5),
            continuous_mask=cont_mask,
        )
        assert np.array_equal(X_aug, X_aug2)
        assert np.array_equal(y_aug, y_aug2)
        assert np.array_equal(w_aug, w_aug2)

        # Continuous vs discrete noise scaling check on zero matrix
        X_zero = np.zeros((1000, 12), dtype=np.float32)
        X_z_aug, _, _ = build_noise_augmented_training_data(
            X_zero,
            y_train,
            w_train,
            seed=99,
            aug_ratio=1.0,
            sigmas=(0.4,),
            continuous_mask=cont_mask,
        )
        std_cont = float(np.std(X_z_aug[1000:, :6]))
        std_disc = float(np.std(X_z_aug[1000:, 6:]))
        assert np.isclose(std_cont, 0.4, atol=0.04)
        assert np.isclose(std_disc, 0.2, atol=0.04)


# =============================================================================
# 3. Hybrid LOFO Benign Anomaly Scorer & Calibration Tests
# =============================================================================


class TestHybridLofo:
    def test_benign_anomaly_scorer_strict_anti_leakage_ignores_attacks(self):
        rng = np.random.default_rng(101)
        X_benign = rng.normal(0.0, 1.0, size=(600, 20)).astype(np.float32)
        # Constant feature to verify std_floor = 0.05
        X_benign[:, 0] = 0.0
        X_attack_clean = rng.normal(5.0, 2.0, size=(200, 20)).astype(np.float32)
        X_attack_corrupt = np.full((200, 20), 1e4, dtype=np.float32)

        y_train = np.array([0] * 600 + [1] * 200, dtype=np.int64)
        X_train_a = np.vstack([X_benign, X_attack_clean])
        X_train_b = np.vstack([X_benign, X_attack_corrupt])

        scorer_a = BenignAnomalyScorer().fit(X_train_a, y_train, seed=0, n_components=8)
        scorer_b = BenignAnomalyScorer().fit(X_train_b, y_train, seed=0, n_components=8)
        scorer_only_benign = BenignAnomalyScorer().fit(X_benign, None, seed=0, n_components=8)

        # Attack rows must have ZERO influence on any fitted statistic or score!
        assert np.array_equal(scorer_a.benign_mean_, scorer_b.benign_mean_)
        assert np.array_equal(scorer_a.benign_std_, scorer_b.benign_std_)
        assert np.array_equal(scorer_a.benign_p01_, scorer_b.benign_p01_)
        assert np.array_equal(scorer_a.benign_p99_, scorer_b.benign_p99_)
        assert np.array_equal(scorer_a.benign_mean_, scorer_only_benign.benign_mean_)

        # Std floor enforced on constant Benign feature
        assert np.isclose(scorer_a.benign_std_[0], 0.05, atol=1e-7)

        X_probe = rng.normal(0.0, 3.0, size=(100, 20)).astype(np.float32)
        raw_a = scorer_a.score_raw(X_probe)
        raw_b = scorer_b.score_raw(X_probe)
        proba_a = scorer_a.score_proba(X_probe)
        proba_b = scorer_b.score_proba(X_probe)

        assert raw_a.dtype == np.float32
        assert proba_a.dtype == np.float32
        assert np.array_equal(raw_a, raw_b)
        assert np.array_equal(proba_a, proba_b)
        assert np.all(raw_a >= 0.0)
        assert np.all((proba_a >= 0.0) & (proba_a <= 1.0))

    def test_synthetic_ood_lofo_detection_and_persistence(self, tmp_path):
        rng = np.random.default_rng(2026)
        n_feat = 24

        # Low-rank correlated Benign manifold + small isotropic noise
        basis = rng.normal(0.0, 0.5, size=(6, n_feat)).astype(np.float32)

        def sample_benign(n: int) -> np.ndarray:
            z = rng.normal(0.0, 1.0, size=(n, 6)).astype(np.float32)
            return (z @ basis + rng.normal(0.0, 0.25, size=(n, n_feat))).astype(np.float32)

        def sample_seen_attack(n: int) -> np.ndarray:
            x = sample_benign(n)
            x[:, :4] += 4.5
            return x

        def sample_unseen_ood_attack(n: int) -> np.ndarray:
            x = sample_benign(n)
            # Orthogonal shift on features 10..15 unseen during training/val
            x[:, 10:16] += 4.0
            return x

        # Train: Benign + seen attack (NO unseen attack)
        X_train = np.vstack([sample_benign(4000), sample_seen_attack(800)])
        y_train = np.array([0] * 4000 + [1] * 800, dtype=np.int64)

        # Validation: Benign + seen attack (NO unseen attack!)
        X_val = np.vstack([sample_benign(3000), sample_seen_attack(600)])
        y_val = np.array([0] * 3000 + [1] * 600, dtype=np.int64)
        # Simulated XGBoost that separates seen attacks well (~0.1% benign FPR)
        xgb_prob_val = np.concatenate(
            [
                rng.uniform(0.0, 0.08, size=2997),
                rng.uniform(0.6, 0.9, size=3),  # 0.1% baseline benign FPR
                rng.uniform(0.85, 0.99, size=600),
            ]
        ).astype(np.float32)

        # Fit strictly on Benign train
        scorer = BenignAnomalyScorer().fit(
            X_train, y_train, seed=0, max_fit_samples=3500, n_components=8
        )
        assert scorer.n_fit_samples_ == 3500

        anom_raw_val = scorer.score_raw(X_val)
        calib = calibrate_hybrid_lofo_on_val(
            y_val, xgb_prob_val, anom_raw_val, target_benign_fpr=0.0020
        )

        expected_keys = {
            "xgb_thresh",
            "anom_thresh",
            "val_benign_fpr_baseline",
            "val_benign_fpr_hybrid",
            "val_seen_recall_baseline",
            "val_seen_recall_hybrid",
        }
        assert set(calib.keys()) == expected_keys
        assert 1e-4 <= calib["xgb_thresh"] <= 0.5
        assert calib["val_benign_fpr_hybrid"] <= max(
            calib["val_benign_fpr_baseline"] + 0.0012, 0.0022
        ) + 1e-12
        assert calib["val_seen_recall_hybrid"] >= calib["val_seen_recall_baseline"]

        # Test set: Benign + seen attack + WITHHELD unseen OOD attack
        X_test_benign = sample_benign(3000)
        X_test_seen = sample_seen_attack(500)
        X_test_ood = sample_unseen_ood_attack(500)
        X_test = np.vstack([X_test_benign, X_test_seen, X_test_ood])

        # Baseline XGBoost completely misses the unseen OOD attack (prob ~ 0.03)
        xgb_prob_test = np.concatenate(
            [
                rng.uniform(0.0, 0.08, size=2997),
                rng.uniform(0.6, 0.9, size=3),
                rng.uniform(0.85, 0.99, size=500),
                rng.uniform(0.01, 0.05, size=500),  # unseen OOD missed by XGB!
            ]
        ).astype(np.float32)

        # Verify joblib round-trip persistence before evaluating test
        bundle_path = tmp_path / "hybrid_detector.joblib"
        joblib.dump({"anomaly_scorer": scorer, "calib_params": calib}, bundle_path, compress=3)
        loaded = joblib.load(bundle_path)

        anom_raw_test = loaded["anomaly_scorer"].score_raw(X_test)
        anom_prob_test = loaded["anomaly_scorer"].score_proba(X_test)
        y_pred, combined_score = predict_hybrid_lofo(
            xgb_prob_test, anom_raw_test, loaded["calib_params"]
        )

        assert y_pred.dtype == np.int64
        assert combined_score.dtype == np.float32
        assert np.array_equal(y_pred == 1, combined_score >= 0.5)

        # Tail monotonicity: extreme OOD anomalies get higher score_proba than Benign median
        assert float(np.median(anom_prob_test[3500:])) > 0.999
        assert float(np.median(anom_prob_test[:3000])) < 0.6

        # Unseen OOD recall jumps from 0% to >95% while test Benign FPR remains < 0.5%
        test_benign_fpr = float(np.mean(y_pred[:3000] == 1))
        test_seen_recall = float(np.mean(y_pred[3000:3500] == 1))
        test_ood_recall = float(np.mean(y_pred[3500:] == 1))

        assert test_benign_fpr < 0.005
        assert test_seen_recall == 1.0
        assert test_ood_recall > 0.95
