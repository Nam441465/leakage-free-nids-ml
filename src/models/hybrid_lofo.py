"""Hybrid Leave-One-Family-Out (LOFO) Benign-only anomaly detector and calibration.

Strictly leakage-free protocol:
- BenignAnomalyScorer fits ONLY on Benign training flows (y_train == 0).
- Hybrid thresholds are calibrated ONLY on validation flows (with withheld attack
  family excluded), keeping validation Benign FPR tightly bounded.
- Feature computations run in memory-bounded batches in float32.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


class BenignAnomalyScorer:
    """Unsupervised Benign-only tabular anomaly scorer for OOD intrusion detection.

    Combines PCA reconstruction error and standardized excess beyond the Benign
    [P1, P99] and |z| > 2.5 boundaries, fitted strictly on Benign training flows.
    """

    def __init__(self, std_floor: float = 0.05, batch_size: int = 65_536) -> None:
        self.std_floor = float(std_floor)
        self.batch_size = int(batch_size)

        self.benign_mean_: np.ndarray | None = None
        self.benign_std_: np.ndarray | None = None
        self.benign_p01_: np.ndarray | None = None
        self.benign_p99_: np.ndarray | None = None
        self.pca_: PCA | None = None
        self.pca_mean_: np.ndarray | None = None
        self.pca_components_: np.ndarray | None = None
        self.quantile_grid_: np.ndarray | None = None
        self.benign_score_quantiles_: np.ndarray | None = None
        self.p999_: float = 0.0
        self.tail_scale_: float = 1.0
        self.n_features_: int = 0
        self.n_fit_samples_: int = 0
        self.is_fitted_: bool = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray | None = None,
        seed: int = 0,
        max_fit_samples: int = 150_000,
        n_components: int = 16,
    ) -> BenignAnomalyScorer:
        """Fit Benign distribution statistics and PCA strictly on Benign training flows."""
        X_arr = np.asarray(X_train, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D X_train, got shape {X_arr.shape}.")

        if y_train is not None:
            y_arr = np.asarray(y_train)
            if len(y_arr) != len(X_arr):
                raise ValueError(
                    f"Length mismatch: X_train ({len(X_arr)}) vs y_train ({len(y_arr)})."
                )
            benign_indices = np.flatnonzero(y_arr == 0)
            if len(benign_indices) < 2:
                raise ValueError(
                    f"At least 2 Benign (y == 0) training flows are required, found {len(benign_indices)}."
                )
        else:
            if len(X_arr) < 2:
                raise ValueError(
                    f"At least 2 training flows are required, found {len(X_arr)}."
                )
            benign_indices = np.arange(len(X_arr), dtype=np.int64)

        rng = np.random.default_rng(int(seed))
        if len(benign_indices) > int(max_fit_samples):
            chosen = rng.choice(
                benign_indices, size=int(max_fit_samples), replace=False
            )
            chosen.sort()
            X_benign = np.asarray(X_arr[chosen], dtype=np.float32)
        else:
            X_benign = np.asarray(X_arr[benign_indices], dtype=np.float32)

        n_fit, n_features = X_benign.shape
        self.n_features_ = int(n_features)
        self.n_fit_samples_ = int(n_fit)

        # Compute Benign per-feature location, scale (floored at 0.05), and [P1, P99]
        X_benign_f64 = np.nan_to_num(
            X_benign.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )
        mean = np.mean(X_benign_f64, axis=0)
        std = np.std(X_benign_f64, axis=0)
        std = np.maximum(std, self.std_floor)
        p01 = np.percentile(X_benign_f64, 1.0, axis=0)
        p99 = np.percentile(X_benign_f64, 99.0, axis=0)

        self.benign_mean_ = mean.astype(np.float32)
        self.benign_std_ = std.astype(np.float32)
        self.benign_p01_ = p01.astype(np.float32)
        self.benign_p99_ = p99.astype(np.float32)

        # Standardize Benign flows using Benign mean/std and fit PCA
        Z_benign = ((X_benign_f64 - mean) / std).astype(np.float32)
        max_k = min(int(n_components), n_features - 1, n_fit - 1)
        if max_k >= 1:
            pca = PCA(n_components=max_k, random_state=int(seed))
            pca.fit(Z_benign)
            self.pca_ = pca
            self.pca_mean_ = pca.mean_.astype(np.float32)
            self.pca_components_ = pca.components_.astype(np.float32)
        else:
            self.pca_ = None
            self.pca_mean_ = None
            self.pca_components_ = None

        self.is_fitted_ = True

        # Empirical quantile distribution of raw anomaly scores on Benign training sample
        raw_benign = self.score_raw(X_benign)
        self.quantile_grid_ = np.linspace(0.0, 1.0, 1001, dtype=np.float64)
        self.benign_score_quantiles_ = np.quantile(
            raw_benign.astype(np.float64), self.quantile_grid_
        )
        self.p999_ = float(np.quantile(raw_benign.astype(np.float64), 0.999))
        p990 = float(np.quantile(raw_benign.astype(np.float64), 0.990))
        self.tail_scale_ = max(self.p999_ - p990, 1e-6)

        return self

    def score_raw(self, X: np.ndarray) -> np.ndarray:
        """Compute non-negative float32 raw anomaly score for each row in X."""
        if not self.is_fitted_ or self.benign_mean_ is None or self.benign_std_ is None:
            raise RuntimeError("BenignAnomalyScorer must be fitted before scoring.")

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected 2D array with {self.n_features_} features, got shape {X_arr.shape}."
            )

        n_samples = len(X_arr)
        if n_samples == 0:
            return np.empty(0, dtype=np.float32)

        scores = np.empty(n_samples, dtype=np.float32)
        mean = self.benign_mean_
        std = self.benign_std_
        p01 = self.benign_p01_
        p99 = self.benign_p99_
        pca_mean = self.pca_mean_
        pca_comp = self.pca_components_

        step = max(self.batch_size, 1)
        for start in range(0, n_samples, step):
            end = min(start + step, n_samples)
            xb = np.nan_to_num(
                X_arr[start:end], nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32, copy=False)

            zb = (xb - mean) / std

            # 1. PCA reconstruction MSE in Benign-standardized space
            if pca_comp is not None and pca_mean is not None:
                zc = zb - pca_mean
                z_proj = zc @ pca_comp.T
                z_recon = z_proj @ pca_comp
                diff = zc - z_recon
                pca_mse = np.mean(diff * diff, axis=1, dtype=np.float64)
            else:
                pca_mse = np.zeros(end - start, dtype=np.float64)

            # 2. Standardized excess beyond |z_benign| > 2.5
            z_excess = np.maximum(np.float32(0.0), np.abs(zb) - np.float32(2.5))
            z_tail_mse = np.mean(z_excess * z_excess, axis=1, dtype=np.float64)

            # 3. Standardized excess outside Benign [P1, P99] envelope
            below = np.maximum(np.float32(0.0), p01 - xb) / std
            above = np.maximum(np.float32(0.0), xb - p99) / std
            env_excess = below + above
            env_mse = np.mean(env_excess * env_excess, axis=1, dtype=np.float64)

            total = pca_mse + z_tail_mse + env_mse
            total = np.maximum(
                np.nan_to_num(total, nan=0.0, posinf=1e6, neginf=0.0), 0.0
            )
            scores[start:end] = total.astype(np.float32)

        return scores

    def score_proba(self, X: np.ndarray) -> np.ndarray:
        """Map raw anomaly scores to [0.0, 1.0] outlier probabilities using Benign quantiles."""
        if (
            not self.is_fitted_
            or self.benign_score_quantiles_ is None
            or self.quantile_grid_ is None
        ):
            raise RuntimeError("BenignAnomalyScorer must be fitted before scoring.")

        raw = self.score_raw(X).astype(np.float64)
        if len(raw) == 0:
            return np.empty(0, dtype=np.float32)

        # Empirical CDF interpolation up to the 99.9th Benign percentile
        probs = np.interp(
            raw,
            self.benign_score_quantiles_,
            self.quantile_grid_,
            left=0.0,
            right=0.999,
        )
        probs = np.minimum(probs, 0.999)

        # Smooth monotonic tail extrapolation in (0.999, 1.0) above p999_
        tail_mask = raw > self.p999_
        if np.any(tail_mask):
            excess = (raw[tail_mask] - self.p999_) / self.tail_scale_
            probs[tail_mask] = 0.999 + 0.001 * (1.0 - np.exp(-np.minimum(excess, 50.0)))

        return np.clip(probs, 0.0, 1.0).astype(np.float32)


def _extract_1d_prob(xgb_prob: np.ndarray) -> np.ndarray:
    arr = np.asarray(xgb_prob, dtype=np.float64)
    if arr.ndim == 2:
        return arr[:, 1] if arr.shape[1] >= 2 else arr[:, 0]
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D or 2D xgb_prob, got shape {arr.shape}.")
    return arr


def calibrate_hybrid_lofo_on_val(
    y_val_bin: np.ndarray,
    xgb_prob_val: np.ndarray,
    anom_raw_val: np.ndarray,
    target_benign_fpr: float = 0.0020,
) -> dict[str, float]:
    """Calibrate hybrid XGBoost + BenignAnomalyScorer thresholds strictly on validation.

    Searches validation Benign quantiles so combined
        (xgb_prob_val >= xgb_thresh) | (anom_raw_val >= anom_thresh)
    keeps validation Benign FPR <= max(baseline_val_fpr + 0.0012, 0.0022) while
    maximizing validation seen-attack recall and OOD boundary coverage.
    """
    y_arr = np.asarray(y_val_bin, dtype=np.int64)
    p_xgb = _extract_1d_prob(xgb_prob_val)
    a_raw = np.asarray(anom_raw_val, dtype=np.float64)

    if not (len(y_arr) == len(p_xgb) == len(a_raw)):
        raise ValueError("y_val_bin, xgb_prob_val, and anom_raw_val must have equal length.")

    xgb_thresh = 0.5
    benign_mask = y_arr == 0
    attack_mask = y_arr == 1

    xgb_pred_base = p_xgb >= xgb_thresh
    val_benign_fpr_baseline = (
        float(np.mean(xgb_pred_base[benign_mask])) if np.any(benign_mask) else 0.0
    )
    val_seen_recall_baseline = (
        float(np.mean(xgb_pred_base[attack_mask])) if np.any(attack_mask) else 0.0
    )

    if not np.any(benign_mask):
        return {
            "xgb_thresh": xgb_thresh,
            "anom_thresh": 1.0,
            "val_benign_fpr_baseline": val_benign_fpr_baseline,
            "val_benign_fpr_hybrid": val_benign_fpr_baseline,
            "val_seen_recall_baseline": val_seen_recall_baseline,
            "val_seen_recall_hybrid": val_seen_recall_baseline,
        }

    benign_anom = a_raw[benign_mask]
    benign_xgb = p_xgb[benign_mask]
    target_q = float(np.clip(target_benign_fpr, 1e-6, 0.5))
    fpr_cap = max(val_benign_fpr_baseline + 0.0012, 0.0022)
    if target_q > 0.0022:
        fpr_cap = max(fpr_cap, target_q)

    # Calibrate xgb_thresh strictly on validation Benign tail quantile (allocating 85%
    # of the target Benign FPR budget to soft-margin GBDT sub-threshold calibration
    # and the remaining budget to BenignAnomalyScorer OOD tail detection).
    xgb_q_budget = min(max(0.85 * target_q, val_benign_fpr_baseline), fpr_cap * 0.90)
    calibrated_xgb_t = float(np.quantile(benign_xgb, 1.0 - xgb_q_budget))
    xgb_thresh = float(np.clip(min(0.5, calibrated_xgb_t), 1e-4, 0.5))
    xgb_pred_cal = p_xgb >= xgb_thresh
    val_fpr_at_xgb_cal = float(np.mean(xgb_pred_cal[benign_mask]))
    if val_fpr_at_xgb_cal > fpr_cap:
        xgb_thresh = 0.5
        xgb_pred_cal = xgb_pred_base

    # Candidate anomaly FPR budgets from very tight up to target_benign_fpr
    q_candidates = np.unique(
        np.concatenate(
            [
                np.geomspace(1e-5, max(target_q, 1e-4), 40),
                np.linspace(0.0, target_q, 41),
                np.array([target_q], dtype=np.float64),
            ]
        )
    )
    q_candidates = q_candidates[(q_candidates > 0.0) & (q_candidates <= target_q + 1e-12)]
    cand_thresholds = np.unique(np.quantile(benign_anom, 1.0 - q_candidates))
    # Sort descending (tightest threshold first) and also include strict upper guard
    strict_guard = float(np.max(benign_anom) * 1.05 + 1e-6)
    cand_thresholds = np.concatenate(
        [np.array([strict_guard], dtype=np.float64), cand_thresholds[::-1]]
    )

    best_thresh = strict_guard
    best_fpr = float(np.mean(xgb_pred_cal[benign_mask]))
    best_recall = (
        float(np.mean(xgb_pred_cal[attack_mask])) if np.any(attack_mask) else 0.0
    )
    found_feasible = False

    for thresh in cand_thresholds:
        thresh_val = max(float(thresh), 1e-6)
        hybrid_pred = xgb_pred_cal | (a_raw >= thresh_val)
        h_fpr = float(np.mean(hybrid_pred[benign_mask]))
        h_recall = (
            float(np.mean(hybrid_pred[attack_mask])) if np.any(attack_mask) else 0.0
        )

        if h_fpr <= fpr_cap + 1e-12:
            # Since cand_thresholds is sorted from tightest to loosest, taking any
            # feasible candidate with >= best_recall maximizes both seen recall and
            # OOD boundary coverage (smallest anom_thresh within the FPR cap).
            if (not found_feasible) or (h_recall >= best_recall - 1e-12):
                best_thresh = thresh_val
                best_fpr = h_fpr
                best_recall = h_recall
                found_feasible = True

    return {
        "xgb_thresh": float(xgb_thresh),
        "anom_thresh": float(best_thresh),
        "val_benign_fpr_baseline": float(val_benign_fpr_baseline),
        "val_benign_fpr_hybrid": float(best_fpr),
        "val_seen_recall_baseline": float(val_seen_recall_baseline),
        "val_seen_recall_hybrid": float(best_recall),
    }


def predict_hybrid_lofo(
    xgb_prob: np.ndarray,
    anom_raw: np.ndarray,
    calib_params: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Predict binary intrusion labels and continuous combined score for hybrid LOFO."""
    p_xgb = _extract_1d_prob(xgb_prob)
    a_raw = np.asarray(anom_raw, dtype=np.float64)
    if len(p_xgb) != len(a_raw):
        raise ValueError(
            f"Length mismatch: xgb_prob ({len(p_xgb)}) vs anom_raw ({len(a_raw)})."
        )

    xgb_thresh = float(calib_params["xgb_thresh"])
    anom_thresh = float(calib_params["anom_thresh"])

    y_pred = ((p_xgb >= xgb_thresh) | (a_raw >= anom_thresh)).astype(np.int64)
    below_half = float(np.nextafter(np.float32(0.5), np.float32(0.0)))

    xgb_scaled = np.where(
        p_xgb >= xgb_thresh,
        np.clip(
            0.5 + 0.5 * ((p_xgb - xgb_thresh) / max(1.0 - xgb_thresh, 1e-12)),
            0.5,
            1.0,
        ),
        np.clip(
            0.5 * (p_xgb / max(xgb_thresh, 1e-12)),
            0.0,
            below_half,
        ),
    )
    anom_raw_scaled = 0.5 * (a_raw / max(anom_thresh, 1e-6))
    anom_scaled = np.where(
        a_raw >= anom_thresh,
        np.clip(anom_raw_scaled, 0.5, 0.999),
        np.clip(anom_raw_scaled, 0.0, below_half),
    )
    combined_score = np.maximum(xgb_scaled, anom_scaled).astype(np.float32)
    return y_pred, combined_score
