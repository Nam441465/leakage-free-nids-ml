"""Leakage-free tempered class weighting, validation calibration, and noise augmentation.

All functions operate strictly within the leakage-free protocol:
- Class weights and noise augmentation use ONLY training data.
- Binary thresholds and multiclass prior multipliers are tuned ONLY on validation data.
- Feature matrices and sample weights remain in float32 for memory efficiency.
"""

from __future__ import annotations

from typing import Any
import numpy as np


def _extract_binary_pos_prob(y_prob: np.ndarray) -> np.ndarray:
    """Convert 1D or 2D binary probability output into a 1D float64 array."""
    arr = np.asarray(y_prob, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape[1] == 2:
            return arr[:, 1]
        if arr.shape[1] == 1:
            return arr[:, 0]
        return arr[:, -1]
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D or 2D binary probabilities, got shape {arr.shape}.")
    return arr


def compute_tempered_weights(
    y_train: np.ndarray,
    alpha: float = 0.5,
    max_weight: float = 25.0,
) -> tuple[np.ndarray, dict[int, float]]:
    """Compute tempered inverse-frequency class weights strictly from y_train.

    Formula:
        w_c^(alpha) = min((N / (C * n_c)) ** alpha, max_weight)

    where N = len(y_train), C = number of unique classes in y_train, and n_c is
    the sample count of class c in y_train.

    Returns:
        (sample_weights_float32, class_weight_dict)
    """
    if alpha < 0.0:
        raise ValueError(f"alpha must be non-negative, got {alpha}.")
    if max_weight <= 0.0:
        raise ValueError(f"max_weight must be positive, got {max_weight}.")

    y_arr = np.asarray(y_train)
    if y_arr.ndim != 1:
        raise ValueError(f"Expected 1D y_train array, got shape {y_arr.shape}.")
    n_samples = len(y_arr)
    if n_samples == 0:
        return np.empty(0, dtype=np.float32), {}

    classes, inverse, counts = np.unique(
        y_arr, return_inverse=True, return_counts=True
    )
    num_classes = len(classes)

    raw_ratios = n_samples / (float(num_classes) * counts.astype(np.float64))
    tempered = np.minimum(np.power(raw_ratios, float(alpha)), float(max_weight))

    class_weight_dict: dict[int, float] = {
        int(c): float(w) for c, w in zip(classes, tempered)
    }
    sample_weights = tempered[inverse].astype(np.float32)
    return sample_weights, class_weight_dict


def _eval_binary_thresholds_vectorized(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact (f1, fpr) for all thresholds in O(N log N) via sorting."""
    n_samples = len(y_true)
    if n_samples == 0:
        zeros = np.zeros(len(thresholds), dtype=np.float64)
        return zeros, zeros

    order = np.argsort(y_prob, kind="mergesort")
    p_sorted = y_prob[order]
    y_sorted = (y_true[order] == 1).astype(np.int64)

    prefix_pos = np.empty(n_samples + 1, dtype=np.int64)
    prefix_pos[0] = 0
    np.cumsum(y_sorted, out=prefix_pos[1:])

    total_pos = int(prefix_pos[n_samples])
    total_neg = n_samples - total_pos

    # Indices where p_sorted[idx:] >= threshold
    idxs = np.searchsorted(p_sorted, thresholds, side="left")
    pred_pos = n_samples - idxs
    tp = total_pos - prefix_pos[idxs]
    fp = pred_pos - tp
    fn = total_pos - tp

    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, (2.0 * tp) / np.maximum(denom, 1), 0.0)
    fpr = (
        fp.astype(np.float64) / float(total_neg)
        if total_neg > 0
        else np.zeros_like(f1, dtype=np.float64)
    )
    return f1, fpr


def tune_binary_threshold_on_val(
    y_val: np.ndarray,
    y_prob_val: np.ndarray,
    grid_size: int = 181,
    min_thresh: float = 0.05,
    max_thresh: float = 0.95,
) -> dict[str, float]:
    """Tune binary decision threshold strictly on validation predictions to maximize F1.

    Evaluates candidate thresholds from np.linspace(min_thresh, max_thresh, grid_size),
    plus 0.5, plus quantiles of y_prob_val in [0.05, 0.95].
    """
    y_arr = np.asarray(y_val, dtype=np.int64)
    p_arr = _extract_binary_pos_prob(y_prob_val)
    if len(y_arr) != len(p_arr):
        raise ValueError(
            f"Length mismatch: y_val ({len(y_arr)}) vs y_prob_val ({len(p_arr)})."
        )

    grid = np.linspace(float(min_thresh), float(max_thresh), int(max(grid_size, 2)))
    if len(p_arr) > 0:
        q_levels = np.linspace(0.05, 0.95, 37)
        q_vals = np.quantile(p_arr, q_levels)
        q_vals = q_vals[(q_vals >= min_thresh) & (q_vals <= max_thresh)]
    else:
        q_vals = np.empty(0, dtype=np.float64)

    candidates = np.unique(
        np.concatenate([grid, np.array([0.5], dtype=np.float64), q_vals])
    )

    f1_arr, fpr_arr = _eval_binary_thresholds_vectorized(y_arr, p_arr, candidates)

    # Locate baseline threshold 0.5
    idx_05 = int(np.argmin(np.abs(candidates - 0.5)))
    val_f1_at_05 = float(f1_arr[idx_05])
    val_fpr_at_05 = float(fpr_arr[idx_05])

    # Select threshold maximizing F1; tie-break by lower FPR, then closeness to 0.5
    max_f1 = float(np.max(f1_arr)) if len(f1_arr) > 0 else 0.0
    tied_idxs = np.where(f1_arr >= max_f1 - 1e-12)[0]
    best_idx = int(
        min(
            tied_idxs,
            key=lambda i: (round(float(fpr_arr[i]), 12), abs(float(candidates[i]) - 0.5)),
        )
    )

    return {
        "best_threshold": float(candidates[best_idx]),
        "val_f1_at_0.5": val_f1_at_05,
        "val_f1_at_best": float(f1_arr[best_idx]),
        "val_fpr_at_0.5": val_fpr_at_05,
        "val_fpr_at_best": float(fpr_arr[best_idx]),
    }


def predict_binary_with_threshold(
    y_prob: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Apply binary probability decision threshold and return int64 predictions."""
    p_arr = _extract_binary_pos_prob(y_prob)
    return (p_arr >= float(threshold)).astype(np.int64)


def apply_multiclass_priors(
    prob: np.ndarray,
    gamma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale multiclass probabilities by per-class multipliers gamma and renormalize.

    Returns:
        (y_pred_int64, prob_calibrated_float32)
    """
    prob_arr = np.asarray(prob, dtype=np.float64)
    gamma_arr = np.asarray(gamma, dtype=np.float64)
    if prob_arr.ndim != 2:
        raise ValueError(f"Expected 2D probability array, got shape {prob_arr.shape}.")
    if gamma_arr.ndim != 1 or gamma_arr.shape[0] != prob_arr.shape[1]:
        raise ValueError(
            f"gamma shape {gamma_arr.shape} does not match prob classes {prob_arr.shape[1]}."
        )

    scaled = prob_arr * gamma_arr[np.newaxis, :]
    scaled = np.maximum(np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    row_sums = scaled.sum(axis=1, keepdims=True)
    n_classes = prob_arr.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        prob_norm = np.where(
            row_sums > 0.0,
            scaled / np.where(row_sums > 0.0, row_sums, 1.0),
            1.0 / float(n_classes),
        )
    prob_cal = prob_norm.astype(np.float32)
    y_pred = np.argmax(prob_cal, axis=1).astype(np.int64)
    return y_pred, prob_cal


def _multiclass_macro_weighted_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    classes_present: np.ndarray,
    true_counts: np.ndarray,
) -> tuple[float, float]:
    """Fast exact Macro-F1 and Weighted-F1 over classes present in y_true."""
    n_samples = len(y_true)
    if n_samples == 0 or len(classes_present) == 0:
        return 0.0, 0.0

    pred_counts = np.bincount(y_pred, minlength=n_classes).astype(np.float64)
    correct_mask = y_true == y_pred
    tp_counts = np.bincount(y_true[correct_mask], minlength=n_classes).astype(np.float64)

    denom = pred_counts + true_counts
    f1_per_class = np.where(denom > 0.0, (2.0 * tp_counts) / np.maximum(denom, 1.0), 0.0)

    present_f1 = f1_per_class[classes_present]
    present_true = true_counts[classes_present]
    macro_f1 = float(np.mean(present_f1))
    weighted_f1 = float(np.sum(present_f1 * present_true) / float(n_samples))
    return macro_f1, weighted_f1


def tune_multiclass_priors_on_val(
    y_val: np.ndarray,
    prob_val: np.ndarray,
    max_iter: int = 3,
) -> dict[str, Any]:
    """Coordinate-ascent optimization of per-class multipliers gamma strictly on validation.

    Maximizes validation Macro-F1 subject to validation Weighted-F1 not dropping by
    more than 0.0015 relative to the uncalibrated (gamma = 1) baseline.
    """
    y_arr = np.asarray(y_val, dtype=np.int64)
    prob_arr = np.asarray(prob_val, dtype=np.float64)
    if prob_arr.ndim != 2:
        raise ValueError(f"Expected 2D prob_val array, got shape {prob_arr.shape}.")
    prob_arr = np.maximum(
        np.nan_to_num(prob_arr, nan=0.0, posinf=0.0, neginf=0.0), 0.0
    )
    n_samples, n_classes = prob_arr.shape
    if len(y_arr) != n_samples:
        raise ValueError(
            f"Length mismatch: y_val ({len(y_arr)}) vs prob_val ({n_samples})."
        )
    if n_samples > 0 and (int(np.min(y_arr)) < 0 or int(np.max(y_arr)) >= n_classes):
        raise ValueError(
            f"y_val labels must lie in [0, {n_classes - 1}], got [{int(np.min(y_arr))}, {int(np.max(y_arr))}]."
        )

    gamma = np.ones(n_classes, dtype=np.float64)
    classes_present = np.unique(y_arr)
    true_counts = np.bincount(y_arr, minlength=n_classes).astype(np.float64)

    y_pred_init, _ = apply_multiclass_priors(prob_arr, gamma)
    val_macro_before, val_weighted_before = _multiclass_macro_weighted_f1(
        y_arr, y_pred_init, n_classes, classes_present, true_counts
    )

    if n_samples == 0 or n_classes <= 1 or max_iter <= 0:
        return {
            "gamma": gamma,
            "val_macro_f1_before": val_macro_before,
            "val_macro_f1_after": val_macro_before,
            "val_weighted_f1_before": val_weighted_before,
            "val_weighted_f1_after": val_weighted_before,
        }

    # Log-spaced search grid in [0.05, 20.0] guaranteed to include 1.0
    grid = np.unique(
        np.concatenate(
            [np.geomspace(0.05, 20.0, 61), np.array([1.0], dtype=np.float64)]
        )
    )
    n_grid = len(grid)
    min_weighted_f1 = val_weighted_before - 0.0015

    best_macro = val_macro_before
    # Maintain scaled scores S = prob_arr * gamma[None, :]
    scaled = prob_arr.copy()
    row_indices = np.arange(n_samples)

    for _ in range(int(max_iter)):
        improved_any = False
        for c in range(n_classes):
            # Best competing class j != c and its current scaled score
            saved_col = scaled[:, c].copy()
            scaled[:, c] = -np.inf
            other_best_idx = np.argmax(scaled, axis=1)
            other_best_val = scaled[row_indices, other_best_idx]
            scaled[:, c] = saved_col

            p_c = prob_arr[:, c]
            # Critical multiplier r_i where class c overtakes other_best_idx[i]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(p_c > 0.0, other_best_val / p_c, np.inf)
            # Handle degenerate all-zero rows where tie breaks by smaller class index
            zero_tie = (p_c == 0.0) & (other_best_val == 0.0) & (c < other_best_idx)
            ratio = np.where(zero_tie, 0.0, ratio)

            # Tie-breaking matches np.argmax (smallest class index wins exact ties)
            bins = np.empty(n_samples, dtype=np.int64)
            lt_mask = c < other_best_idx
            bins[lt_mask] = np.searchsorted(grid, ratio[lt_mask], side="left")
            bins[~lt_mask] = np.searchsorted(grid, ratio[~lt_mask], side="right")

            # Base counts when nobody picks class c (all samples use other_best_idx)
            base_pred = np.bincount(other_best_idx, minlength=n_classes).astype(np.int64)
            base_tp = np.bincount(
                other_best_idx[other_best_idx == y_arr], minlength=n_classes
            ).astype(np.int64)

            # Accumulate transitions into class c at each grid index b in 0..n_grid-1
            valid_switch = bins < n_grid
            if np.any(valid_switch):
                b_sw = bins[valid_switch]
                o_sw = other_best_idx[valid_switch]
                y_sw = y_arr[valid_switch]

                delta_pred_to_c = np.bincount(b_sw, minlength=n_grid).astype(np.int64)
                delta_tp_to_c = np.bincount(
                    b_sw[y_sw == c], minlength=n_grid
                ).astype(np.int64)

                flat_pred_from = b_sw * n_classes + o_sw
                delta_pred_from = np.bincount(
                    flat_pred_from, minlength=n_grid * n_classes
                ).reshape(n_grid, n_classes)

                match_other = o_sw == y_sw
                if np.any(match_other):
                    flat_tp_from = b_sw[match_other] * n_classes + o_sw[match_other]
                    delta_tp_from = np.bincount(
                        flat_tp_from, minlength=n_grid * n_classes
                    ).reshape(n_grid, n_classes)
                else:
                    delta_tp_from = np.zeros((n_grid, n_classes), dtype=np.int64)

                cum_pred_to_c = np.cumsum(delta_pred_to_c)
                cum_tp_to_c = np.cumsum(delta_tp_to_c)
                cum_pred_from = np.cumsum(delta_pred_from, axis=0)
                cum_tp_from = np.cumsum(delta_tp_from, axis=0)
            else:
                cum_pred_to_c = np.zeros(n_grid, dtype=np.int64)
                cum_tp_to_c = np.zeros(n_grid, dtype=np.int64)
                cum_pred_from = np.zeros((n_grid, n_classes), dtype=np.int64)
                cum_tp_from = np.zeros((n_grid, n_classes), dtype=np.int64)

            pred_all = base_pred[np.newaxis, :] - cum_pred_from
            pred_all[:, c] = cum_pred_to_c

            tp_all = base_tp[np.newaxis, :] - cum_tp_from
            tp_all[:, c] = cum_tp_to_c

            denom_all = pred_all.astype(np.float64) + true_counts[np.newaxis, :]
            f1_all = np.where(
                denom_all > 0.0,
                (2.0 * tp_all.astype(np.float64)) / np.maximum(denom_all, 1.0),
                0.0,
            )

            present_f1_all = f1_all[:, classes_present]
            present_true = true_counts[classes_present]
            macro_all = np.mean(present_f1_all, axis=1)
            weighted_all = (
                present_f1_all @ present_true
            ) / float(n_samples)

            feasible = weighted_all >= (min_weighted_f1 - 1e-12)
            if not np.any(feasible):
                continue

            feasible_idxs = np.where(feasible)[0]
            cand_best_macro = float(np.max(macro_all[feasible_idxs]))
            if cand_best_macro > best_macro + 1e-12:
                tied = feasible_idxs[macro_all[feasible_idxs] >= cand_best_macro - 1e-12]
                # Tie-break by higher weighted F1, then multiplier closest to 1.0
                chosen_m = int(
                    min(
                        tied,
                        key=lambda m: (
                            -round(float(weighted_all[m]), 12),
                            abs(float(np.log(grid[m]))),
                        ),
                    )
                )
                cand_g = float(grid[chosen_m])

                # Verify exact float32 apply_multiclass_priors metrics before committing
                gamma_try = gamma.copy()
                gamma_try[c] = cand_g
                y_pred_try, _ = apply_multiclass_priors(prob_arr, gamma_try)
                exact_macro, exact_weighted = _multiclass_macro_weighted_f1(
                    y_arr, y_pred_try, n_classes, classes_present, true_counts
                )
                if (
                    exact_weighted >= min_weighted_f1 - 1e-12
                    and exact_macro > best_macro + 1e-12
                ):
                    gamma[c] = cand_g
                    scaled[:, c] = prob_arr[:, c] * cand_g
                    best_macro = exact_macro
                    improved_any = True

        if not improved_any:
            break

    y_pred_after, _ = apply_multiclass_priors(prob_arr, gamma)
    val_macro_after, val_weighted_after = _multiclass_macro_weighted_f1(
        y_arr, y_pred_after, n_classes, classes_present, true_counts
    )

    return {
        "gamma": gamma,
        "val_macro_f1_before": float(val_macro_before),
        "val_macro_f1_after": float(val_macro_after),
        "val_weighted_f1_before": float(val_weighted_before),
        "val_weighted_f1_after": float(val_weighted_after),
    }


def _stratified_sample_indices(
    y: np.ndarray,
    n_target: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw n_target stratified row indices from y using largest-remainder allocation."""
    n_total = len(y)
    if n_target <= 0 or n_total == 0:
        return np.empty(0, dtype=np.int64)

    classes, counts = np.unique(y, return_counts=True)
    exact_alloc = (counts.astype(np.float64) / float(n_total)) * float(n_target)
    base_alloc = np.floor(exact_alloc).astype(np.int64)

    # Ensure every class gets at least 1 sample when n_target >= len(classes)
    if n_target >= len(classes):
        zero_mask = base_alloc == 0
        if np.any(zero_mask):
            base_alloc[zero_mask] = 1

    diff = int(n_target - int(np.sum(base_alloc)))
    if diff > 0:
        remainders = exact_alloc - np.floor(exact_alloc)
        order = np.argsort(-remainders, kind="mergesort")
        for idx in order[:diff]:
            base_alloc[idx] += 1
    elif diff < 0:
        # Reduce from largest classes (keeping at least 1)
        order = np.argsort(-base_alloc, kind="mergesort")
        remaining = -diff
        for idx in order:
            if remaining == 0:
                break
            can_take = max(0, int(base_alloc[idx]) - 1)
            take = min(can_take, remaining)
            base_alloc[idx] -= take
            remaining -= take

    picked_chunks: list[np.ndarray] = []
    for cls_val, k_c in zip(classes, base_alloc):
        if k_c <= 0:
            continue
        cls_indices = np.flatnonzero(y == cls_val)
        replace = int(k_c) > len(cls_indices)
        chosen = rng.choice(cls_indices, size=int(k_c), replace=replace)
        picked_chunks.append(chosen.astype(np.int64, copy=False))

    if not picked_chunks:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(picked_chunks)


def build_noise_augmented_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight_train: np.ndarray,
    seed: int,
    aug_ratio: float = 0.5,
    sigmas: tuple[float, ...] = (0.1, 0.25, 0.5),
    continuous_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct noise-augmented training set strictly from standardized X_train.

    Keeps 100% of clean (X_train, y_train, sample_weight_train) and appends
    int(len(X_train) * aug_ratio) stratified rows perturbed with Gaussian noise
    N(0, sigma^2 I) across `sigmas`, clipped to [-10.0, 10.0] in float32.

    If `continuous_mask` is provided (bool array of shape (n_features,)),
    continuous features are perturbed with full sigma and discrete features with
    0.5 * sigma so tree models learn wide split margins around zero/flag constants.
    """
    X_clean = np.asarray(X_train, dtype=np.float32)
    y_clean = np.asarray(y_train)
    w_clean = np.asarray(sample_weight_train, dtype=np.float32)

    if X_clean.ndim != 2:
        raise ValueError(f"Expected 2D X_train, got shape {X_clean.shape}.")
    n_clean, n_features = X_clean.shape
    if len(y_clean) != n_clean or len(w_clean) != n_clean:
        raise ValueError(
            "X_train, y_train, and sample_weight_train must have identical lengths."
        )
    if aug_ratio < 0.0:
        raise ValueError(f"aug_ratio must be non-negative, got {aug_ratio}.")

    n_aug = int(n_clean * float(aug_ratio))
    if n_aug == 0 or len(sigmas) == 0 or n_clean == 0:
        return X_clean.copy(), y_clean.copy(), w_clean.copy()

    if continuous_mask is not None:
        mask_arr = np.asarray(continuous_mask, dtype=bool)
        if mask_arr.shape != (n_features,):
            raise ValueError(
                f"continuous_mask shape {mask_arr.shape} must match n_features ({n_features})."
            )
        feat_scale = np.where(mask_arr, np.float32(1.0), np.float32(0.5))
    else:
        feat_scale = None

    rng = np.random.default_rng(int(seed))
    aug_indices = _stratified_sample_indices(y_clean, n_aug, rng)
    # Shuffle selected indices so classes are evenly mixed across sigma chunks
    rng.shuffle(aug_indices)

    X_out = np.empty((n_clean + n_aug, n_features), dtype=np.float32)
    y_out = np.empty(n_clean + n_aug, dtype=y_clean.dtype)
    w_out = np.empty(n_clean + n_aug, dtype=np.float32)

    X_out[:n_clean] = X_clean
    y_out[:n_clean] = y_clean
    w_out[:n_clean] = w_clean

    sigma_chunks = np.array_split(aug_indices, len(sigmas))
    write_pos = n_clean
    for sigma_val, chunk_idx in zip(sigmas, sigma_chunks):
        chunk_len = len(chunk_idx)
        if chunk_len == 0:
            continue
        sigma_f32 = np.float32(sigma_val)
        X_chunk = X_clean[chunk_idx].copy()
        if sigma_f32 > 0.0:
            noise = rng.normal(
                loc=0.0, scale=float(sigma_f32), size=(chunk_len, n_features)
            ).astype(np.float32)
            if feat_scale is not None:
                noise *= feat_scale[np.newaxis, :]
            X_chunk += noise
            np.clip(X_chunk, np.float32(-10.0), np.float32(10.0), out=X_chunk)

        X_out[write_pos : write_pos + chunk_len] = X_chunk
        y_out[write_pos : write_pos + chunk_len] = y_clean[chunk_idx]
        w_out[write_pos : write_pos + chunk_len] = w_clean[chunk_idx]
        write_pos += chunk_len

    return X_out, y_out, w_out
