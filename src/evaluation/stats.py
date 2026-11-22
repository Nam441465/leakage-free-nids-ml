"""Statistical evaluation module for NIDS ML benchmarks.

Implements:
1. Mean ± SD across random seeds {0, 1, 2}.
2. Paired differences with 95% Student's t-confidence intervals across seeds.
3. Paired t-tests with Holm-Bonferroni family-wise error rate correction.
"""

from __future__ import annotations

from typing import Sequence, Any
import numpy as np
import scipy.stats as stats


def compute_mean_std(
    values: Sequence[float],
    as_percentage: bool = False,
    precision: int = 2,
) -> dict[str, Any]:
    """Compute mean and sample standard deviation (ddof=1) across seeds."""
    arr = np.asarray(values, dtype=np.float64)
    if as_percentage:
        arr = arr * 100.0

    mean_val = float(np.mean(arr))
    if len(arr) > 1:
        std_val = float(np.std(arr, ddof=1))
    else:
        std_val = 0.0

    formatted = f"{mean_val:.{precision}f} ± {std_val:.{precision}f}"
    return {
        "mean": mean_val,
        "std": std_val,
        "formatted": formatted,
        "values": arr.tolist(),
    }


def compute_paired_difference_ci(
    values_a: Sequence[float],
    values_b: Sequence[float],
    confidence: float = 0.95,
    as_percentage: bool = False,
) -> dict[str, float]:
    """Compute paired differences (A - B), sample mean, standard error,

    and Student's t confidence interval (default 95%).
    """
    arr_a = np.asarray(values_a, dtype=np.float64)
    arr_b = np.asarray(values_b, dtype=np.float64)

    if len(arr_a) != len(arr_b):
        raise ValueError(
            f"Arrays must have equal length, got {len(arr_a)} and {len(arr_b)}"
        )

    if as_percentage:
        arr_a = arr_a * 100.0
        arr_b = arr_b * 100.0

    diffs = arr_a - arr_b
    n = len(diffs)
    mean_diff = float(np.mean(diffs))

    if n < 2:
        return {
            "mean_diff": mean_diff,
            "std_diff": 0.0,
            "se": 0.0,
            "ci_lower": mean_diff,
            "ci_upper": mean_diff,
            "margin_of_error": 0.0,
        }

    df = n - 1
    std_diff = float(np.std(diffs, ddof=1))
    se = std_diff / np.sqrt(n)

    alpha = 1.0 - confidence
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
    margin_of_error = t_crit * se

    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "se": se,
        "ci_lower": mean_diff - margin_of_error,
        "ci_upper": mean_diff + margin_of_error,
        "margin_of_error": margin_of_error,
    }


def holm_bonferroni_correction(p_values: Sequence[float]) -> np.ndarray:
    """Apply step-down Holm-Bonferroni correction to a sequence of p-values."""
    m = len(p_values)
    if m == 0:
        return np.array([], dtype=np.float64)

    p_arr = np.asarray(p_values, dtype=np.float64)
    # Sort p-values with original indices
    sort_idx = np.argsort(p_arr)
    sorted_p = p_arr[sort_idx]

    adjusted = np.empty(m, dtype=np.float64)
    for i, p in enumerate(sorted_p):
        multiplier = m - i
        adjusted[i] = p * multiplier

    # Enforce monotonicity: p_adj[i] = max(p_adj[i], p_adj[i-1]) and <= 1.0
    for i in range(1, m):
        if adjusted[i] < adjusted[i - 1]:
            adjusted[i] = adjusted[i - 1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    # Invert sorting back to original order
    inv_idx = np.empty(m, dtype=np.int64)
    inv_idx[sort_idx] = np.arange(m)
    return adjusted[inv_idx]


def paired_t_test_with_holm(
    comparisons: list[dict[str, Any]],
    alpha: float = 0.05,
    as_percentage: bool = False,
) -> list[dict[str, Any]]:
    """Perform paired Student's t-tests for multiple comparisons across seeds

    and apply Holm-Bonferroni correction.

    Each comparison dict should have:
    - 'name_a': Name of model A
    - 'name_b': Name of model B
    - 'values_a': Sequence of metric values across seeds for A
    - 'values_b': Sequence of metric values across seeds for B
    """
    raw_results = []
    p_values = []

    for comp in comparisons:
        name_a = comp["name_a"]
        name_b = comp["name_b"]
        v_a = np.asarray(comp["values_a"], dtype=np.float64)
        v_b = np.asarray(comp["values_b"], dtype=np.float64)

        if as_percentage:
            v_a = v_a * 100.0
            v_b = v_b * 100.0

        diffs = v_a - v_b
        n = len(diffs)
        mean_diff = float(np.mean(diffs))

        if n < 2:
            t_stat = 0.0
            p_val = 1.0
            ci = compute_paired_difference_ci(v_a, v_b, as_percentage=False)
        else:
            df = n - 1
            std_diff = float(np.std(diffs, ddof=1))
            se = std_diff / np.sqrt(n)

            if se < 1e-12:
                # Diffs are identical across seeds
                t_stat = 0.0 if abs(mean_diff) < 1e-12 else np.sign(mean_diff) * 1e6
                p_val = 1.0 if abs(mean_diff) < 1e-12 else 0.0
            else:
                t_stat = float(mean_diff / se)
                p_val = float(2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=df)))

            ci = compute_paired_difference_ci(v_a, v_b, as_percentage=False)

        raw_results.append(
            {
                "comparison": f"{name_a} vs {name_b}",
                "model_a": name_a,
                "model_b": name_b,
                "mean_diff": mean_diff,
                "ci_lower": ci["ci_lower"],
                "ci_upper": ci["ci_upper"],
                "t_stat": t_stat,
                "p_raw": p_val,
            }
        )
        p_values.append(p_val)

    # Apply Holm correction
    p_adj = holm_bonferroni_correction(p_values)

    final_results = []
    for res, padj in zip(raw_results, p_adj):
        res_copy = dict(res)
        res_copy["p_adjusted"] = float(padj)
        res_copy["reject_null"] = bool(padj < alpha)
        final_results.append(res_copy)

    return final_results
