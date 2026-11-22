"""Publication-grade plotting module reproducing Figures 4, 6, 7b, 9, and 10
from 2611.04660v1.pdf (tabular ML baselines only; no GNN/graph models).

Usage:
    python -m src.evaluation.plot_figures \
        --tables-dir results/tables \
        --output-dir results/figures
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger("plot_figures")

# Canonical class orders matching Tables 2 & 3 and Figures 6, 7b, 9 of 2611.04660v1.pdf
CICIDS2017_ORDER: list[str] = [
    "Benign",
    "DoS",
    "DDoS",
    "PortScan",
    "BruteForce",
    "Web",
    "Bot",
    "Rare",
]

NF_UNSW_ORDER: list[str] = [
    "Benign",
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Rare",
]

LOFO_ORDER: list[str] = [
    "DoS",
    "DDoS",
    "PortScan",
    "BruteForce",
    "Web",
    "Bot",
]

MODEL_ORDER: list[str] = ["xgboost", "lightgbm", "random_forest"]

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "random_forest": "Random Forest",
}

# Colorblind-safe Okabe-Ito-inspired palette
MODEL_COLORS: dict[str, str] = {
    "xgboost": "#0173B2",
    "lightgbm": "#DE8F05",
    "random_forest": "#029E73",
}

MODEL_MARKERS: dict[str, str] = {
    "xgboost": "o",
    "lightgbm": "s",
    "random_forest": "^",
}

FALLBACK_COLORS: list[str] = ["#CC78BC", "#CA9161", "#949494", "#ECE133", "#56B4E9"]
FALLBACK_MARKERS: list[str] = ["D", "v", "P", "X", "*"]

DPI: int = 300


def _apply_pub_style() -> None:
    """Configure clean, publication-grade matplotlib/seaborn styling."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update(
        {
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "grid.color": "#DCDCDC",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#CCCCCC",
            "savefig.dpi": DPI,
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ordered_models(models_dict: dict[str, Any]) -> list[str]:
    """Return model keys in canonical order (XGBoost, LightGBM, Random Forest),
    followed by any additional model keys in deterministic order.
    """
    present = [m for m in MODEL_ORDER if m in models_dict]
    extras = [m for m in models_dict if m not in MODEL_ORDER]
    return present + extras


def _display_name(model_key: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_key, model_key.replace("_", " ").title())


def _model_color(model_key: str, idx: int) -> str:
    return MODEL_COLORS.get(model_key, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])


def _model_marker(model_key: str, idx: int) -> str:
    return MODEL_MARKERS.get(model_key, FALLBACK_MARKERS[idx % len(FALLBACK_MARKERS)])


def reorder_per_class_matrix(
    models_dict: dict[str, Any],
    canonical_order: list[str],
) -> tuple[list[str], list[str], np.ndarray]:
    """Build a (n_models, n_classes) matrix of per-class F1 (%) means
    ordered strictly by canonical_order (Table 2 / Table 3).
    """
    model_keys = _ordered_models(models_dict)
    if not model_keys:
        raise ValueError("No models found in multiclass summary JSON.")

    available_classes: set[str] = set()
    for m in model_keys:
        available_classes.update(models_dict[m].get("per_class_f1", {}).keys())

    ordered_classes = [c for c in canonical_order if c in available_classes]
    if not ordered_classes:
        raise ValueError(
            f"None of the canonical classes {canonical_order} were found in {sorted(available_classes)}."
        )

    matrix = np.full((len(model_keys), len(ordered_classes)), np.nan, dtype=np.float64)
    for r, m in enumerate(model_keys):
        pcf1 = models_dict[m].get("per_class_f1", {})
        for c_idx, cls_name in enumerate(ordered_classes):
            if cls_name in pcf1:
                matrix[r, c_idx] = float(pcf1[cls_name]["mean"])

    return model_keys, ordered_classes, matrix


def reorder_confusion_matrix(
    cm: np.ndarray,
    source_order: list[str],
    target_order: list[str],
) -> tuple[list[str], np.ndarray]:
    """Reorder both rows AND columns of a square confusion matrix from
    `source_order` (e.g. alphabetical JSON `class_order`) to `target_order`
    (e.g. Table 2 `CICIDS2017_ORDER`).
    """
    cm_arr = np.asarray(cm, dtype=np.float64)
    if cm_arr.ndim != 2 or cm_arr.shape[0] != cm_arr.shape[1]:
        raise ValueError(f"Expected square 2D confusion matrix, got shape {cm_arr.shape}.")
    if cm_arr.shape[0] != len(source_order):
        raise ValueError(
            f"Confusion matrix size ({cm_arr.shape[0]}) does not match "
            f"source_order length ({len(source_order)})."
        )

    index_map = {cls_name: idx for idx, cls_name in enumerate(source_order)}
    ordered_classes = [c for c in target_order if c in index_map]
    if not ordered_classes:
        raise ValueError(
            f"No overlap between source_order {source_order} and target_order {target_order}."
        )

    perm = [index_map[c] for c in ordered_classes]
    reordered = cm_arr[np.ix_(perm, perm)]
    return ordered_classes, reordered


def format_cm_annotations(cm_pct: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Format percentage confusion matrix cells to 1 decimal place,
    leaving cells strictly below `threshold` (%) blank (matching Figure 7b).
    """
    annot = np.empty(cm_pct.shape, dtype=object)
    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            val = float(cm_pct[i, j])
            annot[i, j] = "" if val < threshold else f"{val:.1f}"
    return annot


def _extract_row_normalized_cm(model_entry: dict[str, Any]) -> np.ndarray:
    """Return row-normalized (%) confusion matrix from a multiclass model summary."""
    if "confusion_matrix_normalized" in model_entry:
        cm_norm = np.asarray(model_entry["confusion_matrix_normalized"], dtype=np.float64)
        # Guard in case a file stores proportions in [0, 1] instead of percentages [0, 100]
        if cm_norm.size > 0 and float(np.nanmax(cm_norm)) <= 1.0 + 1e-9:
            cm_norm = cm_norm * 100.0
        return cm_norm

    if "confusion_matrix_raw" in model_entry:
        cm_raw = np.asarray(model_entry["confusion_matrix_raw"], dtype=np.float64)
        row_sums = cm_raw.sum(axis=1, keepdims=True)
        return np.divide(
            cm_raw,
            row_sums,
            out=np.zeros_like(cm_raw, dtype=np.float64),
            where=row_sums > 0,
        ) * 100.0

    raise KeyError(
        "Model entry must contain 'confusion_matrix_normalized' or 'confusion_matrix_raw'."
    )


def plot_fig4_binary_f1(
    cicids_binary_json: Path,
    nf_unsw_binary_json: Path,
    output_path: Path,
) -> Path:
    """Figure 4: Two-panel comparison of Binary F1 (%) on CICIDS2017 and
    NF-UNSW-NB15-v2 showing individual seed points and mean ± SD error bars.
    """
    _apply_pub_style()
    cicids_data = _load_json(Path(cicids_binary_json))
    nf_data = _load_json(Path(nf_unsw_binary_json))

    panels = [
        ("(a) CICIDS2017", cicids_data["models"]),
        ("(b) NF-UNSW-NB15-v2", nf_data["models"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)
    try:
        for ax, (title, models_dict) in zip(axes, panels):
            model_keys = _ordered_models(models_dict)
            x_positions = np.arange(len(model_keys), dtype=np.float64)

            all_y_vals: list[float] = []
            for idx, m_key in enumerate(model_keys):
                f1_info = models_dict[m_key]["f1"]
                mean_val = float(f1_info["mean"])
                std_val = float(f1_info["std"])
                seed_vals = [float(v) for v in f1_info.get("values", [])]
                color = _model_color(m_key, idx)
                marker = _model_marker(m_key, idx)

                all_y_vals.extend([mean_val - std_val, mean_val + std_val])
                all_y_vals.extend(seed_vals)

                # Individual seed scatter points with deterministic horizontal spread
                if seed_vals:
                    n_s = len(seed_vals)
                    offsets = (
                        np.linspace(-0.14, 0.14, n_s) if n_s > 1 else np.array([0.0])
                    )
                    ax.scatter(
                        x_positions[idx] + offsets,
                        seed_vals,
                        color=color,
                        alpha=0.55,
                        s=44,
                        edgecolors="white",
                        linewidths=0.6,
                        zorder=3,
                        label="Individual seeds" if idx == 0 else None,
                    )

                # Mean ± SD marker and error bar
                ax.errorbar(
                    x_positions[idx],
                    mean_val,
                    yerr=std_val,
                    fmt=marker,
                    color=color,
                    ecolor="#222222",
                    elinewidth=1.6,
                    capsize=6,
                    capthick=1.4,
                    markersize=8.5,
                    markeredgecolor="white",
                    markeredgewidth=0.9,
                    zorder=4,
                    label="Mean ± SD" if idx == 0 else None,
                )

            y_min = min(all_y_vals)
            y_max = max(all_y_vals)
            span = max(y_max - y_min, 0.20)
            low_lim = max(0.0, y_min - 0.30 * span)
            high_lim = min(100.15, y_max + 0.45 * span)
            ax.set_ylim(low_lim, high_lim)

            # Annotate mean ± SD above each model's highest point
            text_offset = 0.06 * (high_lim - low_lim)
            for idx, m_key in enumerate(model_keys):
                f1_info = models_dict[m_key]["f1"]
                mean_val = float(f1_info["mean"])
                std_val = float(f1_info["std"])
                seed_vals = [float(v) for v in f1_info.get("values", [])]
                top_y = max([mean_val + std_val] + seed_vals)
                ax.text(
                    x_positions[idx],
                    min(high_lim - 0.04 * (high_lim - low_lim), top_y + text_offset),
                    f"{mean_val:.2f} ± {std_val:.2f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                    fontweight="semibold",
                    color="#222222",
                )

            ax.set_xticks(x_positions)
            ax.set_xticklabels([_display_name(m) for m in model_keys], fontsize=10.5)
            ax.set_xlim(-0.55, len(model_keys) - 0.45)
            ax.set_ylabel("Binary F1 (%)", fontsize=11)
            ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
            ax.yaxis.grid(True, linestyle="--", alpha=0.65)
            ax.xaxis.grid(False)
            sns.despine(ax=ax)

        axes[0].legend(loc="lower left", fontsize=9.2)
        fig.suptitle(
            "Binary Intrusion Detection Performance Across 3 Random Seeds",
            fontsize=13,
            fontweight="bold",
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
        logger.info("Saved Figure 4 to %s", output_path)
        return output_path
    finally:
        plt.close(fig)


def plot_fig6_per_class_f1(
    cicids_multiclass_json: Path,
    nf_unsw_multiclass_json: Path,
    output_path: Path,
) -> Path:
    """Figure 6: Two heatmaps of per-class F1 (%) means over 3 seeds across models
    for (a) CICIDS2017 and (b) NF-UNSW-NB15-v2 in exact Table 2 & Table 3 class order.
    """
    _apply_pub_style()
    cicids_data = _load_json(Path(cicids_multiclass_json))
    nf_data = _load_json(Path(nf_unsw_multiclass_json))

    cicids_models, cicids_classes, cicids_mat = reorder_per_class_matrix(
        cicids_data["models"], CICIDS2017_ORDER
    )
    nf_models, nf_classes, nf_mat = reorder_per_class_matrix(
        nf_data["models"], NF_UNSW_ORDER
    )

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.8), constrained_layout=True)
    try:
        panel_specs = [
            (
                axes[0],
                cicids_mat,
                [_display_name(m) for m in cicids_models],
                cicids_classes,
                "(a) CICIDS2017 — Per-Class F1 (%) (Mean over 3 Seeds)",
            ),
            (
                axes[1],
                nf_mat,
                [_display_name(m) for m in nf_models],
                nf_classes,
                "(b) NF-UNSW-NB15-v2 — Per-Class F1 (%) (Mean over 3 Seeds)",
            ),
        ]

        for ax, mat, row_labels, col_labels, title in panel_specs:
            sns.heatmap(
                mat,
                ax=ax,
                annot=True,
                fmt=".1f",
                cmap="YlGnBu",
                vmin=0.0,
                vmax=100.0,
                linewidths=0.7,
                linecolor="white",
                xticklabels=col_labels,
                yticklabels=row_labels,
                cbar_kws={"label": "F1 (%)", "shrink": 0.92},
                annot_kws={"fontsize": 9.5, "fontweight": "medium"},
            )
            ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=22, ha="right", fontsize=10)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10.5)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
        logger.info("Saved Figure 6 to %s", output_path)
        return output_path
    finally:
        plt.close(fig)


def plot_fig7b_cm_cicids2017(
    cicids_multiclass_json: Path,
    output_path: Path,
    model_name: str = "xgboost",
) -> Path:
    """Figure 7b: Row-normalized (%) confusion matrix of XGBoost on CICIDS2017
    summed over 3 seeds, rows and columns reordered to Table 2 (`CICIDS2017_ORDER`),
    leaving cells < 0.05% blank.
    """
    _apply_pub_style()
    data = _load_json(Path(cicids_multiclass_json))

    source_order = data.get("class_order")
    if not source_order:
        raise KeyError("Multiclass summary JSON missing 'class_order'.")
    if model_name not in data.get("models", {}):
        raise KeyError(f"Model '{model_name}' not found in {cicids_multiclass_json}.")

    cm_pct = _extract_row_normalized_cm(data["models"][model_name])
    ordered_classes, cm_reordered = reorder_confusion_matrix(
        cm_pct, source_order, CICIDS2017_ORDER
    )
    annot = format_cm_annotations(cm_reordered, threshold=0.05)

    fig, ax = plt.subplots(figsize=(8.6, 7.4), constrained_layout=True)
    try:
        sns.heatmap(
            cm_reordered,
            ax=ax,
            annot=annot,
            fmt="",
            cmap="Blues",
            vmin=0.0,
            vmax=100.0,
            square=True,
            linewidths=0.6,
            linecolor="#E2E2E2",
            xticklabels=ordered_classes,
            yticklabels=ordered_classes,
            cbar_kws={"label": "Row-Normalized (%)", "shrink": 0.85},
            annot_kws={"fontsize": 9.5},
        )
        ax.set_xlabel("Predicted Class", fontsize=11.5, fontweight="semibold", labelpad=8)
        ax.set_ylabel("True Class", fontsize=11.5, fontweight="semibold", labelpad=8)
        ax.set_title(
            f"{_display_name(model_name)} — CICIDS2017 Row-Normalized Confusion Matrix (%)\n"
            "(Summed over 3 Seeds; Cells < 0.05% Blank)",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
        logger.info("Saved Figure 7b to %s", output_path)
        return output_path
    finally:
        plt.close(fig)


def plot_fig9_leave_one_out(
    lofo_summary_json: Path,
    output_path: Path,
) -> Path:
    """Figure 9: Bar chart with SD error bars of held-out recall (%) for the
    6 withheld attack families in `LOFO_ORDER` on CICIDS2017.
    """
    _apply_pub_style()
    data = _load_json(Path(lofo_summary_json))
    models_dict = data.get("models", {})
    model_keys = _ordered_models(models_dict)
    if not model_keys:
        raise ValueError("No models found in leave_one_out_summary.json.")

    # Filter strictly to canonical LOFO_ORDER (ignoring overall summary keys)
    families = [
        fam
        for fam in LOFO_ORDER
        if any(
            isinstance(models_dict[m].get(fam), dict)
            and "held_out_recall" in models_dict[m][fam]
            for m in model_keys
        )
    ]
    if not families:
        raise ValueError(
            f"No withheld families from {LOFO_ORDER} found in {lofo_summary_json}."
        )

    n_models = len(model_keys)
    x_positions = np.arange(len(families), dtype=np.float64)
    total_width = 0.72 if n_models > 1 else 0.56
    bar_width = total_width / n_models

    fig, ax = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
    try:
        for m_idx, m_key in enumerate(model_keys):
            m_data = models_dict[m_key]
            means = np.array(
                [
                    float(m_data[fam]["held_out_recall"]["mean"])
                    if fam in m_data and isinstance(m_data[fam], dict)
                    else 0.0
                    for fam in families
                ],
                dtype=np.float64,
            )
            stds = np.array(
                [
                    float(m_data[fam]["held_out_recall"]["std"])
                    if fam in m_data and isinstance(m_data[fam], dict)
                    else 0.0
                    for fam in families
                ],
                dtype=np.float64,
            )

            # Clip error bars within physical [0, 100] recall bounds
            lower_err = np.minimum(means, stds)
            upper_err = np.minimum(100.0 - means, stds)
            offset = (m_idx - (n_models - 1) / 2.0) * bar_width
            bar_x = x_positions + offset
            color = _model_color(m_key, m_idx)

            ax.bar(
                bar_x,
                means,
                width=bar_width * 0.92,
                color=color,
                edgecolor="#222222",
                linewidth=0.7,
                alpha=0.88,
                label=_display_name(m_key),
                zorder=3,
            )
            ax.errorbar(
                bar_x,
                means,
                yerr=np.vstack([lower_err, upper_err]),
                fmt="none",
                ecolor="#222222",
                elinewidth=1.4,
                capsize=4,
                capthick=1.2,
                zorder=4,
            )

            for bx, mean_v, up_e in zip(bar_x, means, upper_err):
                ax.text(
                    bx,
                    min(106.5, mean_v + up_e + 1.8),
                    f"{mean_v:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9.2 if n_models == 1 else 8.2,
                    fontweight="semibold",
                    color="#222222",
                )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(families, fontsize=10.5)
        ax.set_ylim(0.0, 110.0)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_xlabel("Withheld Attack Family (CICIDS2017)", fontsize=11.5, fontweight="semibold")
        ax.set_ylabel("Held-Out Family Recall (%)", fontsize=11.5, fontweight="semibold")
        ax.set_title(
            "Leave-One-Family-Out Zero-Day Detection Recall (Mean ± SD over 3 Seeds)",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax.yaxis.grid(True, linestyle="--", alpha=0.65)
        ax.xaxis.grid(False)
        ax.legend(loc="upper right", fontsize=9.8)
        sns.despine(ax=ax)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
        logger.info("Saved Figure 9 to %s", output_path)
        return output_path
    finally:
        plt.close(fig)


def _parse_sigma_entries(model_noise_dict: dict[str, Any]) -> tuple[list[float], list[float], list[float]]:
    """Parse sorted (sigmas, f1_means, f1_stds) from a noise summary model entry."""
    parsed: list[tuple[float, float, float]] = []
    for key, val in model_noise_dict.items():
        if not isinstance(val, dict) or "f1" not in val:
            continue
        if key.startswith("sigma_"):
            sigma_str = key[len("sigma_") :]
        else:
            sigma_str = key
        try:
            sigma_f = float(sigma_str)
        except ValueError:
            continue
        parsed.append((sigma_f, float(val["f1"]["mean"]), float(val["f1"]["std"])))

    parsed.sort(key=lambda item: item[0])
    sigmas = [p[0] for p in parsed]
    means = [p[1] for p in parsed]
    stds = [p[2] for p in parsed]
    return sigmas, means, stds


def plot_fig10_gaussian_noise(
    noise_summary_json: Path,
    output_path: Path,
) -> Path:
    """Figure 10: Line plot with SD error bars of Binary F1 (%) vs Gaussian feature
    noise sigma in [0.0, 0.1, 0.25, 0.5] across all models in `noise_test_summary.json`.
    """
    _apply_pub_style()
    data = _load_json(Path(noise_summary_json))
    models_dict = data.get("models", {})
    model_keys = _ordered_models(models_dict)
    if not model_keys:
        raise ValueError("No models found in noise_test_summary.json.")

    fig, ax = plt.subplots(figsize=(8.6, 5.0), constrained_layout=True)
    try:
        all_sigmas: set[float] = set()
        for m_idx, m_key in enumerate(model_keys):
            sigmas, means, stds = _parse_sigma_entries(models_dict[m_key])
            if not sigmas:
                continue
            all_sigmas.update(sigmas)

            means_arr = np.asarray(means, dtype=np.float64)
            stds_arr = np.asarray(stds, dtype=np.float64)
            lower_err = np.minimum(means_arr, stds_arr)
            upper_err = np.minimum(100.0 - means_arr, stds_arr)

            color = _model_color(m_key, m_idx)
            marker = _model_marker(m_key, m_idx)

            ax.errorbar(
                sigmas,
                means_arr,
                yerr=np.vstack([lower_err, upper_err]),
                color=color,
                marker=marker,
                markersize=7.5,
                linewidth=2.2,
                elinewidth=1.4,
                capsize=4.5,
                capthick=1.2,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=_display_name(m_key),
                zorder=4,
            )

        if not all_sigmas:
            raise ValueError(f"No valid sigma entries found in {noise_summary_json}.")

        sorted_sigmas = sorted(all_sigmas)
        ax.set_xticks(sorted_sigmas)
        ax.set_xticklabels([f"{s:g}" if s != 0.0 else "0.0" for s in sorted_sigmas], fontsize=10.5)
        ax.set_xlim(min(sorted_sigmas) - 0.03, max(sorted_sigmas) + 0.03)
        ax.set_ylim(-3.0, 105.0)
        ax.set_yticks(np.arange(0, 101, 20))

        ax.set_xlabel(r"Gaussian Feature Noise Level ($\sigma$)", fontsize=11.5, fontweight="semibold")
        ax.set_ylabel("Binary F1 (%)", fontsize=11.5, fontweight="semibold")
        ax.set_title(
            "CICIDS2017 — Binary F1 Robustness Under Gaussian Feature Noise (Mean ± SD)",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
        ax.grid(True, linestyle="--", alpha=0.65)
        ax.legend(loc="upper right", fontsize=10)
        sns.despine(ax=ax)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
        logger.info("Saved Figure 10 to %s", output_path)
        return output_path
    finally:
        plt.close(fig)


def generate_all_figures(tables_dir: Path, output_dir: Path) -> list[Path]:
    """Generate all available paper figures (4, 6, 7b, 9, 10) from summary JSONs
    in `tables_dir`, saving 300 DPI PNGs into `output_dir`.

    Any figure whose required input JSON(s) do not exist yet is skipped cleanly
    with an informative log message.
    """
    tables_dir = Path(tables_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []

    cicids_bin = tables_dir / "cicids2017_binary_summary.json"
    nf_bin = tables_dir / "nf_unsw_binary_summary.json"
    cicids_multi = tables_dir / "cicids2017_multiclass_summary.json"
    nf_multi = tables_dir / "nf_unsw_multiclass_summary.json"
    lofo_json = tables_dir / "leave_one_out_summary.json"
    noise_json = tables_dir / "noise_test_summary.json"

    figure_jobs = [
        (
            "fig4_binary_f1.png",
            [cicids_bin, nf_bin],
            lambda out: plot_fig4_binary_f1(cicids_bin, nf_bin, out),
        ),
        (
            "fig6_multiclass_per_class_f1.png",
            [cicids_multi, nf_multi],
            lambda out: plot_fig6_per_class_f1(cicids_multi, nf_multi, out),
        ),
        (
            "fig7b_xgboost_cm_cicids2017.png",
            [cicids_multi],
            lambda out: plot_fig7b_cm_cicids2017(cicids_multi, out),
        ),
        (
            "fig9_leave_one_out_recall.png",
            [lofo_json],
            lambda out: plot_fig9_leave_one_out(lofo_json, out),
        ),
        (
            "fig10_gaussian_noise_f1.png",
            [noise_json],
            lambda out: plot_fig10_gaussian_noise(noise_json, out),
        ),
    ]

    for filename, required_files, plot_fn in figure_jobs:
        missing = [str(p) for p in required_files if not p.exists()]
        if missing:
            logger.info("Skipping %s — missing required input(s): %s", filename, ", ".join(missing))
            continue
        out_path = output_dir / filename
        generated.append(plot_fn(out_path))

    return generated


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Generate paper figures (4, 6, 7b, 9, 10) from summary JSONs."
    )
    parser.add_argument(
        "--tables-dir",
        type=str,
        default="results/tables",
        help="Directory containing summary JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/figures",
        help="Directory to write 300 DPI PNG figures",
    )
    args = parser.parse_args()
    generated = generate_all_figures(Path(args.tables_dir), Path(args.output_dir))
    logger.info("Generated %d figure(s) in %s", len(generated), args.output_dir)


if __name__ == "__main__":
    main()
