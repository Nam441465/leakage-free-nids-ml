"""Publication-grade plotting module for Phase 4 Improved Tabular ML results.

Generates 3 high-resolution (300 DPI) comparison figures in `results/figures/`:
1. `phase4_fig1_main_comparison.png`:
   3-panel bar chart with SD error bars comparing Paper Baseline vs Phase 4 Improved:
   - (a) NF-UNSW-NB15-v2 Binary F1 (%) & FPR (%)
   - (b) CICIDS2017 Multiclass Macro-F1 (%) & Bot F1 (%)
   - (c) NF-UNSW-NB15-v2 Multiclass Macro-F1 (%)
2. `phase4_fig2_noise_robustness.png`:
   Line plot with SD error bars of Binary F1 (%) vs sigma in {0.0, 0.1, 0.25, 0.5}
   comparing Baseline (Table 12) vs Phase 4 Noise-Robust for XGBoost, LightGBM,
   and Random Forest.
3. `phase4_fig3_hybrid_lofo.png`:
   Grouped bar chart with SD error bars of Unseen Attack Family Recall (%) across
   DoS, DDoS, PortScan, BruteForce, Web, Bot + Overall Seen Recall (%) & Overall
   Benign FPR (%) comparing Baseline XGBoost (Table 11) vs Phase 4 Hybrid
   XGBoost + Benign Anomaly.

Usage:
    python -m src.evaluation.plot_phase4_figures \
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

from src.evaluation.plot_figures import (
    DPI,
    LOFO_ORDER,
    MODEL_COLORS,
    MODEL_DISPLAY_NAMES,
    MODEL_MARKERS,
    MODEL_ORDER,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("plot_phase4_figures")

# Colorblind-safe Okabe-Ito comparison palette
COLOR_BASELINE = "#8C96A0"
COLOR_PHASE4 = "#0173B2"
COLOR_BASELINE_ALT = "#E69F00"
COLOR_PHASE4_ALT = "#D55E00"
COLOR_PHASE4_BOT = "#029E73"
COLOR_BASELINE_BOT = "#A6D8C6"


def _apply_pub_style() -> None:
    """Configure clean, publication-grade matplotlib/seaborn styling."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.12)
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
    if not path.exists():
        raise FileNotFoundError(f"Required table JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_mean_std(entry: dict[str, Any] | None) -> tuple[float, float]:
    """Safely extract (mean, std) pair from a compute_mean_std dictionary."""
    if not isinstance(entry, dict):
        return 0.0, 0.0
    return float(entry.get("mean", 0.0)), float(entry.get("std", 0.0))


def _resolve_baseline_task_dict(
    task_block: dict[str, Any],
    tables_dir: Path,
    dataset: str,
    task: str,
) -> dict[str, Any]:
    """Read baseline metrics from comparison JSON, falling back to Stage 1–3 summary JSON."""
    baseline = task_block.get("baseline", {})
    if baseline:
        return baseline
    fallback_path = tables_dir / f"{dataset}_{task}_summary.json"
    if fallback_path.exists():
        return _load_json(fallback_path).get("models", {})
    return {}


def _resolve_phase4_variant(
    variants: dict[str, Any],
    base_model: str,
) -> dict[str, Any]:
    """Select `phase4_best_<model>`, falling back to `<model>_ratio_calibrated` or `<model>_calibrated`."""
    for key in (
        f"phase4_best_{base_model}",
        f"{base_model}_ratio_calibrated",
        f"{base_model}_calibrated",
    ):
        if key in variants:
            return variants[key]
    return {}


def plot_phase4_main_comparison(
    tables_dir: Path,
    output_dir: Path,
) -> Path:
    """Generate `phase4_fig1_main_comparison.png` (3-panel bar chart with SD error bars)."""
    _apply_pub_style()
    comp = _load_json(tables_dir / "phase4_main_comparison.json")
    datasets_block = comp.get("datasets", {})

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.4), dpi=DPI)
    x = np.arange(len(MODEL_ORDER))
    labels = [MODEL_DISPLAY_NAMES[m] for m in MODEL_ORDER]

    # ------------------------------------------------------------------
    # Panel (a): NF-UNSW-NB15-v2 Binary F1 (%) & FPR (%)
    # ------------------------------------------------------------------
    ax_a = axes[0]
    ax_a_fpr = ax_a.twinx()
    nf_bin_block = datasets_block.get("nf_unsw", {}).get("binary", {})
    nf_bin_base = _resolve_baseline_task_dict(nf_bin_block, tables_dir, "nf_unsw", "binary")
    nf_bin_vars = nf_bin_block.get("variants", {})

    base_f1_m, base_f1_s = [], []
    p4_f1_m, p4_f1_s = [], []
    base_fpr_m, base_fpr_s = [], []
    p4_fpr_m, p4_fpr_s = [], []

    for m in MODEL_ORDER:
        bm = nf_bin_base.get(m, {})
        pm = _resolve_phase4_variant(nf_bin_vars, m)
        m1, s1 = _extract_mean_std(bm.get("f1"))
        m2, s2 = _extract_mean_std(pm.get("f1"))
        m3, s3 = _extract_mean_std(bm.get("fpr_at_0.5"))
        m4, s4 = _extract_mean_std(pm.get("fpr_at_0.5"))
        base_f1_m.append(m1)
        base_f1_s.append(s1)
        p4_f1_m.append(m2)
        p4_f1_s.append(s2)
        base_fpr_m.append(m3)
        base_fpr_s.append(s3)
        p4_fpr_m.append(m4)
        p4_fpr_s.append(s4)

    w = 0.18
    b1 = ax_a.bar(
        x - 1.55 * w,
        base_f1_m,
        width=w,
        yerr=base_f1_s,
        capsize=3,
        color=COLOR_BASELINE,
        edgecolor="#333333",
        linewidth=0.7,
        label="Baseline F1 (%)",
    )
    b2 = ax_a.bar(
        x - 0.50 * w,
        p4_f1_m,
        width=w,
        yerr=p4_f1_s,
        capsize=3,
        color=COLOR_PHASE4,
        edgecolor="#333333",
        linewidth=0.7,
        label="Phase 4 F1 (%)",
    )
    b3 = ax_a_fpr.bar(
        x + 0.55 * w,
        base_fpr_m,
        width=w,
        yerr=base_fpr_s,
        capsize=3,
        color=COLOR_BASELINE_ALT,
        edgecolor="#333333",
        linewidth=0.7,
        hatch="//",
        alpha=0.88,
        label="Baseline FPR (%)",
    )
    b4 = ax_a_fpr.bar(
        x + 1.60 * w,
        p4_fpr_m,
        width=w,
        yerr=p4_fpr_s,
        capsize=3,
        color=COLOR_PHASE4_ALT,
        edgecolor="#333333",
        linewidth=0.7,
        hatch="//",
        alpha=0.92,
        label="Phase 4 FPR (%)",
    )

    f1_min_val = min(base_f1_m + p4_f1_m) if (base_f1_m + p4_f1_m) else 80.0
    ax_a.set_ylim(max(0.0, min(85.0, f1_min_val - 5.0)), 102.5)
    fpr_max_val = max(
        [m + s for m, s in zip(base_fpr_m + p4_fpr_m, base_fpr_s + p4_fpr_s)] + [0.5]
    )
    ax_a_fpr.set_ylim(0.0, max(1.65, fpr_max_val * 1.55))
    ax_a_fpr.grid(False)

    for bars, sds, axis, fmt in [
        (b1, base_f1_s, ax_a, "{:.1f}"),
        (b2, p4_f1_s, ax_a, "{:.1f}"),
        (b3, base_fpr_s, ax_a_fpr, "{:.2f}"),
        (b4, p4_fpr_s, ax_a_fpr, "{:.2f}"),
    ]:
        for bar, sd in zip(bars, sds):
            h = bar.get_height()
            if h > 0:
                axis.annotate(
                    fmt.format(h),
                    xy=(bar.get_x() + bar.get_width() / 2.0, h + sd),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.3,
                    fontweight="bold" if bars in (b2, b4) else "normal",
                )

    ax_a.set_title("(a) NF-UNSW-NB15-v2 Binary: F1 & FPR", fontsize=11.5, fontweight="bold")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, fontsize=9.5)
    ax_a.set_ylabel("Binary F1 (%)", fontsize=10, color=COLOR_PHASE4)
    ax_a_fpr.set_ylabel("False Positive Rate (%)", fontsize=10, color=COLOR_PHASE4_ALT)
    h1, l1 = ax_a.get_legend_handles_labels()
    h2, l2 = ax_a_fpr.get_legend_handles_labels()
    ax_a_fpr.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7.6, ncol=2, framealpha=0.95)

    # ------------------------------------------------------------------
    # Panel (b): CICIDS2017 Multiclass Macro-F1 (%) & Bot F1 (%)
    # ------------------------------------------------------------------
    ax_b = axes[1]
    cic_mc_block = datasets_block.get("cicids2017", {}).get("multiclass", {})
    cic_mc_base = _resolve_baseline_task_dict(
        cic_mc_block, tables_dir, "cicids2017", "multiclass"
    )
    cic_mc_vars = cic_mc_block.get("variants", {})

    base_mac_m, base_mac_s = [], []
    p4_mac_m, p4_mac_s = [], []
    base_bot_m, base_bot_s = [], []
    p4_bot_m, p4_bot_s = [], []

    for m in MODEL_ORDER:
        bm = cic_mc_base.get(m, {})
        pm = _resolve_phase4_variant(cic_mc_vars, m)
        m1, s1 = _extract_mean_std(bm.get("macro_f1"))
        m2, s2 = _extract_mean_std(pm.get("macro_f1"))
        m3, s3 = _extract_mean_std(bm.get("per_class_f1", {}).get("Bot"))
        m4, s4 = _extract_mean_std(pm.get("per_class_f1", {}).get("Bot"))
        base_mac_m.append(m1)
        base_mac_s.append(s1)
        p4_mac_m.append(m2)
        p4_mac_s.append(s2)
        base_bot_m.append(m3)
        base_bot_s.append(s3)
        p4_bot_m.append(m4)
        p4_bot_s.append(s4)

    bm1 = ax_b.bar(
        x - 1.55 * w,
        base_mac_m,
        width=w,
        yerr=base_mac_s,
        capsize=3,
        color=COLOR_BASELINE,
        edgecolor="#333333",
        linewidth=0.7,
        label="Baseline Macro-F1",
    )
    bm2 = ax_b.bar(
        x - 0.50 * w,
        p4_mac_m,
        width=w,
        yerr=p4_mac_s,
        capsize=3,
        color=COLOR_PHASE4,
        edgecolor="#333333",
        linewidth=0.7,
        label="Phase 4 Macro-F1",
    )
    bm3 = ax_b.bar(
        x + 0.55 * w,
        base_bot_m,
        width=w,
        yerr=base_bot_s,
        capsize=3,
        color=COLOR_BASELINE_BOT,
        edgecolor="#333333",
        linewidth=0.7,
        hatch="..",
        label="Baseline Bot F1",
    )
    bm4 = ax_b.bar(
        x + 1.60 * w,
        p4_bot_m,
        width=w,
        yerr=p4_bot_s,
        capsize=3,
        color=COLOR_PHASE4_BOT,
        edgecolor="#333333",
        linewidth=0.7,
        hatch="..",
        label="Phase 4 Bot F1",
    )

    for bars, sds in (
        (bm1, base_mac_s),
        (bm2, p4_mac_s),
        (bm3, base_bot_s),
        (bm4, p4_bot_s),
    ):
        for bar, sd in zip(bars, sds):
            h = bar.get_height()
            if h > 0:
                ax_b.annotate(
                    f"{h:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2.0, h + sd),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.3,
                    fontweight="bold" if bars in (bm2, bm4) else "normal",
                )

    ax_b.set_ylim(0.0, 110.0)
    ax_b.set_title("(b) CICIDS2017 Multiclass: Macro-F1 & Bot F1", fontsize=11.5, fontweight="bold")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels, fontsize=9.5)
    ax_b.set_ylabel("F1 Score (%)", fontsize=10)
    ax_b.legend(loc="lower right", fontsize=7.6, ncol=2, framealpha=0.95)

    # ------------------------------------------------------------------
    # Panel (c): NF-UNSW-NB15-v2 Multiclass Macro-F1 (%)
    # ------------------------------------------------------------------
    ax_c = axes[2]
    nf_mc_block = datasets_block.get("nf_unsw", {}).get("multiclass", {})
    nf_mc_base = _resolve_baseline_task_dict(
        nf_mc_block, tables_dir, "nf_unsw", "multiclass"
    )
    nf_mc_vars = nf_mc_block.get("variants", {})

    nf_b_m, nf_b_s = [], []
    nf_p_m, nf_p_s = [], []
    for m in MODEL_ORDER:
        bm = nf_mc_base.get(m, {})
        pm = _resolve_phase4_variant(nf_mc_vars, m)
        m1, s1 = _extract_mean_std(bm.get("macro_f1"))
        m2, s2 = _extract_mean_std(pm.get("macro_f1"))
        nf_b_m.append(m1)
        nf_b_s.append(s1)
        nf_p_m.append(m2)
        nf_p_s.append(s2)

    w_c = 0.32
    bc1 = ax_c.bar(
        x - w_c / 2.0,
        nf_b_m,
        width=w_c,
        yerr=nf_b_s,
        capsize=4,
        color=COLOR_BASELINE,
        edgecolor="#333333",
        linewidth=0.7,
        label="Paper Baseline",
    )
    bc2 = ax_c.bar(
        x + w_c / 2.0,
        nf_p_m,
        width=w_c,
        yerr=nf_p_s,
        capsize=4,
        color=COLOR_PHASE4,
        edgecolor="#333333",
        linewidth=0.7,
        label="Phase 4 Improved",
    )

    for idx_m, (bar_b, bar_p) in enumerate(zip(bc1, bc2)):
        hb = bar_b.get_height()
        hp = bar_p.get_height()
        ax_c.annotate(
            f"{hb:.2f}%",
            xy=(bar_b.get_x() + bar_b.get_width() / 2.0, hb + nf_b_s[idx_m]),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.0,
        )
        delta = hp - hb
        sign = "+" if delta >= 0 else ""
        ax_c.annotate(
            f"{hp:.2f}%\n({sign}{delta:.2f})",
            xy=(bar_p.get_x() + bar_p.get_width() / 2.0, hp + nf_p_s[idx_m]),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color="#0B4F6C",
        )

    ymax_c = max([m + s for m, s in zip(nf_b_m + nf_p_m, nf_b_s + nf_p_s)] + [50.0])
    ax_c.set_ylim(0.0, min(105.0, ymax_c * 1.25))
    ax_c.set_title("(c) NF-UNSW-NB15-v2 Multiclass: Macro-F1", fontsize=11.5, fontweight="bold")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels, fontsize=9.5)
    ax_c.set_ylabel("Macro-F1 (%)", fontsize=10)
    ax_c.legend(loc="lower right", fontsize=8.5)

    fig.suptitle(
        "Phase 4 Main Benchmark: Paper Baseline vs. Leakage-Free Calibrated & Ratio-Engineered ML (Mean ± SD, 3 Seeds)",
        fontsize=12.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "phase4_fig1_main_comparison.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def plot_phase4_noise_robustness(
    tables_dir: Path,
    output_dir: Path,
) -> Path:
    """Generate `phase4_fig2_noise_robustness.png` (Binary F1 vs sigma, Baseline vs Phase 4)."""
    _apply_pub_style()
    comp = _load_json(tables_dir / "phase4_noise_comparison.json")

    sigmas = [float(s) for s in comp.get("sigmas", [0.0, 0.1, 0.25, 0.5])]
    p4_models = comp.get("models", {})
    base_models = comp.get("baseline", {})
    if not base_models:
        base_path = tables_dir / "noise_test_summary.json"
        if base_path.exists():
            base_models = _load_json(base_path).get("models", {})

    pairs = [
        ("xgboost", "xgboost_noise_robust", "XGBoost"),
        ("lightgbm", "lightgbm_noise_robust", "LightGBM"),
        ("random_forest", "random_forest_noise_calibrated", "Random Forest"),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=DPI)

    for base_key, p4_key, display_name in pairs:
        color = MODEL_COLORS[base_key]
        marker = MODEL_MARKERS[base_key]

        # Baseline (Table 12) dashed line with hollow markers
        if base_key in base_models:
            b_means, b_stds = [], []
            for s in sigmas:
                s_key = f"sigma_{s}"
                m, sd = _extract_mean_std(base_models[base_key].get(s_key, {}).get("f1"))
                b_means.append(m)
                b_stds.append(sd)
            ax.errorbar(
                sigmas,
                b_means,
                yerr=b_stds,
                color=color,
                marker=marker,
                markersize=7.0,
                markerfacecolor="white",
                markeredgewidth=1.6,
                linestyle="--",
                linewidth=1.8,
                capsize=4,
                alpha=0.80,
                label=f"{display_name} — Baseline (Table 12)",
            )

        # Phase 4 Noise-Robust solid line with filled markers
        if p4_key in p4_models:
            p_means, p_stds = [], []
            for s in sigmas:
                s_key = f"sigma_{s}"
                m, sd = _extract_mean_std(p4_models[p4_key].get(s_key, {}).get("f1"))
                p_means.append(m)
                p_stds.append(sd)
            ax.errorbar(
                sigmas,
                p_means,
                yerr=p_stds,
                color=color,
                marker=marker,
                markersize=7.5,
                linestyle="-",
                linewidth=2.4,
                capsize=4,
                label=f"{display_name} — Phase 4 Noise-Robust",
            )

    ax.set_title(
        "CICIDS2017 Gaussian Feature Noise Robustness: Baseline (Table 12) vs. Phase 4 (Mean ± SD)",
        fontsize=11.8,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel(r"Standardized Feature Noise Level ($\sigma$)", fontsize=10.8)
    ax.set_ylabel("Binary F1 Score (%)", fontsize=10.8)
    ax.set_xticks(sigmas)
    ax.set_xticklabels([f"{s:g}" for s in sigmas], fontsize=10)
    ax.set_ylim(-3.0, 105.0)
    ax.set_xlim(-0.02, max(sigmas) + 0.03)
    ax.legend(loc="best", fontsize=8.8, ncol=2)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "phase4_fig2_noise_robustness.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def plot_phase4_hybrid_lofo(
    tables_dir: Path,
    output_dir: Path,
) -> Path:
    """Generate `phase4_fig3_hybrid_lofo.png` (Unseen Recall + Seen Recall + Benign FPR)."""
    _apply_pub_style()
    comp = _load_json(tables_dir / "phase4_lofo_comparison.json")

    p4_dict = comp.get("models", {}).get("xgboost_hybrid_lofo", {})
    base_dict = comp.get("baseline", {}).get("xgboost", {})
    if not base_dict:
        base_path = tables_dir / "leave_one_out_summary.json"
        if base_path.exists():
            base_dict = _load_json(base_path).get("models", {}).get("xgboost", {})

    families = [f for f in LOFO_ORDER if f in p4_dict or f in base_dict]
    if not families:
        families = list(LOFO_ORDER)

    recall_labels = families + ["Overall\nSeen Recall"]
    base_rec_m, base_rec_s = [], []
    p4_rec_m, p4_rec_s = [], []

    for fam in families:
        bm, bs = _extract_mean_std(base_dict.get(fam, {}).get("held_out_recall"))
        pm, ps = _extract_mean_std(p4_dict.get(fam, {}).get("held_out_recall"))
        base_rec_m.append(bm)
        base_rec_s.append(bs)
        p4_rec_m.append(pm)
        p4_rec_s.append(ps)

    # Append Overall Seen Attack Recall
    bs_m, bs_s = _extract_mean_std(base_dict.get("overall_seen_attack_recall"))
    ps_m, ps_s = _extract_mean_std(p4_dict.get("overall_seen_attack_recall"))
    base_rec_m.append(bs_m)
    base_rec_s.append(bs_s)
    p4_rec_m.append(ps_m)
    p4_rec_s.append(ps_s)

    # Overall Benign FPR
    bfpr_m, bfpr_s = _extract_mean_std(base_dict.get("overall_benign_fpr"))
    pfpr_m, pfpr_s = _extract_mean_std(p4_dict.get("overall_benign_fpr"))

    fig, (ax_rec, ax_fpr) = plt.subplots(
        1,
        2,
        figsize=(13.6, 5.4),
        dpi=DPI,
        gridspec_kw={"width_ratios": [4.4, 1.25]},
    )

    x = np.arange(len(recall_labels))
    w = 0.36
    b_base = ax_rec.bar(
        x - w / 2.0,
        base_rec_m,
        width=w,
        yerr=base_rec_s,
        capsize=3.5,
        color=COLOR_BASELINE,
        edgecolor="#333333",
        linewidth=0.7,
        label="Baseline XGBoost (Table 11)",
    )
    b_p4 = ax_rec.bar(
        x + w / 2.0,
        p4_rec_m,
        width=w,
        yerr=p4_rec_s,
        capsize=3.5,
        color=COLOR_PHASE4,
        edgecolor="#333333",
        linewidth=0.7,
        label="Phase 4 Hybrid XGBoost + Benign Anomaly",
    )

    # Subtle separator between withheld families and Overall Seen Recall
    ax_rec.axvline(len(families) - 0.5, color="#666666", linestyle=":", linewidth=1.2)

    for idx_i, (bar_b, bar_p) in enumerate(zip(b_base, b_p4)):
        hb = bar_b.get_height()
        hp = bar_p.get_height()
        ax_rec.annotate(
            f"{hb:.1f}%",
            xy=(bar_b.get_x() + bar_b.get_width() / 2.0, hb + base_rec_s[idx_i]),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.6,
        )
        ax_rec.annotate(
            f"{hp:.1f}%",
            xy=(bar_p.get_x() + bar_p.get_width() / 2.0, hp + p4_rec_s[idx_i]),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
            color="#0B4F6C",
        )

    ax_rec.set_ylim(0.0, 112.0)
    ax_rec.set_xticks(x)
    ax_rec.set_xticklabels(recall_labels, fontsize=9.5)
    ax_rec.set_ylabel("Recall (%)", fontsize=10.5)
    ax_rec.set_title(
        "Unseen Attack Family Recall & Overall Seen Recall",
        fontsize=11.2,
        fontweight="bold",
    )
    ax_rec.legend(loc="upper left", fontsize=8.8)

    # Right panel: Overall Benign FPR (%)
    x_fpr = np.array([0.0, 0.55])
    bars_fpr = ax_fpr.bar(
        x_fpr,
        [bfpr_m, pfpr_m],
        width=0.36,
        yerr=[bfpr_s, pfpr_s],
        capsize=4,
        color=[COLOR_BASELINE_ALT, COLOR_PHASE4_ALT],
        edgecolor="#333333",
        linewidth=0.7,
    )
    for idx_f, (bar_f, sd_f) in enumerate(zip(bars_fpr, [bfpr_s, pfpr_s])):
        hf = bar_f.get_height()
        ax_fpr.annotate(
            f"{hf:.3f}%",
            xy=(bar_f.get_x() + bar_f.get_width() / 2.0, hf + sd_f),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold" if idx_f == 1 else "normal",
        )

    ax_fpr.axhline(
        0.25,
        color="#555555",
        linestyle="--",
        linewidth=1.0,
        label="0.25% FPR Ref.",
    )
    fpr_top = max(0.35, (max(bfpr_m + bfpr_s, pfpr_m + pfpr_s)) * 1.45)
    ax_fpr.set_ylim(0.0, fpr_top)
    ax_fpr.set_xticks(x_fpr)
    ax_fpr.set_xticklabels(["Baseline\nXGBoost", "Phase 4\nHybrid"], fontsize=9.2)
    ax_fpr.set_ylabel("Overall Benign FPR (%)", fontsize=10.2)
    ax_fpr.set_title("Benign FPR Control", fontsize=11.2, fontweight="bold")
    ax_fpr.legend(loc="upper right", fontsize=8.0)

    fig.suptitle(
        "CICIDS2017 Leave-One-Family-Out (Zero-Day): Baseline XGBoost vs. Phase 4 Hybrid Detector (Mean ± SD)",
        fontsize=12.2,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "phase4_fig3_hybrid_lofo.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)
    return out_path


def generate_all_phase4_figures(
    tables_dir: Path | str = Path("results/tables"),
    output_dir: Path | str = Path("results/figures"),
) -> dict[str, Path]:
    """Generate all 3 Phase 4 comparison figures at 300 DPI."""
    t_dir = Path(tables_dir)
    o_dir = Path(output_dir)
    o_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "phase4_fig1_main_comparison": plot_phase4_main_comparison(t_dir, o_dir),
        "phase4_fig2_noise_robustness": plot_phase4_noise_robustness(t_dir, o_dir),
        "phase4_fig3_hybrid_lofo": plot_phase4_hybrid_lofo(t_dir, o_dir),
    }
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Phase 4 comparison figures at 300 DPI."
    )
    parser.add_argument(
        "--tables-dir",
        type=str,
        default="results/tables",
        help="Directory containing Phase 4 comparison JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/figures",
        help="Directory to save generated 300 DPI PNG figures",
    )
    args = parser.parse_args()
    generate_all_phase4_figures(
        tables_dir=Path(args.tables_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
