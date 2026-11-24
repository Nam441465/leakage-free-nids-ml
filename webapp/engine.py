"""Read-only inference, stress-test, and verification engine for the NIDS Web Demo.

Strictly loads existing saved bundles from `saved_models/` and `results/tables/`
without modifying any underlying project logic, data, or artifacts.
"""

from __future__ import annotations

from collections import OrderedDict
import gc
import json
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.feature_engineering import append_behavioral_ratios
from src.evaluation.metrics import evaluate_binary_metrics, evaluate_multiclass_metrics
from src.experiments.demo_evaluate import evaluate_saved_model
from src.models.calibration import apply_multiclass_priors, predict_binary_with_threshold
from src.models.hybrid_lofo import predict_hybrid_lofo
from src.models.persistence import load_model_bundle
from src.models.tree_baselines import predict_baseline_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = Path(__file__).resolve().parent
CACHE_DIR = WEBAPP_DIR / "cache"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
MODELS_DIR = PROJECT_ROOT / "saved_models"
P4_MODELS_DIR = MODELS_DIR / "phase4_improved"

# Exact published reference numbers from 2611.04660v1.pdf (Tables 5, 6, 8, 9, 11, 12)
PAPER_REFERENCE: dict[str, Any] = {
    "cicids2017": {
        "binary": {
            "xgboost": {"f1": "99.67 ± 0.02", "mcc": "0.996 ± 0.000", "pr_auc": "99.99 ± 0.00", "fpr_at_0.5": "0.12 ± 0.01", "recall_at_0.1_fpr": "99.88 ± 0.01", "recall_at_1.0_fpr": "99.99 ± 0.00"},
            "lightgbm": {"f1": "99.66 ± 0.01", "mcc": "0.996 ± 0.000", "pr_auc": "99.98 ± 0.01", "fpr_at_0.5": "0.13 ± 0.00", "recall_at_0.1_fpr": "99.85 ± 0.01", "recall_at_1.0_fpr": "99.99 ± 0.00"},
            "random_forest": {"f1": "99.59 ± 0.01", "mcc": "0.995 ± 0.000", "pr_auc": "99.97 ± 0.00", "fpr_at_0.5": "0.08 ± 0.01", "recall_at_0.1_fpr": "99.70 ± 0.01", "recall_at_1.0_fpr": "99.98 ± 0.00"},
            "r_gcn_ref": {"f1": "98.11 ± 0.18", "pr_auc": "99.86 ± 0.01", "fpr_at_0.5": "0.64 ± 0.07"},
        },
        "multiclass": {
            "xgboost": {"macro_f1": "93.08 ± 1.22", "weighted_f1": "99.85 ± 0.01", "accuracy": "99.84 ± 0.01", "bot_f1": "71.30", "web_f1": "97.64"},
            "lightgbm": {"macro_f1": "81.57 ± 0.75", "weighted_f1": "99.66 ± 0.01", "accuracy": "99.58 ± 0.04", "bot_f1": "56.00", "web_f1": "94.00"},
            "random_forest": {"macro_f1": "88.37 ± 2.11", "weighted_f1": "99.85 ± 0.01", "accuracy": "99.85 ± 0.01", "bot_f1": "75.00", "web_f1": "97.00"},
            "r_gcn_ref": {"macro_f1": "64.51 ± 0.23", "weighted_f1": "97.80 ± 0.05", "accuracy": "96.60 ± 0.04"},
        },
    },
    "nf_unsw": {
        "binary": {
            "xgboost": {"f1": "91.92 ± 0.20", "mcc": "0.917 ± 0.002", "pr_auc": "96.94 ± 0.13", "fpr_at_0.5": "1.04 ± 0.03", "recall_at_0.1_fpr": "56.95 ± 3.09", "recall_at_1.0_fpr": "99.66 ± 0.15"},
            "lightgbm": {"f1": "91.79 ± 0.23", "mcc": "0.916 ± 0.002", "pr_auc": "96.81 ± 0.05", "fpr_at_0.5": "1.06 ± 0.03", "recall_at_0.1_fpr": "55.72 ± 3.04", "recall_at_1.0_fpr": "99.63 ± 0.13"},
            "random_forest": {"f1": "93.38 ± 0.18", "mcc": "0.930 ± 0.002", "pr_auc": "96.42 ± 0.13", "fpr_at_0.5": "0.51 ± 0.02", "recall_at_0.1_fpr": "50.66 ± 4.37", "recall_at_1.0_fpr": "99.34 ± 0.12"},
            "r_gcn_ref": {"f1": "90.57 ± 0.38", "pr_auc": "95.11 ± 0.14", "fpr_at_0.5": "1.22 ± 0.06"},
        },
        "multiclass": {
            "xgboost": {"macro_f1": "55.98 ± 0.88", "weighted_f1": "97.90 ± 0.06", "accuracy": "97.48 ± 0.08"},
            "lightgbm": {"macro_f1": "53.82 ± 1.04", "weighted_f1": "97.83 ± 0.05", "accuracy": "97.39 ± 0.07"},
            "random_forest": {"macro_f1": "— (chưa báo cáo)", "weighted_f1": "—", "accuracy": "—"},
            "r_gcn_ref": {"macro_f1": "44.21 ± 1.07", "weighted_f1": "97.04 ± 0.11", "accuracy": "96.25 ± 0.21"},
        },
    },
    "noise_table12_xgboost": {"0.0": "99.69 ± 0.04", "0.1": "5.84 ± 2.51", "0.25": "1.84 ± 0.96", "0.5": "0.67 ± 0.40"},
    "noise_table12_rgcn_ref": {"0.0": "98.26 ± 0.24", "0.1": "98.21 ± 0.21", "0.25": "97.91 ± 0.25", "0.5": "96.16 ± 0.29"},
    "lofo_table11_xgboost": {
        "DoS": "3.19 ± 0.07",
        "DDoS": "59.42 ± 5.47",
        "PortScan": "0.39 ± 0.11",
        "BruteForce": "0.26 ± 0.03",
        "Web": "83.36 ± 1.81",
        "Bot": "0.00 ± 0.00",
        "seen": "99.95 ± 0.02",
        "fpr": "0.10 ± 0.02",
    },
}

KEY_FEATURES_DISPLAY: dict[str, list[str]] = {
    "cicids2017": [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Fwd Packet Length Mean",
        "Bwd Packet Length Mean",
        "SYN Flag Count",
        "PSH Flag Count",
        "ACK Flag Count",
        "Init_Win_bytes_forward",
        "Active Mean",
    ],
    "nf_unsw": [
        "PROTOCOL",
        "L7_PROTO",
        "IN_BYTES",
        "OUT_BYTES",
        "IN_PKTS",
        "OUT_PKTS",
        "FLOW_DURATION_MILLISECONDS",
        "TCP_FLAGS",
        "MIN_TTL",
        "MAX_TTL",
        "LONGEST_FLOW_PKT",
        "SRC_TO_DST_AVG_THROUGHPUT",
        "DST_TO_SRC_AVG_THROUGHPUT",
        "RETRANSMITTED_IN_PKTS",
    ],
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class NIDSDemoEngine:
    """Memory-bounded inference engine with LRU bundle caching (max 6 bundles in RAM)."""

    def __init__(self, max_cached_bundles: int = 6) -> None:
        self.max_cached_bundles = max_cached_bundles
        self._bundles: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._sample_dfs: dict[str, pd.DataFrame] = {}
        self._sample_meta: dict[str, dict[str, Any]] = {}
        self._noise_df: pd.DataFrame | None = None
        self._rng = np.random.default_rng(2026)

    def _put_bundle(self, key: tuple[Any, ...], value: Any) -> Any:
        if key in self._bundles:
            self._bundles.move_to_end(key)
            return self._bundles[key]
        while len(self._bundles) >= self.max_cached_bundles:
            _, evicted = self._bundles.popitem(last=False)
            del evicted
            gc.collect()
        self._bundles[key] = value
        return value

    def _get_bundle(
        self, base_dir: Path, dataset: str, task: str, seed: int, model_name: str
    ) -> dict[str, Any]:
        key = (str(base_dir), dataset, task, int(seed), model_name)
        if key in self._bundles:
            self._bundles.move_to_end(key)
            return self._bundles[key]
        bundle = load_model_bundle(base_dir, dataset, task, seed, model_name)
        model = bundle["model"]
        # Force CPU prediction for sub-millisecond single/small-batch inference without CUDA device warnings
        if hasattr(model, "set_params") and "xgboost" in model_name:
            try:
                model.set_params(device="cpu")
            except Exception:
                pass
        return self._put_bundle(key, bundle)

    def _get_hybrid_detector(self, family: str, seed: int) -> dict[str, Any]:
        key = ("hybrid_lofo", family, int(seed))
        if key in self._bundles:
            self._bundles.move_to_end(key)
            return self._bundles[key]
        p = (
            P4_MODELS_DIR
            / "cicids2017"
            / "leave_one_out"
            / family
            / f"seed_{seed}"
            / "hybrid_detector.joblib"
        )
        obj = joblib.load(p)
        return self._put_bundle(key, obj)

    def _load_sample(self, dataset: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        if dataset not in self._sample_dfs:
            p = CACHE_DIR / f"{dataset}_test_sample.parquet"
            m = CACHE_DIR / f"{dataset}_test_sample_meta.json"
            if not p.exists() or not m.exists():
                raise FileNotFoundError(
                    f"Demo cache not found for '{dataset}'. Run `python -m webapp.prepare_demo_samples` first."
                )
            self._sample_dfs[dataset] = pd.read_parquet(p)
            self._sample_meta[dataset] = _read_json(m)
        return self._sample_dfs[dataset], self._sample_meta[dataset]

    def _load_noise_10k(self) -> pd.DataFrame:
        if self._noise_df is None:
            p = CACHE_DIR / "cicids2017_noise_test_10k.parquet"
            if not p.exists():
                raise FileNotFoundError(
                    "Noise demo cache not found. Run `python -m webapp.prepare_demo_samples` first."
                )
            self._noise_df = pd.read_parquet(p)
        return self._noise_df

    def get_overview(self) -> dict[str, Any]:
        """Return all verified JSON summaries, dataset stats, and figure catalog."""
        cic_meta = _read_json(PROJECT_ROOT / "data" / "processed" / "cicids2017.json")
        nf_meta = _read_json(PROJECT_ROOT / "data" / "processed" / "nf_unsw_nb15_v2_meta.json")

        base_counts = sum(
            1
            for sub in ["cicids2017", "nf_unsw"]
            for f in (MODELS_DIR / sub).rglob("*")
            if f.is_file()
        )
        p4_counts = sum(1 for f in P4_MODELS_DIR.rglob("*") if f.is_file())

        return {
            "kpis": {
                "cicids2017_rows": cic_meta["cleaned_rows"],
                "cicids2017_features": cic_meta["n_features"] - 1,
                "cicids2017_classes": len(cic_meta["class_distribution"]),
                "nf_unsw_rows": nf_meta["cleaned_rows"],
                "nf_unsw_features": nf_meta["n_features"] - 1,
                "nf_unsw_classes": len(nf_meta["class_distribution"]),
                "saved_baseline_files": base_counts,
                "saved_phase4_files": p4_counts,
                "total_saved_files": base_counts + p4_counts,
                "pytest_passed": 69,
            },
            "datasets_meta": {"cicids2017": cic_meta, "nf_unsw": nf_meta},
            "paper_reference": PAPER_REFERENCE,
            "baseline_tables": {
                "cicids2017_binary": _read_json(TABLES_DIR / "cicids2017_binary_summary.json"),
                "cicids2017_multiclass": _read_json(TABLES_DIR / "cicids2017_multiclass_summary.json"),
                "nf_unsw_binary": _read_json(TABLES_DIR / "nf_unsw_binary_summary.json"),
                "nf_unsw_multiclass": _read_json(TABLES_DIR / "nf_unsw_multiclass_summary.json"),
                "noise": _read_json(TABLES_DIR / "noise_test_summary.json"),
                "lofo": _read_json(TABLES_DIR / "leave_one_out_summary.json"),
            },
            "phase4_tables": {
                "main": _read_json(TABLES_DIR / "phase4_main_comparison.json"),
                "noise": _read_json(TABLES_DIR / "phase4_noise_comparison.json"),
                "lofo": _read_json(TABLES_DIR / "phase4_lofo_comparison.json"),
            },
            "figures": {
                "baseline": [
                    {"id": "fig4", "title": "Hình 4 — Binary F1 trên CICIDS2017 & NF-UNSW-NB15-v2", "url": "/figures/baseline/fig4_binary_f1.png"},
                    {"id": "fig6", "title": "Hình 6 — Per-class F1 Đa lớp (Bảng 2 & Bảng 3)", "url": "/figures/baseline/fig6_multiclass_per_class_f1.png"},
                    {"id": "fig7b", "title": "Hình 7b — Ma trận nhầm lẫn XGBoost (CICIDS2017)", "url": "/figures/baseline/fig7b_xgboost_cm_cicids2017.png"},
                    {"id": "fig9", "title": "Hình 9 — Leave-One-Attack-Family-Out Recall (Bảng 11)", "url": "/figures/baseline/fig9_leave_one_out_recall.png"},
                    {"id": "fig10", "title": "Hình 10 — Sụp đổ F1 dưới nhiễu Gaussian (Bảng 12)", "url": "/figures/baseline/fig10_gaussian_noise_f1.png"},
                ],
                "phase4": [
                    {"id": "p4_fig1", "title": "Phase 4 Hình 1 — Đối chiếu Main Benchmark (Binary & Multiclass)", "url": "/figures/phase4_improved/phase4_fig1_main_comparison.png"},
                    {"id": "p4_fig2", "title": "Phase 4 Hình 2 — Đột phá Kháng nhiễu Gaussian (96%–99% vs <6%)", "url": "/figures/phase4_improved/phase4_fig2_noise_robustness.png"},
                    {"id": "p4_fig3", "title": "Phase 4 Hình 3 — Đối chiếu Zero-Day Hybrid LOFO (DoS 5.3x, Web 87.7%)", "url": "/figures/phase4_improved/phase4_fig3_hybrid_lofo.png"},
                ],
            },
        }

    def _run_pair_inference(
        self,
        dataset: str,
        task: str,
        model_name: str,
        X_raw: np.ndarray,
        feature_cols: list[str],
        inv_map: dict[int, str],
    ) -> dict[str, Any]:
        """Run both Baseline and Phase 4 Improved models on raw float32 matrix X_raw."""
        # 1. Baseline inference (saved_models/<dataset>/<task>/seed_0/<model>.joblib)
        t0 = time.perf_counter()
        base_b = self._get_bundle(MODELS_DIR, dataset, task, 0, model_name)
        X_base = base_b["preprocessor"].transform(X_raw)
        y_pred_base, prob_base = predict_baseline_model(base_b["model"], X_base, task=task)
        base_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Phase 4 inference (Ratio + Tempered + Calibrated for XGB/LGB; Calibrated for RF)
        t1 = time.perf_counter()
        use_ratio = model_name in ("xgboost", "lightgbm")
        if use_ratio:
            p4_task = f"{task}_ratio"
            p4_b = self._get_bundle(P4_MODELS_DIR, dataset, p4_task, 0, model_name)
            calib_path = (
                P4_MODELS_DIR / dataset / p4_task / "seed_0" / f"calibration_{model_name}.json"
            )
            calib = _read_json(calib_path)
            X_aug, _ = append_behavioral_ratios(X_raw, feature_cols, dataset)
            X_p4 = p4_b["preprocessor"].transform(X_aug)
            _, prob_p4_raw = predict_baseline_model(p4_b["model"], X_p4, task=task)
            p4_variant = f"{model_name}_ratio_calibrated"
        else:
            p4_task = f"{task}_calibrated"
            calib_path = (
                P4_MODELS_DIR / dataset / p4_task / "seed_0" / f"calibration_{model_name}.json"
            )
            calib = _read_json(calib_path)
            prob_p4_raw = prob_base
            p4_variant = f"{model_name}_calibrated"

        cal_inner = calib.get("calibration", calib)
        if task == "binary":
            thresh = float(cal_inner["best_threshold"])
            y_pred_p4 = predict_binary_with_threshold(prob_p4_raw, thresh)
            prob_p4 = prob_p4_raw
            calib_info = {"threshold": round(thresh, 4), "baseline_threshold": 0.5}
        else:
            gamma = np.asarray(cal_inner["gamma"], dtype=np.float64)
            y_pred_p4, prob_p4 = apply_multiclass_priors(prob_p4_raw, gamma)
            calib_info = {"gamma_min": round(float(np.min(gamma)), 3), "gamma_max": round(float(np.max(gamma)), 3)}

        p4_ms = (time.perf_counter() - t1) * 1000.0

        return {
            "y_pred_base": y_pred_base,
            "prob_base": prob_base,
            "base_ms": base_ms,
            "y_pred_p4": y_pred_p4,
            "prob_p4": prob_p4,
            "p4_ms": p4_ms,
            "p4_variant": p4_variant,
            "calib_info": calib_info,
        }

    def predict_single_sample(
        self,
        dataset: str,
        task: str,
        model_name: str,
        class_filter: str = "ALL",
        only_phase4_fixed: bool = False,
    ) -> dict[str, Any]:
        df, meta = self._load_sample(dataset)
        feature_cols: list[str] = meta["feature_cols"]
        inv_map = {int(k): v for k, v in meta["inverse_label_mapping"].items()}
        label_map = meta["label_mapping"]

        raw_labels = df["label_multiclass"].to_numpy()
        if pd.api.types.is_numeric_dtype(df["label_multiclass"]):
            y_multi_names = np.array([inv_map[int(v)] for v in raw_labels])
        else:
            y_multi_names = raw_labels.astype(str)

        if only_phase4_fixed:
            key = f"{task}_fixed_rows"
            pool = meta.get(key, [])
            if not pool:
                pool = list(range(len(df)))
            row_idx = int(self._rng.choice(pool))
        elif class_filter and class_filter != "ALL":
            matches = np.flatnonzero(y_multi_names == class_filter)
            if len(matches) == 0:
                raise ValueError(f"No test flows found for class '{class_filter}'.")
            row_idx = int(self._rng.choice(matches))
        else:
            row_idx = int(self._rng.integers(0, len(df)))

        row = df.iloc[row_idx]
        X_raw = row[feature_cols].to_numpy(dtype=np.float32).reshape(1, -1)
        true_multi_name = str(y_multi_names[row_idx])
        true_bin_int = 0 if true_multi_name.lower() == "benign" else 1
        true_label_str = ("Benign" if true_bin_int == 0 else f"Attack ({true_multi_name})") if task == "binary" else true_multi_name
        true_class_id = true_bin_int if task == "binary" else int(label_map[true_multi_name])

        res = self._run_pair_inference(dataset, task, model_name, X_raw, feature_cols, inv_map)

        def _format_probs(prob_arr: np.ndarray) -> list[dict[str, Any]]:
            if task == "binary":
                p1 = float(prob_arr[0]) if prob_arr.ndim == 1 else float(prob_arr[0, 1])
                return [
                    {"class": "Benign", "prob": round((1.0 - p1) * 100.0, 2)},
                    {"class": "Attack", "prob": round(p1 * 100.0, 2)},
                ]
            row_p = prob_arr[0]
            items = [
                {"class": inv_map[i], "prob": round(float(row_p[i]) * 100.0, 2)}
                for i in range(len(row_p))
            ]
            items.sort(key=lambda x: x["prob"], reverse=True)
            return items

        base_pred_id = int(res["y_pred_base"][0])
        p4_pred_id = int(res["y_pred_p4"][0])
        base_pred_name = ("Attack" if base_pred_id == 1 else "Benign") if task == "binary" else inv_map[base_pred_id]
        p4_pred_name = ("Attack" if p4_pred_id == 1 else "Benign") if task == "binary" else inv_map[p4_pred_id]

        # Extract key human-readable raw features
        key_cols = [c for c in KEY_FEATURES_DISPLAY.get(dataset, []) if c in df.columns]
        raw_features = {c: round(float(row[c]), 3) for c in key_cols}

        return {
            "dataset": dataset,
            "task": task,
            "model": model_name,
            "sample_row_id": row_idx,
            "ground_truth": true_label_str,
            "ground_truth_family": true_multi_name,
            "ground_truth_is_attack": bool(true_bin_int == 1),
            "baseline": {
                "prediction": base_pred_name,
                "correct": bool(base_pred_id == true_class_id),
                "latency_ms": round(res["base_ms"], 2),
                "probabilities": _format_probs(res["prob_base"]),
            },
            "phase4": {
                "variant": res["p4_variant"],
                "prediction": p4_pred_name,
                "correct": bool(p4_pred_id == true_class_id),
                "latency_ms": round(res["p4_ms"], 2),
                "probabilities": _format_probs(res["prob_p4"]),
                "calibration": res["calib_info"],
            },
            "key_features": raw_features,
        }

    def predict_stream_batch(
        self, dataset: str, task: str, model_name: str, batch_size: int = 8
    ) -> dict[str, Any]:
        df, meta = self._load_sample(dataset)
        feature_cols: list[str] = meta["feature_cols"]
        inv_map = {int(k): v for k, v in meta["inverse_label_mapping"].items()}
        label_map = meta["label_mapping"]

        raw_labels = df["label_multiclass"].to_numpy()
        if pd.api.types.is_numeric_dtype(df["label_multiclass"]):
            y_multi_names = np.array([inv_map[int(v)] for v in raw_labels])
        else:
            y_multi_names = raw_labels.astype(str)

        ben_idx = np.flatnonzero(y_multi_names == "Benign")
        atk_idx = np.flatnonzero(y_multi_names != "Benign")
        n_atk = max(1, int(round(batch_size * 0.3)))
        n_ben = max(1, batch_size - n_atk)
        picked = np.concatenate([
            self._rng.choice(ben_idx, size=min(n_ben, len(ben_idx)), replace=False),
            self._rng.choice(atk_idx, size=min(n_atk, len(atk_idx)), replace=False),
        ])
        self._rng.shuffle(picked)

        sub = df.iloc[picked]
        X_raw = sub[feature_cols].to_numpy(dtype=np.float32, copy=True)
        res = self._run_pair_inference(dataset, task, model_name, X_raw, feature_cols, inv_map)

        events = []
        for i, orig_i in enumerate(picked):
            true_fam = str(y_multi_names[orig_i])
            true_bin = 0 if true_fam.lower() == "benign" else 1
            true_id = true_bin if task == "binary" else int(label_map[true_fam])
            bp = int(res["y_pred_base"][i])
            pp = int(res["y_pred_p4"][i])
            b_name = ("Attack" if bp == 1 else "Benign") if task == "binary" else inv_map[bp]
            p_name = ("Attack" if pp == 1 else "Benign") if task == "binary" else inv_map[pp]
            if task == "binary":
                p4_conf = float(res["prob_p4"][i]) if pp == 1 else (1.0 - float(res["prob_p4"][i]))
            else:
                p4_conf = float(res["prob_p4"][i, pp])

            dur_col = "Flow Duration" if dataset == "cicids2017" else "FLOW_DURATION_MILLISECONDS"
            pkt_col = "Total Fwd Packets" if dataset == "cicids2017" else "IN_PKTS"
            events.append({
                "flow_id": int(orig_i),
                "ground_truth": true_fam,
                "is_attack": bool(true_bin == 1),
                "baseline_pred": b_name,
                "baseline_correct": bool(bp == true_id),
                "phase4_pred": p_name,
                "phase4_correct": bool(pp == true_id),
                "phase4_conf": round(p4_conf * 100.0, 1),
                "duration": round(float(sub.iloc[i][dur_col]), 1) if dur_col in sub.columns else 0.0,
                "packets": int(sub.iloc[i][pkt_col]) if pkt_col in sub.columns else 0,
            })

        return {
            "batch_size": len(events),
            "baseline_ms": round(res["base_ms"], 2),
            "phase4_ms": round(res["p4_ms"], 2),
            "events": events,
        }

    def predict_csv_upload(
        self, dataset: str, task: str, model_name: str, df_upload: pd.DataFrame
    ) -> dict[str, Any]:
        _, meta = self._load_sample(dataset)
        feature_cols: list[str] = meta["feature_cols"]
        inv_map = {int(k): v for k, v in meta["inverse_label_mapping"].items()}
        label_map = meta["label_mapping"]

        matched = [c for c in feature_cols if c in df_upload.columns]
        if len(matched) < max(5, int(len(feature_cols) * 0.5)):
            raise ValueError(
                f"File CSV chỉ khớp {len(matched)}/{len(feature_cols)} cột đặc trưng của {dataset}. "
                f"Hãy tải file CSV mẫu của đúng bộ dữ liệu {dataset}!"
            )

        # Limit batch to 2,000 rows for instant web response
        df_use = df_upload.iloc[:2000].copy()
        X_mat = np.zeros((len(df_use), len(feature_cols)), dtype=np.float32)
        for j, col in enumerate(feature_cols):
            if col in df_use.columns:
                vals = pd.to_numeric(df_use[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
                X_mat[:, j] = vals.to_numpy(dtype=np.float32)

        res = self._run_pair_inference(dataset, task, model_name, X_mat, feature_cols, inv_map)

        has_truth = "label_multiclass" in df_use.columns
        rows_out = []
        base_correct_cnt = 0
        p4_correct_cnt = 0
        p4_attack_cnt = 0
        base_attack_cnt = 0

        for i in range(len(df_use)):
            bp = int(res["y_pred_base"][i])
            pp = int(res["y_pred_p4"][i])
            b_name = ("Attack" if bp == 1 else "Benign") if task == "binary" else inv_map[bp]
            p_name = ("Attack" if pp == 1 else "Benign") if task == "binary" else inv_map[pp]
            if b_name != "Benign":
                base_attack_cnt += 1
            if p_name != "Benign":
                p4_attack_cnt += 1

            gt_name = None
            b_ok = None
            p_ok = None
            if has_truth:
                raw_v = df_use.iloc[i]["label_multiclass"]
                gt_name = inv_map[int(raw_v)] if isinstance(raw_v, (int, np.integer)) else str(raw_v)
                gt_bin = 0 if gt_name.lower() == "benign" else 1
                gt_id = gt_bin if task == "binary" else label_map.get(gt_name, -1)
                b_ok = bool(bp == gt_id)
                p_ok = bool(pp == gt_id)
                if b_ok:
                    base_correct_cnt += 1
                if p_ok:
                    p4_correct_cnt += 1

            if i < 100:
                rows_out.append({
                    "row": i + 1,
                    "ground_truth": gt_name,
                    "baseline_pred": b_name,
                    "baseline_correct": b_ok,
                    "phase4_pred": p_name,
                    "phase4_correct": p_ok,
                })

        n = len(df_use)
        return {
            "total_rows": n,
            "matched_features": len(matched),
            "expected_features": len(feature_cols),
            "baseline_ms": round(res["base_ms"], 2),
            "phase4_ms": round(res["p4_ms"], 2),
            "baseline_attacks_flagged": base_attack_cnt,
            "phase4_attacks_flagged": p4_attack_cnt,
            "has_ground_truth": has_truth,
            "baseline_accuracy": round(base_correct_cnt / n * 100.0, 2) if has_truth and n > 0 else None,
            "phase4_accuracy": round(p4_correct_cnt / n * 100.0, 2) if has_truth and n > 0 else None,
            "preview_rows": rows_out,
        }

    def run_live_noise_lab(
        self, model_name: str, sigma: float, n_samples: int = 5000
    ) -> dict[str, Any]:
        """Run LIVE Gaussian feature noise stress test comparing Baseline vs Phase 4 Noise-Robust."""
        df = self._load_noise_10k()
        _, meta = self._load_sample("cicids2017")
        feature_cols: list[str] = meta["feature_cols"]
        n_use = min(max(int(n_samples), 500), len(df))
        sub = df.iloc[:n_use]

        X_raw = sub[feature_cols].to_numpy(dtype=np.float32, copy=True)
        raw_labels = sub["label_multiclass"].to_numpy()
        y_bin = np.array([0 if str(v).lower() == "benign" or v == 0 else 1 for v in raw_labels], dtype=np.int64)

        # Load baseline bundle (seed 0)
        base_b = self._get_bundle(MODELS_DIR, "cicids2017", "binary", 0, model_name)
        X_std = base_b["preprocessor"].transform(X_raw)

        sigma_f = float(np.clip(sigma, 0.0, 1.0))
        if sigma_f > 0.0:
            rng = np.random.RandomState(seed=1000 + int(round(sigma_f * 100)))
            noise = rng.normal(0.0, sigma_f, size=X_std.shape).astype(np.float32)
            X_noisy = np.clip(X_std + noise, -10.0, 10.0).astype(np.float32)
        else:
            X_noisy = X_std

        # 1. Baseline prediction (threshold = 0.5)
        t0 = time.perf_counter()
        _, prob_base = predict_baseline_model(base_b["model"], X_noisy, task="binary", threshold=0.5)
        m_base = evaluate_binary_metrics(y_bin, prob_base, threshold=0.5)
        base_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Phase 4 Noise-Robust prediction
        t1 = time.perf_counter()
        if model_name in ("xgboost", "lightgbm"):
            p4_name = f"{model_name}_noise_robust"
            p4_b = self._get_bundle(P4_MODELS_DIR, "cicids2017", "noise_robust", 0, p4_name)
            calib = _read_json(
                P4_MODELS_DIR / "cicids2017" / "noise_robust" / "seed_0" / f"calibration_{p4_name}.json"
            )
            p4_thresh = float(calib.get("calibration", calib)["best_threshold"])
            _, prob_p4 = predict_baseline_model(p4_b["model"], X_noisy, task="binary", threshold=p4_thresh)
        else:
            p4_name = "random_forest_noise_calibrated"
            calib = _read_json(
                P4_MODELS_DIR / "cicids2017" / "noise_robust" / "seed_0" / f"calibration_{p4_name}.json"
            )
            p4_thresh = float(calib.get("calibration", calib)["best_threshold"])
            prob_p4 = prob_base

        m_p4 = evaluate_binary_metrics(y_bin, prob_p4, threshold=p4_thresh)
        p4_ms = (time.perf_counter() - t1) * 1000.0

        def _pct(val: float) -> float:
            v = float(val)
            return v * 100.0 if v <= 1.00001 else v

        return {
            "model": model_name,
            "p4_model": p4_name,
            "sigma": sigma_f,
            "n_samples": n_use,
            "n_attacks": int(np.sum(y_bin == 1)),
            "n_benign": int(np.sum(y_bin == 0)),
            "baseline": {
                "threshold": 0.5,
                "latency_ms": round(base_ms, 1),
                "f1": round(_pct(m_base["f1"]), 2),
                "precision": round(_pct(m_base["precision"]), 2),
                "recall": round(_pct(m_base["recall"]), 2),
                "pr_auc": round(_pct(m_base["pr_auc"]), 2),
                "fpr": round(_pct(m_base["fpr_at_0.5"]), 3),
            },
            "phase4": {
                "threshold": round(p4_thresh, 4),
                "latency_ms": round(p4_ms, 1),
                "f1": round(_pct(m_p4["f1"]), 2),
                "precision": round(_pct(m_p4["precision"]), 2),
                "recall": round(_pct(m_p4["recall"]), 2),
                "pr_auc": round(_pct(m_p4["pr_auc"]), 2),
                "fpr": round(_pct(m_p4["fpr_at_0.5"]), 3),
            },
        }

    def run_live_lofo_lab(self, family: str, seed: int = 0) -> dict[str, Any]:
        """Run LIVE Leave-One-Family-Out Zero-Day inference on cached real test flows."""
        df, meta = self._load_sample("cicids2017")
        feature_cols: list[str] = meta["feature_cols"]
        inv_map = {int(k): v for k, v in meta["inverse_label_mapping"].items()}

        raw_labels = df["label_multiclass"].to_numpy()
        if pd.api.types.is_numeric_dtype(df["label_multiclass"]):
            y_names = np.array([inv_map[int(v)] for v in raw_labels])
        else:
            y_names = raw_labels.astype(str)

        X_raw = df[feature_cols].to_numpy(dtype=np.float32, copy=True)

        t0 = time.perf_counter()
        lofo_b = self._get_bundle(MODELS_DIR, "cicids2017", f"leave_one_out/{family}", seed, "xgboost")
        X_std = lofo_b["preprocessor"].transform(X_raw)
        y_pred_base, prob_xgb = predict_baseline_model(lofo_b["model"], X_std, task="binary", threshold=0.5)
        base_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        hybrid_obj = self._get_hybrid_detector(family, seed)
        scorer = hybrid_obj["anomaly_scorer"]
        calib = hybrid_obj["calib_params"]
        anom_raw = scorer.score_raw(X_std)
        y_pred_hyb, _ = predict_hybrid_lofo(prob_xgb, anom_raw, calib)
        hyb_ms = (time.perf_counter() - t1) * 1000.0

        held_mask = y_names == family
        ben_mask = y_names == "Benign"
        seen_mask = (~held_mask) & (~ben_mask)

        # Also read exact full-test 504,160-flow metrics from saved CSV for complete transparency
        full_p4_df = pd.read_csv(TABLES_DIR / "phase4_lofo_detailed.csv")
        full_base_df = pd.read_csv(TABLES_DIR / "leave_one_out_detailed.csv")
        row_p4 = full_p4_df[(full_p4_df["held_out_family"] == family) & (full_p4_df["seed"] == seed)].iloc[0]
        row_base = full_base_df[(full_base_df["held_out_family"] == family) & (full_base_df["seed"] == seed)].iloc[0]

        def _lofo_pct(v: float) -> float:
            vf = float(v)
            return vf * 100.0 if vf <= 1.00001 else vf

        return {
            "family": family,
            "seed": int(seed),
            "sample_counts": {
                "held_out": int(np.sum(held_mask)),
                "seen_attacks": int(np.sum(seen_mask)),
                "benign": int(np.sum(ben_mask)),
            },
            "live_sample_metrics": {
                "baseline_ms": round(base_ms, 1),
                "hybrid_ms": round(hyb_ms, 1),
                "baseline_unseen_recall": round(float(np.mean(y_pred_base[held_mask] == 1) * 100.0), 2),
                "hybrid_unseen_recall": round(float(np.mean(y_pred_hyb[held_mask] == 1) * 100.0), 2),
                "baseline_seen_recall": round(float(np.mean(y_pred_base[seen_mask] == 1) * 100.0), 2),
                "hybrid_seen_recall": round(float(np.mean(y_pred_hyb[seen_mask] == 1) * 100.0), 2),
                "baseline_benign_fpr": round(float(np.mean(y_pred_base[ben_mask] == 1) * 100.0), 3),
                "hybrid_benign_fpr": round(float(np.mean(y_pred_hyb[ben_mask] == 1) * 100.0), 3),
                "xgb_thresh": round(float(calib["xgb_thresh"]), 5),
                "anom_thresh": round(float(calib["anom_thresh"]), 3),
            },
            "full_test_504k_metrics": {
                "held_out_flows": int(row_p4["n_held_test"]),
                "baseline_unseen_recall": round(_lofo_pct(row_base["held_out_recall"]), 2),
                "hybrid_unseen_recall": round(_lofo_pct(row_p4["held_out_recall"]), 2),
                "baseline_seen_recall": round(_lofo_pct(row_base["seen_attack_recall"]), 2),
                "hybrid_seen_recall": round(_lofo_pct(row_p4["seen_attack_recall"]), 2),
                "baseline_benign_fpr": round(_lofo_pct(row_base["benign_fpr"]), 3),
                "hybrid_benign_fpr": round(_lofo_pct(row_p4["benign_fpr"]), 3),
            },
        }

    def run_full_audit(
        self, dataset: str, task: str, model_name: str, seed: int
    ) -> dict[str, Any]:
        """Run full live test evaluation (146k / 504k rows) via existing read-only `demo_evaluate`."""
        result = evaluate_saved_model(
            dataset=dataset,
            task=task,
            model_name=model_name,
            seed=int(seed),
            models_dir=MODELS_DIR,
            data_dir=PROJECT_ROOT / "data" / "processed",
            results_dir=TABLES_DIR,
        )
        # Build full multi-metric comparison against <dataset>_detailed_results.csv
        csv_path = TABLES_DIR / f"{dataset}_detailed_results.csv"
        comparison: dict[str, dict[str, float]] = {}
        max_abs_diff = 0.0
        live_m = result.get("metrics", {})
        if csv_path.exists():
            cdf = pd.read_csv(csv_path)
            match = cdf[
                (cdf["dataset"] == dataset)
                & (cdf["task"] == task)
                & (cdf["model"] == model_name)
                & (cdf["seed"] == int(seed))
            ]
            if not match.empty:
                row = match.iloc[0]
                metric_keys = (
                    ["f1", "mcc", "pr_auc", "fpr_at_0.5", "recall_at_0.1_fpr", "recall_at_1.0_fpr"]
                    if task == "binary"
                    else ["macro_f1", "weighted_f1", "accuracy"]
                )
                for mk in metric_keys:
                    if mk in live_m and mk in row and not pd.isna(row[mk]):
                        rep_v = float(row[mk])
                        liv_raw = float(live_m[mk])
                        # Align scale if CSV is stored in % (0-100) and live_m is [0-1]
                        if mk != "mcc" and rep_v > 1.5 and liv_raw <= 1.00001:
                            liv_v = liv_raw * 100.0
                        else:
                            liv_v = liv_raw
                        diff = abs(liv_v - rep_v)
                        max_abs_diff = max(max_abs_diff, diff)
                        comparison[mk] = {
                            "live": liv_v,
                            "reported": rep_v,
                            "abs_diff": diff,
                        }
        if not comparison and result.get("reported_comparison"):
            rc = result["reported_comparison"]
            max_abs_diff = float(rc["abs_diff"])
            comparison[rc["metric"]] = {
                "live": float(rc["live_value"]),
                "reported": float(rc["reported_value"]),
                "abs_diff": float(rc["abs_diff"]),
            }

        gc.collect()
        return {
            "dataset": dataset,
            "task": task,
            "model": model_name,
            "seed": int(seed),
            "n_test_samples": int(result["n_test_samples"]),
            "inference_time_s": float(result["inference_time_sec"]),
            "model_size_mb": float(result["model_size_mb"]),
            "max_abs_diff": float(max_abs_diff),
            "all_within_tolerance": bool(max_abs_diff < 1e-4),
            "comparison": comparison,
        }
