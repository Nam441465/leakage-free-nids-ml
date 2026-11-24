"""Build compact real-test sample caches in `webapp/cache/` for sub-100ms live demos.

Reads strictly from `data/processed/*.parquet` and `saved_models/` (read-only) and
writes ONLY into `webapp/cache/`. Never modifies any project source, data, or model file.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.feature_engineering import append_behavioral_ratios
from src.experiments.run_phase4_main import (
    _extract_labels_and_mappings,
    _reconstruct_split_indices,
)
from src.models.calibration import apply_multiclass_priors, predict_binary_with_threshold
from src.models.persistence import load_model_bundle
from src.models.tree_baselines import predict_baseline_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = Path(__file__).resolve().parent / "cache"


def _build_dataset_cache(dataset: str, parquet_name: str, drop_port_col: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = PROJECT_ROOT / "data" / "processed" / parquet_name
    print(f"[{dataset}] Loading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)
    y_multi, y_bin, label_map, inv_map, label_cols = _extract_labels_and_mappings(df)
    _, _, test_idx = _reconstruct_split_indices(y_multi, seed=0)

    feature_cols = [
        c for c in df.columns if c not in (set(label_cols) | {drop_port_col})
    ]
    df_test = df.iloc[test_idx].reset_index(drop=True)
    ym_test = y_multi[test_idx]
    yb_test = y_bin[test_idx]
    del df
    gc.collect()

    rng = np.random.default_rng(42)

    # 1. Stratified 10,000-row sample preserving exact real test class proportions (ideal for Noise Lab / Stream)
    if dataset == "cicids2017":
        strat_idx, _ = train_test_split(
            np.arange(len(df_test)),
            train_size=10_000,
            random_state=42,
            stratify=ym_test,
        )
        strat_df = df_test.iloc[np.sort(strat_idx)].reset_index(drop=True)
        strat_df.to_parquet(CACHE_DIR / "cicids2017_noise_test_10k.parquet", index=False)
        print(f"  Saved cicids2017_noise_test_10k.parquet ({len(strat_df)} rows)")

    # 2. Balanced-per-class real test sample + curated "Phase 4 fixed Baseline error" cases
    chosen_set: set[int] = set()
    per_class_quota = 400 if dataset == "cicids2017" else 350
    benign_quota = 4000

    for cid, cname in sorted(inv_map.items()):
        c_idx = np.flatnonzero(ym_test == cid)
        quota = benign_quota if cname.lower() == "benign" else per_class_quota
        if len(c_idx) <= quota:
            picked = c_idx
        else:
            picked = rng.choice(c_idx, size=quota, replace=False)
        chosen_set.update(int(i) for i in picked)

    # Find real test cases where Baseline XGBoost was WRONG and Phase 4 Improved XGBoost was RIGHT
    fixed_meta: dict[str, list[int]] = {"binary_fixed": [], "multiclass_fixed": []}
    X_test_raw = df_test[feature_cols].to_numpy(dtype=np.float32, copy=True)

    for task in ["binary", "multiclass"]:
        base_b = load_model_bundle(
            PROJECT_ROOT / "saved_models", dataset, task, 0, "xgboost"
        )
        X_base = base_b["preprocessor"].transform(X_test_raw)
        y_pred_base, _ = predict_baseline_model(base_b["model"], X_base, task=task)
        del X_base, base_b
        gc.collect()

        # Load Phase 4 ratio + calibration model (seed 0)
        p4_task = f"{task}_ratio"
        p4_b = load_model_bundle(
            PROJECT_ROOT / "saved_models" / "phase4_improved",
            dataset,
            p4_task,
            0,
            "xgboost",
        )
        calib_path = (
            PROJECT_ROOT
            / "saved_models"
            / "phase4_improved"
            / dataset
            / p4_task
            / "seed_0"
            / "calibration_xgboost.json"
        )
        calib = json.loads(calib_path.read_text(encoding="utf-8"))
        cal_inner = calib.get("calibration", calib)
        X_aug, _ = append_behavioral_ratios(X_test_raw, feature_cols, dataset)
        X_p4 = p4_b["preprocessor"].transform(X_aug)
        _, prob_p4 = predict_baseline_model(p4_b["model"], X_p4, task=task)
        if task == "binary":
            y_pred_p4 = predict_binary_with_threshold(prob_p4, cal_inner["best_threshold"])
            y_true = yb_test
        else:
            gamma = np.asarray(cal_inner["gamma"], dtype=np.float64)
            y_pred_p4, _ = apply_multiclass_priors(prob_p4, gamma)
            y_true = ym_test
        del X_aug, X_p4, p4_b
        gc.collect()

        fixed_idx = np.flatnonzero((y_pred_base != y_true) & (y_pred_p4 == y_true))
        if len(fixed_idx) > 120:
            fixed_idx = rng.choice(fixed_idx, size=120, replace=False)
        chosen_set.update(int(i) for i in fixed_idx)
        fixed_meta[f"{task}_fixed_orig"] = [int(i) for i in sorted(fixed_idx)]
        print(f"  [{dataset}/{task}] Found {len(fixed_idx)} real test flows fixed by Phase 4!")

    ordered_idx = sorted(chosen_set)
    pos_map = {orig_i: new_i for new_i, orig_i in enumerate(ordered_idx)}
    sample_df = df_test.iloc[ordered_idx].reset_index(drop=True)
    sample_df.to_parquet(CACHE_DIR / f"{dataset}_test_sample.parquet", index=False)

    meta_out = {
        "dataset": dataset,
        "seed": 0,
        "n_rows": len(sample_df),
        "feature_cols": feature_cols,
        "label_mapping": label_map,
        "inverse_label_mapping": {str(k): v for k, v in inv_map.items()},
        "binary_fixed_rows": [pos_map[i] for i in fixed_meta.get("binary_fixed_orig", [])],
        "multiclass_fixed_rows": [pos_map[i] for i in fixed_meta.get("multiclass_fixed_orig", [])],
    }
    (CACHE_DIR / f"{dataset}_test_sample_meta.json").write_text(
        json.dumps(meta_out, indent=2), encoding="utf-8"
    )
    print(f"  Saved {dataset}_test_sample.parquet ({len(sample_df)} rows)")

    # 3. Export a 50-row sample CSV for instant Upload CSV demo (preserving raw feature columns + ground truth label)
    csv_indices = []
    for cid in sorted(inv_map.keys()):
        c_idx = np.flatnonzero(sample_df["label_multiclass"].to_numpy() == inv_map[cid]) if not pd.api.types.is_numeric_dtype(sample_df["label_multiclass"]) else np.flatnonzero(sample_df["label_multiclass"].to_numpy() == cid)
        if len(c_idx) > 0:
            k = min(len(c_idx), 12 if inv_map[cid].lower() == "benign" else 6)
            csv_indices.extend(rng.choice(c_idx, size=k, replace=False).tolist())
    csv_indices = sorted(set(csv_indices))[:50]
    sample_df.iloc[csv_indices].to_csv(CACHE_DIR / f"{dataset}_sample_upload.csv", index=False)
    print(f"  Saved {dataset}_sample_upload.csv ({len(csv_indices)} rows)")

    del df_test, sample_df, X_test_raw
    gc.collect()


def main() -> None:
    _build_dataset_cache("nf_unsw", "nf_unsw_nb15_v2.parquet", "L4_DST_PORT")
    _build_dataset_cache("cicids2017", "cicids2017.parquet", "Destination Port")
    print("All webapp demo sample caches built successfully!")


if __name__ == "__main__":
    main()
