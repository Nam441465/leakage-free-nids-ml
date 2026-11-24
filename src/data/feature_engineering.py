"""Stateless per-flow behavioral ratio feature engineering for NIDS tabular ML.

All engineered features are strictly stateless (computed independently per flow row)
so they introduce zero cross-sample or train/val/test data leakage. Intended to be
applied on raw float32 features AFTER dropping Destination Port / L4_DST_PORT and
BEFORE fitting LeakageFreePreprocessor strictly on the training split.
"""

from __future__ import annotations

from typing import Callable
import numpy as np
import pandas as pd


def _pos(arr: np.ndarray) -> np.ndarray:
    """Sanitize non-finite values and clamp negative sentinels (e.g. -1) to 0.0."""
    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(clean, 0.0).astype(np.float64, copy=False)


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Compute non-negative ratio num / (den + 1.0) safely in float32."""
    # den is already non-negative via _pos, so den + 1.0 >= 1.0 (never divides by zero).
    out = num / (den + 1.0)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(out, 0.0).astype(np.float32, copy=False)


def _safe_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute non-negative spread max(a - b, 0.0) safely in float32."""
    out = np.maximum(a - b, 0.0)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32, copy=False)


RatioSpec = tuple[str, tuple[str, ...], Callable[[dict[str, np.ndarray]], np.ndarray]]


NF_UNSW_RATIO_SPECS: tuple[RatioSpec, ...] = (
    (
        "RATIO_IN_OUT_BYTES",
        ("IN_BYTES", "OUT_BYTES"),
        lambda c: _safe_ratio(_pos(c["IN_BYTES"]), _pos(c["OUT_BYTES"])),
    ),
    (
        "RATIO_IN_OUT_PKTS",
        ("IN_PKTS", "OUT_PKTS"),
        lambda c: _safe_ratio(_pos(c["IN_PKTS"]), _pos(c["OUT_PKTS"])),
    ),
    (
        "BYTES_PER_PKT_IN",
        ("IN_BYTES", "IN_PKTS"),
        lambda c: _safe_ratio(_pos(c["IN_BYTES"]), _pos(c["IN_PKTS"])),
    ),
    (
        "BYTES_PER_PKT_OUT",
        ("OUT_BYTES", "OUT_PKTS"),
        lambda c: _safe_ratio(_pos(c["OUT_BYTES"]), _pos(c["OUT_PKTS"])),
    ),
    (
        "DURATION_ASYMMETRY",
        ("DURATION_IN", "DURATION_OUT"),
        lambda c: _safe_ratio(_pos(c["DURATION_IN"]), _pos(c["DURATION_OUT"])),
    ),
    (
        "THROUGHPUT_RATIO",
        ("SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT"),
        lambda c: _safe_ratio(
            _pos(c["SRC_TO_DST_AVG_THROUGHPUT"]),
            _pos(c["DST_TO_SRC_AVG_THROUGHPUT"]),
        ),
    ),
    (
        "RETRANS_RATIO_IN",
        ("RETRANSMITTED_IN_PKTS", "IN_PKTS"),
        lambda c: _safe_ratio(_pos(c["RETRANSMITTED_IN_PKTS"]), _pos(c["IN_PKTS"])),
    ),
    (
        "RETRANS_RATIO_OUT",
        ("RETRANSMITTED_OUT_PKTS", "OUT_PKTS"),
        lambda c: _safe_ratio(_pos(c["RETRANSMITTED_OUT_PKTS"]), _pos(c["OUT_PKTS"])),
    ),
    (
        "SMALL_PKT_SHARE",
        ("NUM_PKTS_UP_TO_128_BYTES", "IN_PKTS", "OUT_PKTS"),
        lambda c: _safe_ratio(
            _pos(c["NUM_PKTS_UP_TO_128_BYTES"]),
            _pos(c["IN_PKTS"]) + _pos(c["OUT_PKTS"]),
        ),
    ),
    (
        "LARGE_PKT_SHARE",
        ("NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES", "IN_PKTS", "OUT_PKTS"),
        lambda c: _safe_ratio(
            _pos(c["NUM_PKTS_512_TO_1024_BYTES"]) + _pos(c["NUM_PKTS_1024_TO_1514_BYTES"]),
            _pos(c["IN_PKTS"]) + _pos(c["OUT_PKTS"]),
        ),
    ),
    (
        "TTL_SPREAD",
        ("MAX_TTL", "MIN_TTL"),
        lambda c: _safe_diff(_pos(c["MAX_TTL"]), _pos(c["MIN_TTL"])),
    ),
    (
        "PKT_LEN_SPREAD_RATIO",
        ("LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT"),
        lambda c: _safe_ratio(
            np.maximum(_pos(c["LONGEST_FLOW_PKT"]) - _pos(c["SHORTEST_FLOW_PKT"]), 0.0),
            _pos(c["SHORTEST_FLOW_PKT"]),
        ),
    ),
)


CICIDS2017_RATIO_SPECS: tuple[RatioSpec, ...] = (
    (
        "RATIO_FWD_BWD_BYTES",
        ("Total Length of Fwd Packets", "Total Length of Bwd Packets"),
        lambda c: _safe_ratio(
            _pos(c["Total Length of Fwd Packets"]),
            _pos(c["Total Length of Bwd Packets"]),
        ),
    ),
    (
        "RATIO_FWD_BWD_PKTS",
        ("Total Fwd Packets", "Total Backward Packets"),
        lambda c: _safe_ratio(
            _pos(c["Total Fwd Packets"]),
            _pos(c["Total Backward Packets"]),
        ),
    ),
    (
        "PKT_SIZE_SPAN_RATIO",
        ("Max Packet Length", "Min Packet Length", "Average Packet Size"),
        lambda c: _safe_ratio(
            np.maximum(_pos(c["Max Packet Length"]) - _pos(c["Min Packet Length"]), 0.0),
            _pos(c["Average Packet Size"]),
        ),
    ),
    (
        "IAT_FWD_BWD_RATIO",
        ("Fwd IAT Mean", "Bwd IAT Mean"),
        lambda c: _safe_ratio(_pos(c["Fwd IAT Mean"]), _pos(c["Bwd IAT Mean"])),
    ),
    (
        "ACTIVE_IDLE_RATIO",
        ("Active Mean", "Idle Mean"),
        lambda c: _safe_ratio(_pos(c["Active Mean"]), _pos(c["Idle Mean"])),
    ),
    (
        "HEADER_PAYLOAD_RATIO_FWD",
        ("Fwd Header Length", "Total Length of Fwd Packets"),
        lambda c: _safe_ratio(
            _pos(c["Fwd Header Length"]),
            _pos(c["Total Length of Fwd Packets"]),
        ),
    ),
    (
        "SYN_TO_PKT_RATIO",
        ("SYN Flag Count", "Total Fwd Packets", "Total Backward Packets"),
        lambda c: _safe_ratio(
            _pos(c["SYN Flag Count"]),
            _pos(c["Total Fwd Packets"]) + _pos(c["Total Backward Packets"]),
        ),
    ),
    (
        "PSH_TO_PKT_RATIO",
        ("PSH Flag Count", "Total Fwd Packets", "Total Backward Packets"),
        lambda c: _safe_ratio(
            _pos(c["PSH Flag Count"]),
            _pos(c["Total Fwd Packets"]) + _pos(c["Total Backward Packets"]),
        ),
    ),
    (
        "ACK_TO_PKT_RATIO",
        ("ACK Flag Count", "Total Fwd Packets", "Total Backward Packets"),
        lambda c: _safe_ratio(
            _pos(c["ACK Flag Count"]),
            _pos(c["Total Fwd Packets"]) + _pos(c["Total Backward Packets"]),
        ),
    ),
    (
        "RST_FIN_TO_PKT_RATIO",
        ("RST Flag Count", "FIN Flag Count", "Total Fwd Packets", "Total Backward Packets"),
        lambda c: _safe_ratio(
            _pos(c["RST Flag Count"]) + _pos(c["FIN Flag Count"]),
            _pos(c["Total Fwd Packets"]) + _pos(c["Total Backward Packets"]),
        ),
    ),
    (
        "WIN_BYTES_RATIO",
        ("Init_Win_bytes_forward", "Init_Win_bytes_backward"),
        lambda c: _safe_ratio(
            _pos(c["Init_Win_bytes_forward"]),
            _pos(c["Init_Win_bytes_backward"]),
        ),
    ),
    (
        "ACT_DATA_FWD_SHARE",
        ("act_data_pkt_fwd", "Total Fwd Packets"),
        lambda c: _safe_ratio(
            _pos(c["act_data_pkt_fwd"]),
            _pos(c["Total Fwd Packets"]),
        ),
    ),
)


def _resolve_dataset_specs(dataset: str) -> tuple[RatioSpec, ...]:
    canonical = str(dataset).strip().lower().replace("-", "_")
    if canonical in {"nf_unsw", "nf_unsw_nb15_v2"}:
        return NF_UNSW_RATIO_SPECS
    if canonical in {"cicids2017", "cicids_2017"}:
        return CICIDS2017_RATIO_SPECS
    raise ValueError(
        f"Unsupported dataset '{dataset}'. Expected 'cicids2017', 'nf_unsw', or 'nf_unsw_nb15_v2'."
    )


def append_behavioral_ratios(
    X_or_df: np.ndarray | pd.DataFrame,
    feature_names: list[str],
    dataset: str,
) -> tuple[np.ndarray, list[str]]:
    """Append stateless per-flow behavioral ratio features in float32.

    Parameters:
        X_or_df: Raw 2D feature matrix or DataFrame (after dropping Destination Port /
            L4_DST_PORT, before LeakageFreePreprocessor.fit_transform).
        feature_names: Ordered feature names matching the columns of X_or_df.
        dataset: Dataset identifier ('cicids2017', 'nf_unsw', or 'nf_unsw_nb15_v2').

    Returns:
        (X_augmented_float32, augmented_feature_names)
    """
    specs = _resolve_dataset_specs(dataset)
    base_names = list(feature_names)

    if isinstance(X_or_df, pd.DataFrame):
        if len(base_names) > 0 and all(col in X_or_df.columns for col in base_names):
            X_base = X_or_df[base_names].to_numpy(dtype=np.float32, copy=False)
        else:
            X_base = X_or_df.to_numpy(dtype=np.float32, copy=False)
    else:
        X_base = np.asarray(X_or_df, dtype=np.float32)

    if X_base.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape {X_base.shape}.")
    if X_base.shape[1] != len(base_names):
        raise ValueError(
            f"Feature count mismatch: X has {X_base.shape[1]} columns, "
            f"but feature_names has {len(base_names)} entries."
        )

    # Map both exact and whitespace-stripped column names for robustness
    col_to_idx: dict[str, int] = {name: idx for idx, name in enumerate(base_names)}
    stripped_to_idx: dict[str, int] = {
        name.strip(): idx for idx, name in enumerate(base_names)
    }

    new_cols: list[np.ndarray] = []
    new_names: list[str] = []

    for ratio_name, required_cols, fn in specs:
        resolved_indices: list[int] = []
        missing = False
        for req in required_cols:
            if req in col_to_idx:
                resolved_indices.append(col_to_idx[req])
            elif req.strip() in stripped_to_idx:
                resolved_indices.append(stripped_to_idx[req.strip()])
            else:
                missing = True
                break
        if missing:
            continue

        col_dict = {
            req: X_base[:, idx] for req, idx in zip(required_cols, resolved_indices)
        }
        ratio_vec = fn(col_dict)
        new_cols.append(ratio_vec)
        new_names.append(ratio_name)

    if not new_cols:
        return np.ascontiguousarray(X_base, dtype=np.float32), base_names

    n_rows, n_base = X_base.shape
    n_new = len(new_cols)
    X_aug = np.empty((n_rows, n_base + n_new), dtype=np.float32)
    X_aug[:, :n_base] = X_base
    for offset, col_vec in enumerate(new_cols):
        X_aug[:, n_base + offset] = col_vec

    return X_aug, base_names + new_names
