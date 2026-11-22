"""Unit tests for NF-UNSW-NB15-v2 data cleaning pipeline.

Validates the leakage-free cleaning protocol:
1. Identifier removal (L4_SRC_PORT, DNS_QUERY_ID, IPV4_SRC_ADDR, IPV4_DST_ADDR).
2. Rare class merging with configurable threshold and special handling of Worms.
3. Label-aware exact de-duplication (keeping identical features with differing labels).
4. Constant column detection and removal.
5. Inf / NaN handling and float32 dtype enforcement.
6. End-to-end CSV -> Parquet + JSON metadata pipeline.
7. Seamless interoperability with split_preprocess.py.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.clean_nf_unsw import (
    DESTINATION_PORT_COL,
    IDENTIFIER_COLUMNS,
    TARGET_FAMILIES,
    clean_nf_unsw_dataframe,
    clean_nf_unsw_dataset,
)
from src.data.split_preprocess import prepare_splits


def _make_sample_nf_record(
    src_ip: str = "10.0.0.1",
    src_port: int = 12345,
    dst_ip: str = "10.0.0.2",
    dst_port: int = 80,
    protocol: int = 6,
    dns_query_id: int = 0,
    in_bytes: float = 100.0,
    out_bytes: float = 200.0,
    flow_duration: float = 50.0,
    const_val: float = 42.0,
    label: int = 0,
    attack: str = "Benign",
) -> dict:
    """Helper to create a single row with standard NF-UNSW-NB15-v2 columns."""
    return {
        "IPV4_SRC_ADDR": src_ip,
        "L4_SRC_PORT": src_port,
        "IPV4_DST_ADDR": dst_ip,
        "L4_DST_PORT": dst_port,
        "PROTOCOL": protocol,
        "DNS_QUERY_ID": dns_query_id,
        "IN_BYTES": in_bytes,
        "OUT_BYTES": out_bytes,
        "FLOW_DURATION_MILLISECONDS": flow_duration,
        "CONST_FEATURE": const_val,
        "Label": label,
        "Attack": attack,
    }


def test_identifier_columns_removed():
    """Verify that identifiers are dropped and destination port is preserved."""
    rows = [
        _make_sample_nf_record(in_bytes=10.0, dst_port=80, attack="Benign", label=0),
        _make_sample_nf_record(in_bytes=20.0, dst_port=443, attack="Exploits", label=1),
    ]
    df = pd.DataFrame(rows)

    df_clean, meta = clean_nf_unsw_dataframe(df, rare_threshold=1, drop_const=False)

    for id_col in IDENTIFIER_COLUMNS:
        assert id_col not in df_clean.columns, f"Identifier {id_col} must be dropped"

    assert DESTINATION_PORT_COL in df_clean.columns, "L4_DST_PORT must be preserved in cleaned df"
    assert "label_multiclass" in df_clean.columns
    assert "label_binary" in df_clean.columns
    assert "Attack" not in df_clean.columns, "Raw Attack column should be replaced"
    assert "Label" not in df_clean.columns, "Raw Label column should be replaced"


def test_rare_class_merging_configurable_threshold():
    """Verify merging of attack classes with fewer than rare_threshold flows."""
    rows = []
    # Benign: 2 flows (below threshold=3, but Benign is NOT an attack class)
    rows.extend([_make_sample_nf_record(in_bytes=float(i), attack="Benign", label=0) for i in range(2)])
    # Exploits: 4 flows (>= threshold=3, should stay 'Exploits')
    rows.extend([_make_sample_nf_record(in_bytes=float(i + 10), attack="Exploits", label=1) for i in range(4)])
    # Analysis: 2 flows (< threshold=3, should be merged into 'Rare')
    rows.extend([_make_sample_nf_record(in_bytes=float(i + 20), attack="Analysis", label=1) for i in range(2)])
    # Worms: 3 flows (always merged to 'Rare' per Section 2.1)
    rows.extend([_make_sample_nf_record(in_bytes=float(i + 30), attack="Worms", label=1) for i in range(3)])

    df = pd.DataFrame(rows)
    df_clean, meta = clean_nf_unsw_dataframe(df, rare_threshold=3, drop_const=False)

    classes_present = set(df_clean["label_multiclass"].unique())

    assert "Benign" in classes_present, "Benign must not be merged into Rare"
    assert "Exploits" in classes_present, "Exploits has 4 flows (>=3), so must remain Exploits"
    assert "Analysis" not in classes_present, "Analysis has 2 flows (<3), so must be merged into Rare"
    assert "Worms" not in classes_present, "Worms must be merged into Rare"
    assert "Rare" in classes_present, "Rare must contain merged Analysis and Worms flows"

    # Check counts in Rare: 2 (Analysis) + 3 (Worms) = 5
    rare_count = (df_clean["label_multiclass"] == "Rare").sum()
    assert rare_count == 5


def test_label_aware_exact_deduplication():
    """Verify that exact duplicates with identical label are dropped, but identical

    features with DIFFERENT labels are retained.
    """
    rows = [
        # Pair 1: Exact duplicate across features and label -> 1 kept, 1 dropped
        _make_sample_nf_record(in_bytes=100.0, out_bytes=200.0, attack="DoS", label=1),
        _make_sample_nf_record(in_bytes=100.0, out_bytes=200.0, attack="DoS", label=1),
        # Pair 2: Ambiguous flow - identical features but DIFFERENT label -> BOTH kept
        _make_sample_nf_record(in_bytes=300.0, out_bytes=400.0, attack="Benign", label=0),
        _make_sample_nf_record(in_bytes=300.0, out_bytes=400.0, attack="Exploits", label=1),
        # Distinct row
        _make_sample_nf_record(in_bytes=500.0, out_bytes=600.0, attack="Generic", label=1),
    ]
    df = pd.DataFrame(rows)

    df_clean, meta = clean_nf_unsw_dataframe(df, rare_threshold=1, drop_const=False)

    # Total rows: 5 input -> 1 dropped -> 4 remaining
    assert len(df_clean) == 4
    assert meta["dropped_duplicates"] == 1

    # Verify both ambiguous flows remain
    ambiguous = df_clean[
        (df_clean["IN_BYTES"] == 300.0) & (df_clean["OUT_BYTES"] == 400.0)
    ]
    assert len(ambiguous) == 2
    assert set(ambiguous["label_multiclass"]) == {"Benign", "Exploits"}


def test_constant_column_removal():
    """Verify that constant feature columns are dropped when requested."""
    rows = [
        _make_sample_nf_record(in_bytes=10.0, const_val=42.0, attack="Benign", label=0),
        _make_sample_nf_record(in_bytes=20.0, const_val=42.0, attack="Exploits", label=1),
        _make_sample_nf_record(in_bytes=30.0, const_val=42.0, attack="Generic", label=1),
    ]
    df = pd.DataFrame(rows)

    # With drop_const=True: CONST_FEATURE should be dropped
    df_clean, meta = clean_nf_unsw_dataframe(df, rare_threshold=1, drop_const=True)
    assert "CONST_FEATURE" not in df_clean.columns
    assert "CONST_FEATURE" in meta["dropped_constant_columns"]

    # With drop_const=False: CONST_FEATURE should be retained
    df_clean_keep, meta_keep = clean_nf_unsw_dataframe(df, rare_threshold=1, drop_const=False)
    assert "CONST_FEATURE" in df_clean_keep.columns
    assert len(meta_keep["dropped_constant_columns"]) == 0


def test_inf_and_nan_handling():
    """Verify that rows with inf or nan in features are dropped."""
    rows = [
        _make_sample_nf_record(in_bytes=10.0, attack="Benign", label=0),
        _make_sample_nf_record(in_bytes=np.inf, attack="Exploits", label=1),
        _make_sample_nf_record(in_bytes=-np.inf, attack="DoS", label=1),
        _make_sample_nf_record(in_bytes=np.nan, attack="Fuzzers", label=1),
        _make_sample_nf_record(in_bytes=20.0, attack="Generic", label=1),
    ]
    df = pd.DataFrame(rows)

    df_clean, meta = clean_nf_unsw_dataframe(df, rare_threshold=1, drop_const=False)

    assert len(df_clean) == 2
    assert meta["dropped_missing"] == 3
    assert set(df_clean["label_multiclass"]) == {"Benign", "Generic"}


def test_float32_dtype_enforcement():
    """Verify that all feature columns are strictly cast to float32."""
    rows = [
        _make_sample_nf_record(in_bytes=10, out_bytes=20, attack="Benign", label=0),
        _make_sample_nf_record(in_bytes=30, out_bytes=40, attack="Exploits", label=1),
    ]
    df = pd.DataFrame(rows)

    df_clean, _ = clean_nf_unsw_dataframe(df, rare_threshold=1, drop_const=False)

    feature_cols = [c for c in df_clean.columns if c not in ("label_multiclass", "label_binary")]
    for col in feature_cols:
        assert df_clean[col].dtype == np.float32, f"Column {col} has dtype {df_clean[col].dtype}, expected float32"

    assert df_clean["label_binary"].dtype == np.int8
    assert pd.api.types.is_string_dtype(df_clean["label_multiclass"])


def test_end_to_end_csv_to_parquet(tmp_path: Path):
    """Verify full clean_nf_unsw_dataset CLI/function pipeline with Parquet and JSON output."""
    csv_file = tmp_path / "raw_nf.csv"
    parquet_file = tmp_path / "cleaned_nf.parquet"
    meta_file = tmp_path / "cleaned_nf_meta.json"

    # Provide varied dst_port so L4_DST_PORT is not constant
    rows = [
        _make_sample_nf_record(in_bytes=10.0, dst_port=80, const_val=1.0, attack="Benign", label=0),
        _make_sample_nf_record(in_bytes=20.0, dst_port=443, const_val=1.0, attack="Exploits", label=1),
        _make_sample_nf_record(in_bytes=20.0, dst_port=443, const_val=1.0, attack="Exploits", label=1),  # duplicate
        _make_sample_nf_record(in_bytes=30.0, dst_port=8080, const_val=1.0, attack="Worms", label=1),     # rare
    ]
    pd.DataFrame(rows).to_csv(csv_file, index=False)

    df_cleaned, metadata = clean_nf_unsw_dataset(
        input_csv_path=csv_file,
        output_parquet_path=parquet_file,
        metadata_json_path=meta_file,
        rare_threshold=2,
        drop_const=True,
    )

    assert parquet_file.exists()
    assert meta_file.exists()

    # Read back parquet
    df_read = pd.read_parquet(parquet_file)
    assert len(df_read) == 3  # 1 duplicate dropped
    assert "CONST_FEATURE" not in df_read.columns
    assert "L4_DST_PORT" in df_read.columns

    # Read back metadata
    with open(meta_file, "r", encoding="utf-8") as f:
        meta_read = json.load(f)
    assert meta_read["raw_rows"] == 4
    assert meta_read["cleaned_rows"] == 3
    assert meta_read["dropped_duplicates"] == 1


def test_compatibility_with_split_preprocess():
    """Verify that cleaned DataFrame integrates seamlessly with split_preprocess.py."""
    # Create synthetic dataset with multiple classes and sufficient samples for stratified 60/20/20 split
    rows = []
    classes = ["Benign", "Exploits", "DoS", "Rare"]
    for cls_idx, cls_name in enumerate(classes):
        for i in range(10):
            rows.append(
                _make_sample_nf_record(
                    dst_port=80 + i,
                    in_bytes=float(i * 10 + cls_idx),
                    out_bytes=float(i * 20 + cls_idx * 5),
                    flow_duration=float(i * 5 + 1),
                    attack=cls_name,
                    label=0 if cls_name == "Benign" else 1,
                    const_val=float(i),  # non-constant
                )
            )

    df = pd.DataFrame(rows)
    df_clean, meta = clean_nf_unsw_dataframe(df, rare_threshold=1, drop_const=True)

    # Run split_preprocess.prepare_splits
    splits = prepare_splits(
        df=df_clean,
        seed=0,
        exclude_destination_port=True,
    )

    assert splits.X_train.dtype == np.float32
    assert splits.X_val.dtype == np.float32
    assert splits.X_test.dtype == np.float32

    # Total rows: 40. 60% train = 24, 20% val = 8, 20% test = 8
    assert len(splits.X_train) == 24
    assert len(splits.X_val) == 8
    assert len(splits.X_test) == 8

    # Destination port should be excluded from X
    assert DESTINATION_PORT_COL not in splits.feature_names
    assert "label_multiclass" not in splits.feature_names
    assert "label_binary" not in splits.feature_names
