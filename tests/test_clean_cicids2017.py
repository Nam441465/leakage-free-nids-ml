"""Unit tests for CICIDS2017 data cleaning pipeline.

Validates the leakage-free cleaning protocol:
1. Identifier removal (Flow ID, Source IP, Destination IP, Source Port, Timestamp).
2. Duplicate header column removal (Fwd Header Length.1 / duplicate header).
3. Preserving Destination Port in cleaned parquet (excluded later at training time).
4. Label normalization to 8 canonical families: Benign, DoS, DDoS, PortScan,
   BruteForce, Web, Bot, Rare (including Web Attack with en-dash/CP1252/Latin-1).
5. Label-aware exact de-duplication:
   - Rows identical across all features + label are dropped.
   - Rows identical across features but with DIFFERENT labels are strictly KEPT.
6. Handling inf / -inf and NaN values (dropped).
7. Handling blank trailing lines (as in WebAttacks CSV).
8. Constant column removal.
9. Strict enforcement of float32 dtype for all numeric features.
10. End-to-end dataset pipeline (saving Parquet + JSON metadata).
11. Interoperability with split_preprocess.py.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.clean_cicids2017 import (
    DESTINATION_PORT_COL,
    IDENTIFIER_COLUMNS,
    TARGET_FAMILIES,
    clean_cicids_dataframe,
    clean_cicids2017_dataset,
    clean_cicids_file,
    map_cicids_label,
)
from src.data.split_preprocess import prepare_splits


def _make_sample_cicids_row(
    flow_id: str = "192.168.10.1-192.168.10.2-1234-80-6",
    src_ip: str = "192.168.10.1",
    src_port: int = 1234,
    dst_ip: str = "192.168.10.2",
    dst_port: int = 80,
    protocol: int = 6,
    timestamp: str = "7/7/2017 8:30",
    flow_duration: float = 100.0,
    tot_fwd_pkts: float = 5.0,
    tot_bwd_pkts: float = 3.0,
    fwd_hdr_len: float = 20.0,
    fwd_hdr_len_dup: float = 20.0,
    const_val: float = 0.0,
    label: str = "BENIGN",
) -> dict:
    """Helper to construct a single dictionary matching CICIDS2017 headers."""
    return {
        "Flow ID": flow_id,
        "Source IP": src_ip,
        "Source Port": src_port,
        "Destination IP": dst_ip,
        "Destination Port": dst_port,
        "Protocol": protocol,
        "Timestamp": timestamp,
        "Flow Duration": flow_duration,
        "Total Fwd Packets": tot_fwd_pkts,
        "Total Backward Packets": tot_bwd_pkts,
        "Fwd Header Length": fwd_hdr_len,
        "Fwd Header Length.1": fwd_hdr_len_dup,
        "CONST_FEATURE": const_val,
        "Label": label,
    }


def test_identifier_and_duplicate_header_removal():
    """Verify that identifiers and duplicate header are dropped, but Destination Port is kept."""
    rows = [
        _make_sample_cicids_row(flow_duration=10.0, label="BENIGN"),
        _make_sample_cicids_row(flow_duration=20.0, label="DoS Hulk"),
    ]
    df = pd.DataFrame(rows)

    df_clean, meta = clean_cicids_dataframe(df, drop_const=False, dedup=False)

    for id_col in IDENTIFIER_COLUMNS:
        assert id_col not in df_clean.columns, f"Identifier column {id_col} must be dropped"

    assert "Fwd Header Length.1" not in df_clean.columns, "Duplicate Fwd Header Length.1 must be dropped"
    assert "Fwd Header Length" in df_clean.columns, "Original Fwd Header Length must be preserved"
    assert DESTINATION_PORT_COL in df_clean.columns, "Destination Port must be preserved in cleaned df"
    assert "label_multiclass" in df_clean.columns
    assert "label_binary" in df_clean.columns


def test_label_mapping_8_families():
    """Verify mapping of all raw label strings to the exact 8 canonical families."""
    test_cases = [
        # Benign
        ("BENIGN", "Benign", 0),
        ("Benign", "Benign", 0),
        # DoS
        ("DoS Hulk", "DoS", 1),
        ("DoS GoldenEye", "DoS", 1),
        ("DoS slowloris", "DoS", 1),
        ("DoS Slowhttptest", "DoS", 1),
        # DDoS
        ("DDoS", "DDoS", 1),
        # PortScan
        ("PortScan", "PortScan", 1),
        # BruteForce
        ("FTP-Patator", "BruteForce", 1),
        ("SSH-Patator", "BruteForce", 1),
        # Web attacks (standard, en-dash, hyphens)
        ("Web Attack – Brute Force", "Web", 1),
        ("Web Attack \x96 Brute Force", "Web", 1),
        ("Web Attack  Brute Force", "Web", 1),
        ("Web Attack – XSS", "Web", 1),
        ("Web Attack – Sql Injection", "Web", 1),
        # Bot
        ("Bot", "Bot", 1),
        # Rare
        ("Infiltration", "Rare", 1),
        ("Heartbleed", "Rare", 1),
        ("UnknownNovelAttack", "Rare", 1),
    ]

    for raw, expected_family, expected_binary in test_cases:
        family = map_cicids_label(raw)
        assert family == expected_family, f"Failed mapping '{raw}': got '{family}', expected '{expected_family}'"
        assert family in TARGET_FAMILIES, f"Family '{family}' not in TARGET_FAMILIES"

    # Test DataFrame integration
    rows = [_make_sample_cicids_row(flow_duration=float(i), label=raw) for i, (raw, _, _) in enumerate(test_cases)]
    df = pd.DataFrame(rows)
    df_clean, meta = clean_cicids_dataframe(df, drop_const=False, dedup=False)

    for i, (_, expected_family, expected_binary) in enumerate(test_cases):
        assert df_clean.iloc[i]["label_multiclass"] == expected_family
        assert df_clean.iloc[i]["label_binary"] == expected_binary


def test_label_aware_exact_deduplication():
    """Verify label-aware exact de-duplication:

    - Identical features + identical label -> duplicate, dropped.
    - Identical features + DIFFERENT label -> NOT duplicate, kept!
    """
    rows = [
        # Row 0: Features X, label Benign
        _make_sample_cicids_row(flow_duration=100.0, tot_fwd_pkts=5.0, label="BENIGN"),
        # Row 1: Identical features X, identical label Benign (DUPLICATE OF ROW 0)
        _make_sample_cicids_row(flow_duration=100.0, tot_fwd_pkts=5.0, label="BENIGN"),
        # Row 2: Identical features X, DIFFERENT label DoS (MUST BE KEPT)
        _make_sample_cicids_row(flow_duration=100.0, tot_fwd_pkts=5.0, label="DoS Hulk"),
        # Row 3: Identical features X, DIFFERENT label BruteForce (MUST BE KEPT)
        _make_sample_cicids_row(flow_duration=100.0, tot_fwd_pkts=5.0, label="FTP-Patator"),
        # Row 4: Distinct features Y, label PortScan (MUST BE KEPT)
        _make_sample_cicids_row(flow_duration=200.0, tot_fwd_pkts=1.0, label="PortScan"),
        # Row 5: Identical features Y, identical label PortScan (DUPLICATE OF ROW 4)
        _make_sample_cicids_row(flow_duration=200.0, tot_fwd_pkts=1.0, label="PortScan"),
    ]
    df = pd.DataFrame(rows)

    df_clean, meta = clean_cicids_dataframe(df, drop_const=False, dedup=True)

    # Out of 6 rows:
    # Row 1 is a duplicate of Row 0 (same features, same Benign label) -> dropped
    # Row 5 is a duplicate of Row 4 (same features, same PortScan label) -> dropped
    # Rows 0 (Benign), 2 (DoS), 3 (BruteForce), 4 (PortScan) must remain!
    assert len(df_clean) == 4, f"Expected 4 rows after label-aware dedup, got {len(df_clean)}"
    assert meta["dropped_duplicates"] == 2

    # Check that both Benign, DoS, and BruteForce are all present with features X
    counts = df_clean["label_multiclass"].value_counts()
    assert counts.get("Benign", 0) == 1
    assert counts.get("DoS", 0) == 1
    assert counts.get("BruteForce", 0) == 1
    assert counts.get("PortScan", 0) == 1


def test_inf_and_nan_handling():
    """Verify that inf, -inf, and NaN in numeric columns are removed."""
    rows = [
        _make_sample_cicids_row(flow_duration=100.0, label="BENIGN"),
        _make_sample_cicids_row(flow_duration=float("inf"), label="DoS Hulk"),
        _make_sample_cicids_row(flow_duration=float("-inf"), label="DDoS"),
        _make_sample_cicids_row(flow_duration=float("nan"), label="PortScan"),
        _make_sample_cicids_row(flow_duration=200.0, label="Bot"),
    ]
    df = pd.DataFrame(rows)

    df_clean, meta = clean_cicids_dataframe(df, drop_const=False, dedup=False)

    assert len(df_clean) == 2, f"Expected 2 clean rows, got {len(df_clean)}"
    assert meta["dropped_missing"] == 3
    assert set(df_clean["label_multiclass"].unique()) == {"Benign", "Bot"}


def test_blank_lines_handling(tmp_path: Path):
    """Verify handling of blank/empty trailing lines (as in WebAttacks CSV)."""
    csv_file = tmp_path / "test_with_blank_lines.csv"

    # Write a header, 3 valid rows, and 10 completely blank trailing lines
    content = (
        "Flow ID, Source IP, Source Port, Destination IP, Destination Port, Timestamp, Flow Duration, Total Fwd Packets, Total Backward Packets, Fwd Header Length, Fwd Header Length, CONST_FEATURE, Label\n"
        "1,10.0.0.1,1234,10.0.0.2,80,2017-07-07,100,5,3,20,20,0,BENIGN\n"
        "2,10.0.0.1,1235,10.0.0.2,80,2017-07-07,200,6,4,20,20,0,DoS Hulk\n"
        "3,10.0.0.1,1236,10.0.0.2,80,2017-07-07,300,7,5,20,20,0,PortScan\n"
        "\n\n\n\n\n\n\n\n\n\n"
    )
    csv_file.write_text(content, encoding="utf-8")

    df_clean = clean_cicids_file(csv_file, drop_const=False, dedup=False)
    assert len(df_clean) == 3, f"Expected 3 valid rows after skipping blanks, got {len(df_clean)}"


def test_constant_column_removal():
    """Verify that constant features are dropped while Destination Port is kept."""
    rows = [
        _make_sample_cicids_row(flow_duration=10.0, const_val=0.0, label="BENIGN"),
        _make_sample_cicids_row(flow_duration=20.0, const_val=0.0, label="DoS Hulk"),
        _make_sample_cicids_row(flow_duration=30.0, const_val=0.0, label="Bot"),
    ]
    df = pd.DataFrame(rows)

    df_clean, meta = clean_cicids_dataframe(df, drop_const=True, dedup=False)

    assert "CONST_FEATURE" not in df_clean.columns, "CONST_FEATURE had constant value 0.0 and must be dropped"
    assert "CONST_FEATURE" in meta["dropped_constant_columns"]
    assert "Flow Duration" in df_clean.columns, "Varying feature must be kept"
    assert DESTINATION_PORT_COL in df_clean.columns, "Destination Port must be preserved"


def test_float32_dtype_enforcement():
    """Verify that all feature columns are strictly float32."""
    rows = [
        _make_sample_cicids_row(flow_duration=10.5, tot_fwd_pkts=2.0, label="BENIGN"),
        _make_sample_cicids_row(flow_duration=20.5, tot_fwd_pkts=4.0, label="DoS Hulk"),
    ]
    df = pd.DataFrame(rows)

    df_clean, _ = clean_cicids_dataframe(df, drop_const=False, dedup=False)

    feature_cols = [c for c in df_clean.columns if c not in ("label_multiclass", "label_binary")]
    for col in feature_cols:
        assert df_clean[col].dtype == np.float32, f"Column {col} has dtype {df_clean[col].dtype}, expected float32"

    assert df_clean["label_binary"].dtype == np.int8
    assert pd.api.types.is_string_dtype(df_clean["label_multiclass"])


def test_end_to_end_dataset_pipeline(tmp_path: Path):
    """Verify clean_cicids2017_dataset on synthetic directory of multiple CSVs."""
    raw_dir = tmp_path / "raw_cic"
    raw_dir.mkdir()

    # CSV 1: Normal benign and DoS
    csv1 = raw_dir / "Monday.csv"
    csv1_content = (
        "Flow ID, Source IP, Source Port, Destination IP, Destination Port, Timestamp, Flow Duration, Total Fwd Packets, Total Backward Packets, Fwd Header Length, Fwd Header Length, CONST_FEATURE, Label\n"
        "1,10.0.0.1,1001,10.0.0.2,80,2017-07-07,100,5,3,20,20,0,BENIGN\n"
        "2,10.0.0.1,1002,10.0.0.2,80,2017-07-07,200,6,4,20,20,0,DoS Hulk\n"
    )
    csv1.write_text(csv1_content, encoding="utf-8")

    # CSV 2: Web attack with CP1252 en-dash and blank trailing lines
    csv2 = raw_dir / "Thursday.csv"
    csv2_bytes = (
        b"Flow ID, Source IP, Source Port, Destination IP, Destination Port, Timestamp, Flow Duration, Total Fwd Packets, Total Backward Packets, Fwd Header Length, Fwd Header Length, CONST_FEATURE, Label\n"
        b"3,10.0.0.1,1003,10.0.0.2,80,2017-07-07,300,7,5,20,20,0,Web Attack \x96 Brute Force\n"
        b"4,10.0.0.1,1004,10.0.0.2,80,2017-07-07,400,8,6,20,20,0,Heartbleed\n"
        b"\n\n\n"
    )
    csv2.write_bytes(csv2_bytes)

    out_parquet = tmp_path / "processed" / "cicids2017.parquet"
    out_meta = tmp_path / "processed" / "cicids2017.json"

    df_result, meta = clean_cicids2017_dataset(
        raw_dir=raw_dir,
        output_parquet_path=out_parquet,
        metadata_json_path=out_meta,
        drop_const=True,
        dedup=True,
    )

    assert out_parquet.exists()
    assert out_meta.exists()
    assert len(df_result) == 4

    with open(out_meta, "r", encoding="utf-8") as f:
        meta_loaded = json.load(f)

    assert meta_loaded["cleaned_rows"] == 4
    assert "CONST_FEATURE" in meta_loaded["dropped_constant_columns"]
    assert "Web" in meta_loaded["class_distribution"]
    assert "Rare" in meta_loaded["class_distribution"]


def test_interoperability_with_split_preprocess():
    """Verify that cleaned DataFrame seamlessly works with split_preprocess.prepare_splits."""
    rows = []
    # Create sufficient rows across multiple classes for 60/20/20 split
    classes = ["BENIGN", "DoS Hulk", "PortScan", "FTP-Patator"]
    for i in range(40):
        cls_name = classes[i % len(classes)]
        rows.append(
            _make_sample_cicids_row(
                flow_duration=float(i * 10),
                tot_fwd_pkts=float(i + 1),
                dst_port=80 + (i % 5),
                const_val=0.0,
                label=cls_name,
            )
        )
    df = pd.DataFrame(rows)

    df_clean, _ = clean_cicids_dataframe(df, drop_const=True, dedup=True)

    # Destination Port should be present in df_clean
    assert DESTINATION_PORT_COL in df_clean.columns

    # Pass to prepare_splits
    splits = prepare_splits(df_clean, seed=0, exclude_destination_port=True)

    assert splits.X_train.dtype == np.float32
    assert splits.X_val.dtype == np.float32
    assert splits.X_test.dtype == np.float32
    assert DESTINATION_PORT_COL not in splits.feature_names
    assert len(splits.X_train) + len(splits.X_val) + len(splits.X_test) == len(df_clean)


def test_dedup_runs_in_float64_before_float32_cast():
    """Verify dedup runs in float64 (rounded to 10 decimals) before float32 cast."""
    # 16777216.0 and 16777217.0 are distinct in float64, but both round to 16777216.0 in float32.
    assert np.float32(16777216.0) == np.float32(16777217.0)
    rows = [
        _make_sample_cicids_row(flow_duration=16777216.0, label="BENIGN"),
        _make_sample_cicids_row(flow_duration=16777217.0, label="BENIGN"),
        _make_sample_cicids_row(flow_duration=16777217.0, label="BENIGN"),  # true exact duplicate
        _make_sample_cicids_row(flow_duration=100.0, label="BENIGN"),
        _make_sample_cicids_row(flow_duration=100.0 + 1e-12, label="BENIGN"),  # float jitter beyond 10 decimals
    ]
    df = pd.DataFrame(rows)

    df_clean, meta = clean_cicids_dataframe(df, drop_const=False, dedup=True, cast_float32=True)

    assert len(df_clean) == 3, f"Expected 3 distinct rows, got {len(df_clean)}"
    assert meta["dropped_duplicates"] == 2
    assert df_clean["Flow Duration"].dtype == np.float32
    assert "Protocol" not in df_clean.columns
