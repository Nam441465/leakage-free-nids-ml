"""Cleaning pipeline for CICIDS2017 dataset.

Adheres strictly to Section 2.1 of 2611.04660v1.pdf and AGENTS.md:
1. Identifier removal: drops Flow ID, Source IP, Destination IP, Source Port, Timestamp.
2. Duplicate header column removal: drops duplicate 'Fwd Header Length.1' / second header.
3. Label normalization: maps all attack variants to the exact 8 canonical families:
   Benign, DoS, DDoS, PortScan, BruteForce (FTP-Patator, SSH-Patator),
   Web (Web Attack variants), Bot, Rare (Infiltration, Heartbleed, and any other).
4. Value cleaning:
   - Replaces inf and -inf with NaN.
   - Drops rows with missing/NaN values and empty trailing lines.
5. Label-aware exact de-duplication:
   - Drops rows identical across ALL remaining features + label.
   - Retains rows with identical features but differing labels.
6. Constant column removal:
   - Drops columns with zero variance / single unique value across the full dataset.
7. Memory hygiene:
   - Casts all numeric features to float32.
   - Preserves Destination Port in cleaned parquet (excluded at training time by split_preprocess).
   - Explicit garbage collection (del and gc.collect()) to operate safely within 11 GiB RAM.
8. Saves cleaned data to Parquet and metadata to JSON.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# Columns to drop as flow/host identifiers per Section 2.1
# Note: TrafficLabelling CSVs also include 'Protocol' (absent from MachineLearningCVE);
# dropping it alongside the 5 flow identifiers + 1 duplicate header + 8 constant columns
# yields the exact 69 features (68 after excluding Destination Port) reported in Section 2.1.
IDENTIFIER_COLUMNS: list[str] = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Source Port",
    "Timestamp",
    "Protocol",
]

# Duplicate header columns to drop
DUPLICATE_HEADER_COLUMNS: list[str] = [
    "Fwd Header Length.1",
    "Fwd Header Length_1",
    "Fwd Header Length.2",
]

# Destination port column (preserved in cleaned parquet, excluded at training time)
DESTINATION_PORT_COL: str = "Destination Port"

# The 8 canonical target families per Section 2.1 and Table 2
TARGET_FAMILIES: list[str] = [
    "Benign",
    "DoS",
    "DDoS",
    "PortScan",
    "BruteForce",
    "Web",
    "Bot",
    "Rare",
]

# Direct lookup dictionary for raw label strings (lowercased)
RAW_LABEL_MAPPING: dict[str, str] = {
    # Benign
    "benign": "Benign",
    # DoS
    "dos hulk": "DoS",
    "dos goldeneye": "DoS",
    "dos slowloris": "DoS",
    "dos slowhttptest": "DoS",
    # DDoS
    "ddos": "DDoS",
    # PortScan
    "portscan": "PortScan",
    # BruteForce
    "ftp-patator": "BruteForce",
    "ssh-patator": "BruteForce",
    # Bot
    "bot": "Bot",
    # Rare
    "infiltration": "Rare",
    "heartbleed": "Rare",
}


def map_cicids_label(raw_label: str | object) -> str:
    """Map a raw CICIDS2017 label string to one of the 8 canonical families.

    Parameters
    ----------
    raw_label : str | object
        Raw label value from the CSV (e.g. 'DoS Hulk', 'Web Attack \x96 Brute Force').

    Returns
    -------
    str
        One of the 8 canonical families: Benign, DoS, DDoS, PortScan, BruteForce,
        Web, Bot, Rare.
    """
    if raw_label is None or pd.isna(raw_label):
        return "Rare"

    cleaned = str(raw_label).strip()
    lower = cleaned.lower()

    # 1. Direct dictionary match
    if lower in RAW_LABEL_MAPPING:
        return RAW_LABEL_MAPPING[lower]

    # 2. Pattern matching for Web Attacks (handles en-dash, hyphens, encoding artifacts)
    if "web attack" in lower:
        return "Web"

    # 3. Pattern matching for password guessing / brute force
    if "patator" in lower or "bruteforce" in lower or "brute force" in lower:
        return "BruteForce"

    # 4. Pattern matching for DoS and DDoS
    if "ddos" in lower:
        return "DDoS"
    if "dos" in lower:
        return "DoS"

    # 5. Pattern matching for scans and bots
    if "portscan" in lower or "port scan" in lower:
        return "PortScan"
    if "bot" in lower:
        return "Bot"

    # 6. Benign check
    if "benign" in lower:
        return "Benign"

    # 7. Fallback for any other / rare attacks (e.g. Infiltration, Heartbleed)
    return "Rare"


def clean_cicids_dataframe(
    df: pd.DataFrame,
    drop_const: bool = True,
    dedup: bool = True,
    preserve_cols: Sequence[str] | None = None,
    cast_float32: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Clean a CICIDS2017 DataFrame following the leakage-free protocol.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or merged CICIDS2017 dataframe.
    drop_const : bool, default=True
        Whether to drop constant feature columns.
    dedup : bool, default=True
        Whether to perform label-aware exact de-duplication.
    preserve_cols : Sequence[str] | None, default=None
        Columns that must not be dropped as constant columns even if invariant
        (e.g. DESTINATION_PORT_COL). Defaults to (DESTINATION_PORT_COL,).
    cast_float32 : bool, default=True
        Whether to cast numeric feature columns to float32 AFTER constant-column
        removal and label-aware de-duplication (keeping full float64 precision
        during de-duplication to avoid false float32 collisions).

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Cleaned dataframe with float32 features (when cast_float32=True),
        label_multiclass, label_binary, and a metadata dictionary summarizing
        row/column counts and distributions.
    """
    df = df.copy()
    raw_rows = len(df)
    metadata: dict = {
        "raw_rows": raw_rows,
    }

    if preserve_cols is None:
        preserve_cols = (DESTINATION_PORT_COL,)
    preserve_set = set(preserve_cols)

    # 1. Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # 2. Drop duplicate header columns (e.g. 'Fwd Header Length.1' or duplicate column names)
    dup_cols_to_drop = [c for c in DUPLICATE_HEADER_COLUMNS if c in df.columns]
    if dup_cols_to_drop:
        df = df.drop(columns=dup_cols_to_drop)

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # 3. Find and extract the label column
    label_candidates = [c for c in df.columns if c.strip().lower() in ("label", "label_multiclass", "attack")]
    if not label_candidates:
        raise ValueError("Could not find 'Label' column in input DataFrame.")
    raw_label_col = label_candidates[0]

    # Map labels to 8 canonical families
    mapped_multiclass = df[raw_label_col].map(map_cicids_label).astype(str)
    df["label_multiclass"] = mapped_multiclass
    df["label_binary"] = (mapped_multiclass != "Benign").astype(np.int8)

    # Drop raw label column if named differently from label_multiclass
    if raw_label_col not in ("label_multiclass", "label_binary"):
        df = df.drop(columns=[raw_label_col])

    # 4. Drop identifier columns
    cols_to_drop = [c for c in df.columns if c.strip() in IDENTIFIER_COLUMNS or c.strip().lower() in {x.lower() for x in IDENTIFIER_COLUMNS}]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Identify feature columns
    label_cols = ["label_multiclass", "label_binary"]
    feature_cols = [c for c in df.columns if c not in label_cols]

    # 5. Handle infinite and missing values
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    rows_before_na = len(df)
    df = df.dropna(subset=feature_cols + label_cols)
    rows_after_na = len(df)
    metadata["dropped_missing"] = rows_before_na - rows_after_na

    # 6. Drop constant columns if requested (in full precision before float32 cast)
    dropped_constant_cols: list[str] = []
    if drop_const:
        for col in feature_cols:
            if col in preserve_set:
                continue
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min == col_max or df[col].nunique(dropna=False) <= 1:
                dropped_constant_cols.append(col)

        if dropped_constant_cols:
            df = df.drop(columns=dropped_constant_cols)
            feature_cols = [c for c in feature_cols if c not in dropped_constant_cols]

    metadata["dropped_constant_columns"] = dropped_constant_cols

    # 7. Label-aware exact de-duplication in float64 precision (rounded to 10 decimals
    #    to remove Java CICFlowMeter 11th-15th decimal float serialization jitter while
    #    preserving full precision far beyond float32)
    # Only drop rows that are identical across ALL features AND label_multiclass
    rows_before_dedup = len(df)
    if dedup:
        for col in feature_cols:
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].round(10)
        df = df.drop_duplicates(subset=feature_cols + ["label_multiclass"], keep="first")
    rows_after_dedup = len(df)
    metadata["dropped_duplicates"] = rows_before_dedup - rows_after_dedup

    # 8. Cast numeric features to float32 AFTER dedup to save memory for Parquet/training
    if cast_float32:
        for col in feature_cols:
            df[col] = df[col].astype(np.float32)

    # 9. Reset index and reorder columns: features first, then labels
    df = df.reset_index(drop=True)
    ordered_columns = feature_cols + label_cols
    df = df[ordered_columns].copy()

    # 10. Metadata summary
    metadata["cleaned_rows"] = len(df)
    metadata["n_features"] = len(feature_cols)
    metadata["feature_columns"] = feature_cols
    metadata["has_destination_port"] = DESTINATION_PORT_COL in feature_cols
    metadata["destination_port_col"] = DESTINATION_PORT_COL if DESTINATION_PORT_COL in feature_cols else None
    metadata["class_distribution"] = df["label_multiclass"].value_counts().to_dict()
    metadata["binary_distribution"] = {
        int(k): int(v) for k, v in df["label_binary"].value_counts().items()
    }

    del mapped_multiclass
    gc.collect()

    return df, metadata


def clean_cicids_file(
    file_path: str | Path,
    drop_const: bool = False,
    dedup: bool = False,
    cast_float32: bool = True,
) -> pd.DataFrame:
    """Read and clean a single raw CICIDS2017 CSV file.

    Handles encodings (cp1252/latin-1/utf-8), skips blank rows, drops identifiers
    and duplicate headers, maps labels, replaces inf with NaN, drops missing rows,
    and optionally downcasts features to float32.

    Parameters
    ----------
    file_path : str | Path
        Path to the raw CSV file.
    drop_const : bool, default=False
        Whether to drop constant columns. Typically False for single files so that
        constant columns are evaluated globally across all combined files.
    dedup : bool, default=False
        Whether to de-duplicate per-file. Typically False for single files so that
        de-duplication is evaluated globally across all combined files.
    cast_float32 : bool, default=True
        Whether to cast numeric features to float32. Set False when combining
        multiple files before global float64 de-duplication.

    Returns
    -------
    pd.DataFrame
        Cleaned single-file DataFrame with standardized labels.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Read CSV with encoding fallback
    df = None
    for enc in ("cp1252", "latin-1", "utf-8"):
        try:
            df = pd.read_csv(file_path, encoding=enc, low_memory=False)
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise RuntimeError(f"Failed to decode {file_path} with cp1252, latin-1, or utf-8.")

    # Drop completely empty rows (e.g. 288,602 trailing empty lines in WebAttacks CSV)
    df = df.dropna(how="all")

    # Run DataFrame cleaning without global constant/dedup
    df_clean, _ = clean_cicids_dataframe(
        df,
        drop_const=drop_const,
        dedup=dedup,
        cast_float32=cast_float32,
    )
    del df
    gc.collect()

    return df_clean


def clean_cicids2017_dataset(
    raw_dir: str | Path = "data/raw/cicids2017/TrafficLabelling",
    output_parquet_path: str | Path = "data/processed/cicids2017.parquet",
    metadata_json_path: str | Path | None = "data/processed/cicids2017.json",
    drop_const: bool = True,
    dedup: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Execute the full end-to-end cleaning pipeline on all 8 CICIDS2017 CSVs.

    Streams files individually to maintain memory under WSL RAM limits (~11 GiB),
    concatenates, drops constant features globally, performs global label-aware
    de-duplication, saves the resulting Parquet file, and exports JSON metadata.

    Parameters
    ----------
    raw_dir : str | Path
        Directory containing the 8 raw CSV files.
    output_parquet_path : str | Path
        Output path for the cleaned Parquet file.
    metadata_json_path : str | Path | None, default='data/processed/cicids2017.json'
        Output path for metadata JSON file. If None, metadata is not written to disk.
    drop_const : bool, default=True
        Whether to drop constant feature columns globally.
    dedup : bool, default=True
        Whether to perform label-aware exact de-duplication globally.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        The cleaned full DataFrame and metadata dictionary.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_path}")

    csv_files = sorted(list(raw_path.glob("*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_path}")

    dfs: list[pd.DataFrame] = []
    total_raw_rows = 0

    for csv_file in csv_files:
        # Keep float64 precision across files so global dedup avoids float32 collisions
        df_part = clean_cicids_file(csv_file, drop_const=False, dedup=False, cast_float32=False)
        total_raw_rows += len(df_part)
        dfs.append(df_part)
        gc.collect()

    # Concatenate all parts
    full_df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()

    # Apply global constant column dropping and global label-aware de-duplication
    label_cols = ["label_multiclass", "label_binary"]
    feature_cols = [c for c in full_df.columns if c not in label_cols]

    metadata: dict = {
        "raw_csv_files": [f.name for f in csv_files],
        "rows_after_file_clean": len(full_df),
    }

    # Global constant column detection (in float64)
    dropped_constant_cols: list[str] = []
    if drop_const:
        for col in feature_cols:
            if col == DESTINATION_PORT_COL:
                continue
            col_min = full_df[col].min()
            col_max = full_df[col].max()
            if col_min == col_max or full_df[col].nunique(dropna=False) <= 1:
                dropped_constant_cols.append(col)

        if dropped_constant_cols:
            full_df = full_df.drop(columns=dropped_constant_cols)
            feature_cols = [c for c in feature_cols if c not in dropped_constant_cols]

    metadata["dropped_constant_columns"] = dropped_constant_cols

    # Global label-aware exact de-duplication (in float64 rounded to 10 decimals
    # before float32 cast to match Table 2 of 2611.04660v1.pdf)
    rows_before_dedup = len(full_df)
    if dedup:
        for col in feature_cols:
            if pd.api.types.is_float_dtype(full_df[col]):
                full_df[col] = full_df[col].round(10)
        full_df = full_df.drop_duplicates(
            subset=feature_cols + ["label_multiclass"],
            keep="first",
        )
    rows_after_dedup = len(full_df)
    metadata["dropped_duplicates"] = rows_before_dedup - rows_after_dedup

    # Cast numeric features to float32 AFTER global dedup for memory-efficient Parquet/training
    for col in feature_cols:
        full_df[col] = full_df[col].astype(np.float32)

    # Reset index and ensure column ordering
    full_df = full_df.reset_index(drop=True)
    ordered_columns = feature_cols + label_cols
    full_df = full_df[ordered_columns].copy()

    # Compute final metadata
    metadata["cleaned_rows"] = len(full_df)
    metadata["n_features"] = len(feature_cols)
    metadata["feature_columns"] = feature_cols
    metadata["has_destination_port"] = DESTINATION_PORT_COL in feature_cols
    metadata["destination_port_col"] = DESTINATION_PORT_COL if DESTINATION_PORT_COL in feature_cols else None
    metadata["class_distribution"] = full_df["label_multiclass"].value_counts().to_dict()
    metadata["binary_distribution"] = {
        int(k): int(v) for k, v in full_df["label_binary"].value_counts().items()
    }

    # Save output parquet
    out_parquet = Path(output_parquet_path)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(out_parquet, engine="pyarrow", index=False)

    # Save metadata JSON
    if metadata_json_path is not None:
        out_meta = Path(metadata_json_path)
        out_meta.parent.mkdir(parents=True, exist_ok=True)
        with open(out_meta, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    gc.collect()
    return full_df, metadata


def main() -> None:
    """CLI entry point for cleaning CICIDS2017."""
    parser = argparse.ArgumentParser(
        description="Clean raw CICIDS2017 CSVs following the leakage-free protocol."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw/cicids2017/TrafficLabelling",
        help="Path to raw CICIDS2017 TrafficLabelling directory containing 8 CSVs.",
    )
    parser.add_argument(
        "--output-parquet",
        type=str,
        default="data/processed/cicids2017.parquet",
        help="Path to save cleaned Parquet output.",
    )
    parser.add_argument(
        "--output-meta",
        type=str,
        default="data/processed/cicids2017.json",
        help="Path to save cleaning metadata JSON output.",
    )
    parser.add_argument(
        "--no-drop-const",
        action="store_true",
        help="Disable dropping constant columns.",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable label-aware de-duplication.",
    )

    args = parser.parse_args()

    print(f"Starting CICIDS2017 cleaning from: {args.raw_dir}")
    df_clean, meta = clean_cicids2017_dataset(
        raw_dir=args.raw_dir,
        output_parquet_path=args.output_parquet,
        metadata_json_path=args.output_meta,
        drop_const=not args.no_drop_const,
        dedup=not args.no_dedup,
    )

    print(f"Cleaning complete! Saved to {args.output_parquet}")
    print(f"Cleaned rows: {meta['cleaned_rows']:,}")
    print(f"Features: {meta['n_features']}")
    print(f"Destination Port preserved: {meta['has_destination_port']}")
    print("Class distribution:")
    for cls_name, count in meta["class_distribution"].items():
        print(f"  {cls_name}: {count:,}")


if __name__ == "__main__":
    main()
