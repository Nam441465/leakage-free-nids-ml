"""Cleaning pipeline for NF-UNSW-NB15-v2 dataset.

Adheres strictly to Section 2.1 of 2611.04660v1.pdf and AGENTS.md:
1. Identifier removal: drops L4_SRC_PORT, DNS_QUERY_ID, IPV4_SRC_ADDR, IPV4_DST_ADDR.
2. Label extraction & normalization:
   - Attack classes with fewer than 500 flows merged into 'Rare' (here, 'Worms').
   - Preserves 10 families: Benign, Analysis, Backdoor, DoS, Exploits, Fuzzers,
     Generic, Reconnaissance, Shellcode, Rare.
   - Produces label_multiclass (string) and label_binary (int8).
3. Value cleaning:
   - Replaces inf and -inf with NaN.
   - Drops rows with missing/NaN values.
4. Label-aware exact de-duplication:
   - Drops rows identical across ALL remaining features + label.
   - Retains rows with identical features but differing labels.
5. Constant column removal:
   - Drops columns with zero variance / single unique value across the dataset.
6. Memory hygiene:
   - Casts all numeric features to float32.
   - Explicit garbage collection (del and gc.collect()) to operate safely within 11 GiB RAM.
7. Saves cleaned data to Parquet and metadata to JSON.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# Columns to drop as identifiers / host-graph only
IDENTIFIER_COLUMNS: list[str] = [
    "IPV4_SRC_ADDR",
    "L4_SRC_PORT",
    "IPV4_DST_ADDR",
    "DNS_QUERY_ID",
]

# Destination port column (preserved in cleaned parquet, excluded at training time)
DESTINATION_PORT_COL: str = "L4_DST_PORT"

# Raw label column names in NF-UNSW-NB15-v2
RAW_LABEL_COLUMNS: list[str] = ["Label", "Attack"]

# The 10 canonical target families per Section 2.1 and Table 3
TARGET_FAMILIES: list[str] = [
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


def clean_nf_unsw_dataframe(
    df: pd.DataFrame,
    rare_threshold: int = 500,
    drop_const: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Clean NF-UNSW-NB15-v2 DataFrame following the leakage-free protocol.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or pre-filtered NF-UNSW-NB15-v2 dataframe.
    rare_threshold : int, default=500
        Minimum flow count required for an attack family. Families with fewer
        flows are merged into 'Rare'. Default is 500 per Section 2.1.
    drop_const : bool, default=True
        Whether to drop constant feature columns.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Cleaned dataframe with float32 features, label_multiclass, label_binary,
        and a metadata dictionary summarizing row/column counts and distributions.
    """
    df = df.copy()
    raw_rows = len(df)
    metadata: dict = {
        "raw_rows": raw_rows,
        "rare_threshold": rare_threshold,
    }

    # 1. Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()

    # 2. Separate and normalize labels before dropping identifiers
    if "Attack" not in df.columns:
        raise ValueError("Expected 'Attack' column in input DataFrame.")

    # Normalize attack strings
    attack_col = df["Attack"].astype(str).str.strip()

    # Determine attack classes with < rare_threshold flows (excluding 'Benign')
    attack_counts = attack_col.value_counts()
    rare_mask_counts = (attack_counts < rare_threshold) & (attack_counts.index != "Benign")
    rare_classes_to_merge = set(attack_counts[rare_mask_counts].index)

    # Per paper: 'Worms' has 164 flows and is explicitly merged to 'Rare'
    if "Worms" in attack_col.values:
        rare_classes_to_merge.add("Worms")

    # Apply rare mapping
    if rare_classes_to_merge:
        attack_col = attack_col.replace({cls_name: "Rare" for cls_name in rare_classes_to_merge})

    df["label_multiclass"] = attack_col

    # Build binary label (0: Benign, 1: Attack)
    df["label_binary"] = (df["label_multiclass"] != "Benign").astype(np.int8)

    # 3. Drop identifier columns and raw label columns from features
    cols_to_drop = [c for c in IDENTIFIER_COLUMNS if c in df.columns]
    for raw_label in RAW_LABEL_COLUMNS:
        if raw_label in df.columns and raw_label not in ("label_multiclass", "label_binary"):
            cols_to_drop.append(raw_label)

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors="ignore")

    # Identify remaining feature columns
    label_cols = ["label_multiclass", "label_binary"]
    feature_cols = [c for c in df.columns if c not in label_cols]

    # 4. Handle inf / -inf and missing values
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    rows_before_na = len(df)
    df = df.dropna(subset=feature_cols + label_cols)
    rows_after_na = len(df)
    metadata["dropped_missing"] = rows_before_na - rows_after_na

    # 5. Cast numeric features strictly to float32 early to save RAM (matches CICIDS2017 cleaner)
    for col in feature_cols:
        df[col] = df[col].astype(np.float32)

    # 6. Label-aware exact de-duplication
    # Only drop rows that are identical across ALL features AND label_multiclass
    rows_before_dedup = len(df)
    df = df.drop_duplicates(subset=feature_cols + ["label_multiclass"], keep="first")
    rows_after_dedup = len(df)
    metadata["dropped_duplicates"] = rows_before_dedup - rows_after_dedup

    # 7. Drop constant columns if requested
    dropped_constant_cols = []
    if drop_const:
        for col in feature_cols:
            # Check if column has zero variance or single unique value
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min == col_max or df[col].nunique(dropna=False) <= 1:
                dropped_constant_cols.append(col)

        if dropped_constant_cols:
            df = df.drop(columns=dropped_constant_cols)
            feature_cols = [c for c in feature_cols if c not in dropped_constant_cols]

    metadata["dropped_constant_columns"] = dropped_constant_cols

    # Reorder columns: features first, then labels
    ordered_columns = feature_cols + label_cols
    df = df[ordered_columns].copy()

    # Metadata summary
    metadata["cleaned_rows"] = len(df)
    metadata["n_features"] = len(feature_cols)
    metadata["feature_columns"] = feature_cols
    metadata["has_destination_port"] = DESTINATION_PORT_COL in feature_cols
    metadata["class_distribution"] = df["label_multiclass"].value_counts().to_dict()
    metadata["binary_distribution"] = {
        int(k): int(v) for k, v in df["label_binary"].value_counts().items()
    }

    del attack_col
    gc.collect()

    return df, metadata


def clean_nf_unsw_dataset(
    input_csv_path: str | Path,
    output_parquet_path: str | Path,
    metadata_json_path: str | Path | None = None,
    rare_threshold: int = 500,
    drop_const: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Load NF-UNSW-NB15-v2 CSV, clean, and save to Parquet + JSON metadata.

    Parameters
    ----------
    input_csv_path : str | Path
        Path to raw NF-UNSW-NB15-v2 CSV file.
    output_parquet_path : str | Path
        Target destination for cleaned Parquet file.
    metadata_json_path : str | Path, optional
        Target destination for metadata JSON file. If None, derives name from Parquet path.
    rare_threshold : int, default=500
        Minimum flow count required for an attack family.
    drop_const : bool, default=True
        Whether to drop constant feature columns.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Cleaned dataframe and metadata dict.
    """
    input_path = Path(input_csv_path)
    output_path = Path(output_parquet_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    # Read CSV
    df_raw = pd.read_csv(input_path)

    # Clean DataFrame
    df_cleaned, metadata = clean_nf_unsw_dataframe(
        df_raw,
        rare_threshold=rare_threshold,
        drop_const=drop_const,
    )

    # Free raw DataFrame memory immediately
    del df_raw
    gc.collect()

    # Ensure output parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to Parquet
    df_cleaned.to_parquet(output_path, engine="pyarrow", index=False)

    # Save metadata to JSON
    if metadata_json_path is None:
        metadata_json_path = output_path.with_name(f"{output_path.stem}_meta.json")
    else:
        metadata_json_path = Path(metadata_json_path)

    metadata_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return df_cleaned, metadata


def _find_default_raw_path() -> Path:
    """Find default path for NF-UNSW-NB15-v2 raw CSV in workspace."""
    candidates = [
        Path("data/raw/nf_unsw_nb15_v2/fe6cb615d161452c_MOHANAD_A4706/data/NF-UNSW-NB15-v2.csv"),
        Path("data/raw/nf_unsw_nb15_v2/NF-UNSW-NB15-v2.csv"),
        Path("/home/hi_nam/project/data/raw/nf_unsw_nb15_v2/fe6cb615d161452c_MOHANAD_A4706/data/NF-UNSW-NB15-v2.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def main() -> None:
    """CLI entrypoint for NF-UNSW-NB15-v2 data cleaning."""
    parser = argparse.ArgumentParser(
        description="Clean NF-UNSW-NB15-v2 dataset according to leakage-free protocol."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=str(_find_default_raw_path()),
        help="Path to raw NF-UNSW-NB15-v2 CSV file.",
    )
    parser.add_argument(
        "--output-parquet",
        type=str,
        default="data/processed/nf_unsw_nb15_v2.parquet",
        help="Path to output cleaned Parquet file.",
    )
    parser.add_argument(
        "--metadata-json",
        type=str,
        default="data/processed/nf_unsw_nb15_v2_meta.json",
        help="Path to output metadata JSON file.",
    )
    parser.add_argument(
        "--rare-threshold",
        type=int,
        default=500,
        help="Minimum flows required for attack family (default: 500).",
    )
    parser.add_argument(
        "--keep-const",
        action="store_true",
        help="If set, do not drop constant columns.",
    )

    args = parser.parse_args()

    print(f"Cleaning NF-UNSW-NB15-v2 from: {args.input_csv}")
    print(f"Target Parquet: {args.output_parquet}")
    print(f"Target Metadata: {args.metadata_json}")

    df_cleaned, metadata = clean_nf_unsw_dataset(
        input_csv_path=args.input_csv,
        output_parquet_path=args.output_parquet,
        metadata_json_path=args.metadata_json,
        rare_threshold=args.rare_threshold,
        drop_const=not args.keep_const,
    )

    print("\n--- Cleaning Complete ---")
    print(f"Raw rows: {metadata['raw_rows']:,}")
    print(f"Dropped missing: {metadata['dropped_missing']:,}")
    print(f"Dropped duplicates: {metadata['dropped_duplicates']:,}")
    print(f"Dropped constant cols: {metadata['dropped_constant_columns']}")
    print(f"Remaining features: {metadata['n_features']}")
    print(f"Cleaned rows: {metadata['cleaned_rows']:,}")
    print("Class distribution:")
    for k, v in sorted(metadata["class_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {k:15s}: {v:,}")


if __name__ == "__main__":
    main()
