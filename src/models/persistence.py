"""Model, preprocessor, and metadata persistence utilities.

Packages trained Tabular ML models alongside their train-fitted
LeakageFreePreprocessor and seed metadata via joblib (compress=3) so models
can be reloaded for stress tests and live demo evaluation without retraining.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def get_seed_dir(
    models_dir: Path | str,
    dataset: str,
    task: str,
    seed: int,
) -> Path:
    """Return canonical artifact directory: <models_dir>/<dataset>/<task>/seed_<seed>."""
    return Path(models_dir) / str(dataset) / str(task) / f"seed_{int(seed)}"


def save_seed_artifacts(
    models_dir: Path | str,
    dataset: str,
    task: str,
    seed: int,
    preprocessor: Any,
    feature_names: list[str],
    label_mapping: dict[str, int],
    inverse_label_mapping: dict[int, str],
    exclude_destination_port: bool = True,
    compress: int = 3,
) -> Path:
    """Save train-fitted preprocessor and seed metadata into seed_dir.

    Preserves any existing ``models_saved`` list if ``metadata.json`` already exists.
    """
    seed_dir = get_seed_dir(models_dir, dataset, task, seed)
    seed_dir.mkdir(parents=True, exist_ok=True)

    preprocessor_path = seed_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path, compress=compress)

    metadata_path = seed_dir / "metadata.json"
    existing_models_saved: list[str] = []
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            existing_meta = json.load(f)
        if isinstance(existing_meta, dict) and isinstance(
            existing_meta.get("models_saved"), list
        ):
            existing_models_saved = [str(m) for m in existing_meta["models_saved"]]

    metadata: dict[str, Any] = {
        "dataset": str(dataset),
        "task": str(task),
        "seed": int(seed),
        "exclude_destination_port": bool(exclude_destination_port),
        "feature_names": [str(col) for col in feature_names],
        "label_mapping": {str(k): int(v) for k, v in label_mapping.items()},
        "inverse_label_mapping": {
            str(int(k)): str(v) for k, v in inverse_label_mapping.items()
        },
        "models_saved": existing_models_saved,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return seed_dir


def save_model_artifact(
    models_dir: Path | str,
    dataset: str,
    task: str,
    seed: int,
    model_name: str,
    model: Any,
    compress: int = 3,
) -> Path:
    """Save a trained model artifact and register it in metadata.json."""
    seed_dir = get_seed_dir(models_dir, dataset, task, seed)
    seed_dir.mkdir(parents=True, exist_ok=True)

    model_path = seed_dir / f"{model_name}.joblib"
    joblib.dump(model, model_path, compress=compress)

    metadata_path = seed_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {
            "dataset": str(dataset),
            "task": str(task),
            "seed": int(seed),
            "models_saved": [],
        }

    models_saved = list(metadata.get("models_saved", []))
    if model_name not in models_saved:
        models_saved.append(model_name)
    metadata["models_saved"] = models_saved

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return model_path


def load_model_bundle(
    models_dir: Path | str,
    dataset: str,
    task: str,
    seed: int,
    model_name: str,
) -> dict[str, Any]:
    """Load saved model, train-fitted preprocessor, and metadata for a seed."""
    seed_dir = get_seed_dir(models_dir, dataset, task, seed)
    model_path = seed_dir / f"{model_name}.joblib"
    preprocessor_path = seed_dir / "preprocessor.joblib"
    metadata_path = seed_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found: {preprocessor_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if "inverse_label_mapping" in metadata and isinstance(
        metadata["inverse_label_mapping"], dict
    ):
        metadata["inverse_label_mapping"] = {
            int(k): str(v) for k, v in metadata["inverse_label_mapping"].items()
        }

    return {
        "model": model,
        "preprocessor": preprocessor,
        "metadata": metadata,
        "model_path": model_path,
        "preprocessor_path": preprocessor_path,
    }
