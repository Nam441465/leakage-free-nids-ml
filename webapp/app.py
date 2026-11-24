"""FastAPI Web Demo Server for Leakage-Free NIDS Tabular ML.

Launch:
  OMP_WAIT_POLICY=PASSIVE OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
    /home/hi_nam/project/.venv/bin/python -m uvicorn webapp.app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd

from webapp.engine import CACHE_DIR, NIDSDemoEngine, PROJECT_ROOT

STATIC_DIR = Path(__file__).resolve().parent / "static"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

app = FastAPI(
    title="Leakage-Free NIDS ML — SOC & Research Verification Console",
    version="1.0.0",
)

engine = NIDSDemoEngine(max_cached_bundles=6)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if FIGURES_DIR.exists():
    app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/overview")
def api_overview() -> dict[str, Any]:
    return engine.get_overview()


@app.get("/api/predict/sample")
def api_predict_sample(
    dataset: str = Query("cicids2017", pattern="^(cicids2017|nf_unsw)$"),
    task: str = Query("multiclass", pattern="^(binary|multiclass)$"),
    model: str = Query("xgboost", pattern="^(xgboost|lightgbm|random_forest)$"),
    class_filter: str = Query("ALL"),
    only_phase4_fixed: bool = Query(False),
) -> dict[str, Any]:
    try:
        return engine.predict_single_sample(
            dataset=dataset,
            task=task,
            model_name=model,
            class_filter=class_filter,
            only_phase4_fixed=only_phase4_fixed,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/predict/stream")
def api_predict_stream(
    dataset: str = Query("cicids2017", pattern="^(cicids2017|nf_unsw)$"),
    task: str = Query("multiclass", pattern="^(binary|multiclass)$"),
    model: str = Query("xgboost", pattern="^(xgboost|lightgbm|random_forest)$"),
    batch_size: int = Query(8, ge=1, le=30),
) -> dict[str, Any]:
    try:
        return engine.predict_stream_batch(dataset, task, model, batch_size)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/sample-csv/{dataset}")
def api_download_sample_csv(dataset: str) -> FileResponse:
    if dataset not in ("cicids2017", "nf_unsw"):
        raise HTTPException(status_code=400, detail="Unsupported dataset")
    p = CACHE_DIR / f"{dataset}_sample_upload.csv"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Sample CSV not built yet")
    return FileResponse(p, media_type="text/csv", filename=f"{dataset}_sample_50_test_flows.csv")


@app.post("/api/predict/upload-csv")
async def api_upload_csv(
    dataset: str = Form("cicids2017"),
    task: str = Form("multiclass"),
    model: str = Form("xgboost"),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    if dataset not in ("cicids2017", "nf_unsw"):
        raise HTTPException(status_code=400, detail="Invalid dataset")
    if task not in ("binary", "multiclass"):
        raise HTTPException(status_code=400, detail="Invalid task")
    if model not in ("xgboost", "lightgbm", "random_forest"):
        raise HTTPException(status_code=400, detail="Invalid model")
    try:
        content = await file.read()
        df_upload = pd.read_csv(io.BytesIO(content))
        return engine.predict_csv_upload(dataset, task, model, df_upload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/lab/noise")
def api_lab_noise(
    model: str = Query("xgboost", pattern="^(xgboost|lightgbm|random_forest)$"),
    sigma: float = Query(0.1, ge=0.0, le=1.0),
    n_samples: int = Query(5000, ge=500, le=10000),
) -> dict[str, Any]:
    try:
        return engine.run_live_noise_lab(model_name=model, sigma=sigma, n_samples=n_samples)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/lab/lofo")
def api_lab_lofo(
    family: str = Query("DoS", pattern="^(DoS|DDoS|PortScan|BruteForce|Web|Bot)$"),
    seed: int = Query(0, ge=0, le=2),
) -> dict[str, Any]:
    try:
        return engine.run_live_lofo_lab(family=family, seed=seed)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/audit/full")
def api_audit_full(
    dataset: str = Query("nf_unsw", pattern="^(cicids2017|nf_unsw)$"),
    task: str = Query("binary", pattern="^(binary|multiclass)$"),
    model: str = Query("xgboost", pattern="^(xgboost|lightgbm|random_forest)$"),
    seed: int = Query(0, ge=0, le=2),
) -> dict[str, Any]:
    try:
        return engine.run_full_audit(dataset=dataset, task=task, model_name=model, seed=seed)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
