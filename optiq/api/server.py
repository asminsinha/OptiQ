"""FastAPI backend server for OptiQ research platform."""

import json
import os
import threading
from typing import Any, Dict, List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from optiq.cli.main import run_full_experiment

app = FastAPI(
    title="OptiQ CPU Scheduling Framework API",
    version="2.0.0",
    description="REST API exposing real simulation execution, benchmarks, ML diagnostics, and telemetry."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles

OUTPUTS_DIR = os.path.abspath("outputs")
WEB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))

# In-memory background task registry
experiment_jobs: Dict[str, Dict[str, Any]] = {}
lock = threading.Lock()


class ExperimentRequest(BaseModel):
    source: str = Field(default="synthetic", pattern="^(real|synthetic)$")
    jobs: int = Field(default=100, ge=10, le=2000)
    quantum: float = Field(default=30.0, ge=1.0, le=1000.0)
    aging: float = Field(default=60.0, ge=0.0, le=3600.0)
    starvation: float = Field(default=300.0, ge=10.0, le=10000.0)
    seed: int = Field(default=42, ge=0, le=999999)
    profile: str = Field(default="mixed", pattern="^(mixed|short_heavy|long_heavy)$")


def _execute_experiment_task(job_id: str, req: ExperimentRequest):
    with lock:
        experiment_jobs[job_id]["status"] = "running"
    try:
        run_path = run_full_experiment(
            data_source=req.source,
            n_jobs=req.jobs,
            quantum=req.quantum,
            aging_interval=req.aging,
            starvation_threshold=req.starvation,
            seed=req.seed,
            synthetic_profile=req.profile,
        )
        run_name = os.path.basename(run_path)
        with lock:
            experiment_jobs[job_id]["status"] = "completed"
            experiment_jobs[job_id]["run_id"] = run_name
    except Exception as e:
        with lock:
            experiment_jobs[job_id]["status"] = "failed"
            experiment_jobs[job_id]["error"] = str(e)


@app.get("/")
def serve_index():
    index_file = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "OptiQ API Running. Frontend index.html not found."}


if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "service": "OptiQ Framework Backend",
        "version": "2.0.0",
        "outputs_count": len(os.listdir(OUTPUTS_DIR)) if os.path.exists(OUTPUTS_DIR) else 0
    }


@app.get("/api/runs")
def list_runs():
    """List all available runs sorted newest to oldest."""
    if not os.path.exists(OUTPUTS_DIR):
        return []
    runs = []
    for item in os.listdir(OUTPUTS_DIR):
        p = os.path.join(OUTPUTS_DIR, item)
        if os.path.isdir(p):
            meta_path = os.path.join(p, "experiment_metadata.json")
            meta = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                except Exception:
                    pass
            runs.append({
                "run_id": item,
                "is_historical": "preserved" in item,
                "timestamp": meta.get("timestamp", ""),
                "parameters": meta.get("parameters", {}),
                "dataset": meta.get("dataset", {}),
            })
    # Sort descending by run_id
    runs.sort(key=lambda r: r["run_id"], reverse=True)
    return runs


@app.get("/api/runs/{run_id}/summary")
def get_run_summary(run_id: str):
    run_dir = os.path.join(OUTPUTS_DIR, run_id)
    if not os.path.isdir(run_dir):
        raise HTTPException(status_code=404, detail="Run not found")

    meta_file = os.path.join(run_dir, "experiment_metadata.json")
    meta = {}
    if os.path.exists(meta_file):
        with open(meta_file, "r") as f:
            meta = json.load(f)

    # Comparison metrics
    comp_file = os.path.join(run_dir, "algorithm_comparison.csv")
    comparison = []
    if os.path.exists(comp_file):
        comparison = pd.read_csv(comp_file).to_dict(orient="records")

    # Ablation metrics
    ablation_file = os.path.join(run_dir, "ablation_study.csv")
    ablation = []
    if os.path.exists(ablation_file):
        ablation = pd.read_csv(ablation_file).to_dict(orient="records")

    # ML metrics
    ml_file = os.path.join(run_dir, "model_metrics.csv")
    ml_metrics = []
    if os.path.exists(ml_file):
        ml_metrics = pd.read_csv(ml_file).to_dict(orient="records")

    return {
        "run_id": run_id,
        "metadata": meta,
        "comparison": comparison,
        "ablation": ablation,
        "ml_metrics": ml_metrics,
    }


@app.get("/api/runs/{run_id}/trace")
def get_run_trace(run_id: str, limit: int = 100):
    run_dir = os.path.join(OUTPUTS_DIR, run_id)
    trace_file = os.path.join(run_dir, "optiq_scheduling_trace.csv")
    if not os.path.exists(trace_file):
        raise HTTPException(status_code=404, detail="Trace file not found")
    df = pd.read_csv(trace_file)
    total_records = len(df)
    records = df.head(limit).to_dict(orient="records")
    return {
        "run_id": run_id,
        "total_records": total_records,
        "limit": limit,
        "trace": records,
    }


@app.get("/api/runs/{run_id}/file/{filename}")
def download_file(run_id: str, filename: str):
    safe_name = os.path.basename(filename)
    file_path = os.path.join(OUTPUTS_DIR, run_id, safe_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=safe_name)


@app.post("/api/simulate")
def trigger_simulation(req: ExperimentRequest, background_tasks: BackgroundTasks):
    import uuid
    job_id = str(uuid.uuid4())[:8]
    with lock:
        experiment_jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "request": req.dict(),
            "run_id": None,
            "error": None,
        }
    background_tasks.add_task(_execute_experiment_task, job_id, req)
    return {"job_id": job_id, "status": "pending"}


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    with lock:
        job = experiment_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
