"""Sleep-BCI FastAPI application.

Endpoints:
  POST /v1/upload          — Upload EDF files (PSG + Hypnogram)
  POST /v1/preprocess      — Launch preprocessing job
  GET  /v1/preprocess/{id} — Poll preprocessing status
  POST /v1/train           — Launch training job
  GET  /v1/train/{id}      — Poll training status
  GET  /v1/models          — List trained models
  GET  /v1/models/{id}/metrics — Retrieve training metrics
  WS   /v1/stream          — Real-time chunk-based inference
  GET  /v1/health          — Service health + uptime
  GET  /health             — Alias for /v1/health (backward compat)
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

import numpy as np
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware

from sleep_bci.api.job_store import JobStore
from sleep_bci.api.schemas import (
    DryRunFileInfo,
    DryRunResponse,
    ErrorDetail,
    HealthResponse,
    JobCreatedResponse,
    JobStatus,
    ModelMetricsResponse,
    ModelResponse,
    PreprocessingConfig,
    PreprocessRequest,
    PreprocessStatusResponse,
    TrainConfigRequest,
    TrainingJobCreated,
    TrainingStatusResponse,
    UploadResponse,
)
from sleep_bci.config import settings
from sleep_bci.features.bandpower import extract_features_epoch
from sleep_bci.model.artifacts import load_bundle
from sleep_bci.model.train import load_nightly_npz, train_classifier
from sleep_bci.model.artifacts import save_bundle
from sleep_bci.preprocessing.combine import combine_nights
from sleep_bci.preprocessing.core import (
    PreprocessSpec,
    discover_and_validate,
    preprocess_sleep_edf,
)
from sleep_bci.stream.buffer import EpochBuffer

# ─────────────────────────────────────────────────────────────────────────────
# Logging — structured JSON output
# ─────────────────────────────────────────────────────────────────────────────

class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("endpoint", "method", "status_code", "duration_ms", "job_id"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        if record.exc_info:
            log_entry["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def _configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_JSONFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, settings.logger.upper(), logging.INFO))


_configure_logging()
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

UPLOAD_PREFIX = "sleep-bci-upload-"
OUTPUT_PREFIX = "sleep-bci-output-"
SESSION_KEY_PREFIX = "session:"

APP_START_TIME = time.monotonic()

# ─────────────────────────────────────────────────────────────────────────────
# ARQ pool (optional — falls back to BackgroundTasks when unavailable)
# ─────────────────────────────────────────────────────────────────────────────

_arq_pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _arq_pool
    if settings.redis_url:
        try:
            from arq import create_pool
            from arq.connections import RedisSettings

            _arq_pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
            logger.info("ARQ pool connected to Redis: %s", settings.redis_url)
        except Exception as exc:
            logger.warning(
                "ARQ unavailable (%s) — falling back to in-process BackgroundTasks", exc
            )
    yield
    if _arq_pool:
        await _arq_pool.close()
        logger.info("ARQ pool closed")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sleep-BCI API",
    description="EEG sleep-stage classification: preprocessing, training, and real-time inference.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Middleware — request ID + latency
# ─────────────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def request_middleware(request: Request, call_next) -> Response:
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    t0 = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "%s %s",
        request.method,
        request.url.path,
        extra={
            "endpoint": str(request.url.path),
            "method": request.method,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "request_id": request_id,
        },
    )
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────────────────────────────────────────

job_store = JobStore(settings.redis_url)

MODEL_CACHE: dict = {}

# Per-session EpochBuffers for chunk-based WebSocket streaming
_session_buffers: dict[str, EpochBuffer] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/v1/health", response_model=HealthResponse)
async def v1_health():
    return HealthResponse(
        status="ok",
        uptime_s=round(time.monotonic() - APP_START_TIME, 2),
        redis_connected=job_store.backend == "redis",
        model_loaded=len(MODEL_CACHE) > 0,
        job_store_backend=job_store.backend,
    )


@app.get("/health")
async def health():
    """Alias for /v1/health (used by Docker healthcheck)."""
    return await v1_health()


@app.get("/")
def read_root():
    return {"message": "Sleep-BCI API is running. See /docs for API reference."}


# ─────────────────────────────────────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/v1/upload", response_model=UploadResponse)
async def upload_edf_files(files: List[UploadFile] = File(...)):
    for f in files:
        if not f.filename or not f.filename.lower().endswith(".edf"):
            raise HTTPException(
                status_code=400,
                detail=f"File '{f.filename}' is not an EDF file. All files must have a .edf extension.",
            )

    filenames = [f.filename for f in files]
    has_psg = any(name.endswith("PSG.edf") for name in filenames)
    has_hyp = any(name.endswith("Hypnogram.edf") for name in filenames)

    if not has_psg:
        raise HTTPException(
            status_code=400,
            detail="No PSG file found. At least one file matching '*PSG.edf' is required.",
        )
    if not has_hyp:
        raise HTTPException(
            status_code=400,
            detail="No Hypnogram file found. At least one file matching '*Hypnogram.edf' is required.",
        )

    tmp_dir = tempfile.mkdtemp(prefix=UPLOAD_PREFIX)

    try:
        for f in files:
            dest = os.path.join(tmp_dir, f.filename)
            with open(dest, "wb") as out:
                while chunk := await f.read(1024 * 1024):
                    out.write(chunk)

        pairs = discover_and_validate(tmp_dir)

        file_info = [
            DryRunFileInfo(
                psg_file=os.path.basename(psg),
                hypnogram_file=os.path.basename(hyp),
            )
            for psg, hyp in pairs
        ]

        session_id = str(uuid.uuid4())
        job_store.set(f"{SESSION_KEY_PREFIX}{session_id}", {"raw_dir": tmp_dir})

        return UploadResponse(
            session_id=session_id,
            raw_dir=tmp_dir,
            matched_pairs=len(pairs),
            files=file_info,
            message=f"Uploaded {len(files)} files. {len(pairs)} PSG/Hypnogram pairs matched.",
        )
    except HTTPException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    except (FileNotFoundError, ValueError) as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _build_spec(config: PreprocessingConfig) -> PreprocessSpec:
    return PreprocessSpec(
        channel=config.channel,
        epoch_sec=config.epochs,
        bandpass_hz=tuple(config.bandpass),
        notch_hz=config.notch,
    )


def run_preprocess_job(
    job_id: str,
    raw_dir: str,
    out_dir: str,
    spec: PreprocessSpec,
    combine: bool,
    session_id: Optional[str] = None,
) -> None:
    """Synchronous fallback executed via BackgroundTasks when ARQ is unavailable."""
    job = job_store.get(job_id)
    job["status"] = JobStatus.running
    job["started_at"] = datetime.now()
    job["message"] = "Preprocessing started"
    job["progress"] = 0
    job_store.set(job_id, job)

    def on_progress(idx: int, total: int, night_id: str) -> None:
        scale = 90 if combine else 100
        _job = job_store.get(job_id)
        _job["progress"] = (idx + 1) / total * scale
        _job["message"] = f"Processing night {idx + 1}/{total} ({night_id})"
        job_store.set(job_id, _job)

    try:
        kept, skipped = preprocess_sleep_edf(raw_dir, out_dir, spec, on_progress=on_progress)

        if combine:
            _job = job_store.get(job_id)
            _job["progress"] = 90
            _job["message"] = "Combining nights..."
            job_store.set(job_id, _job)
            combine_nights(out_dir, os.path.join(out_dir, "sleep_edf_all.npz"))

        _job = job_store.get(job_id)
        _job["status"] = JobStatus.succeeded
        _job["progress"] = 100
        _job["message"] = f"Done. Kept {kept}, skipped {skipped}."
        _job["finished_at"] = datetime.now()
        _job["output_location"] = out_dir
        job_store.set(job_id, _job)

        if session_id:
            session_data = job_store.get(f"{SESSION_KEY_PREFIX}{session_id}") or {}
            session_data["npz_dir"] = out_dir
            job_store.set(f"{SESSION_KEY_PREFIX}{session_id}", session_data)

        logger.info("Preprocessing %s succeeded: kept=%d skipped=%d", job_id, kept, skipped)

    except Exception as e:
        _job = job_store.get(job_id)
        _job["status"] = JobStatus.failed
        _job["finished_at"] = datetime.now()
        _job["error"] = ErrorDetail(
            code="PREPROCESSING_ERROR",
            message=str(e),
            details={"traceback": traceback.format_exc()},
        )
        _job["message"] = f"Failed: {e}"
        job_store.set(job_id, _job)
        logger.error("Preprocessing %s failed: %s", job_id, e)
    finally:
        upload_dir_prefix = os.path.join(tempfile.gettempdir(), UPLOAD_PREFIX)
        if raw_dir.startswith(upload_dir_prefix):
            try:
                shutil.rmtree(raw_dir)
                logger.info("Cleaned up temp upload dir: %s", raw_dir)
            except Exception as cleanup_err:
                logger.warning("Failed to clean up %s: %s", raw_dir, cleanup_err)


@app.post("/v1/preprocess")
async def preprocess_data(request: PreprocessRequest, background_task: BackgroundTasks):
    if request.session_id:
        session_data = job_store.get(f"{SESSION_KEY_PREFIX}{request.session_id}")
        if session_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{request.session_id}' not found. Upload files first via /v1/upload.",
            )
        raw_dir = session_data["raw_dir"]
    elif request.dataset:
        raw_dir = request.dataset.raw_dir
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'session_id' (from /v1/upload) or 'dataset.raw_dir'.",
        )

    out_dir = request.output.out_dir
    combine = request.output.combine

    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix=OUTPUT_PREFIX)

    if not os.path.exists(raw_dir):
        raise HTTPException(status_code=400, detail="Raw directory path not found.")

    spec = _build_spec(request.preprocessing_config)

    try:
        pairs = discover_and_validate(raw_dir, spec)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    if request.dry_run:
        return DryRunResponse(
            valid=True,
            matched_pairs=len(pairs),
            files=[
                DryRunFileInfo(
                    psg_file=os.path.basename(psg),
                    hypnogram_file=os.path.basename(hyp),
                )
                for psg, hyp in pairs
            ],
            raw_dir=raw_dir,
            out_dir=out_dir,
            preprocessing_config=request.preprocessing_config,
            message=f"Validation passed. {len(pairs)} PSG/Hypnogram pairs matched.",
        )

    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"Cannot create output directory: {e}")

    job_id = str(uuid.uuid4())
    job_store.set(job_id, {
        "status": JobStatus.queued,
        "created_at": datetime.now(),
        "progress": 0,
        "raw_dir": raw_dir,
        "out_dir": out_dir,
        "started_at": None,
        "finished_at": None,
        "message": None,
        "output_location": out_dir,
        "error": None,
    })

    if _arq_pool:
        spec_dict = {
            "channel": spec.channel,
            "epoch_sec": spec.epoch_sec,
            "bandpass_hz": list(spec.bandpass_hz),
            "notch_hz": spec.notch_hz,
            "target_sfreq": spec.target_sfreq,
        }
        await _arq_pool.enqueue_job(
            "preprocess_task",
            job_id, raw_dir, out_dir, spec_dict, combine, request.session_id,
            _job_id=f"preprocess-{job_id}",
        )
        logger.info("Preprocessing %s enqueued via ARQ", job_id)
    else:
        background_task.add_task(
            run_preprocess_job, job_id, raw_dir, out_dir, spec, combine, request.session_id
        )
        logger.info("Preprocessing %s dispatched via BackgroundTasks", job_id)

    return JobCreatedResponse(
        job_id=job_id,
        status=JobStatus.queued,
        status_url=f"/v1/preprocess/{job_id}",
    )


@app.get("/v1/preprocess/{job_id}")
def get_preprocessing_status(job_id: str):
    if not job_store.exists(job_id):
        raise HTTPException(status_code=404, detail="job_id not found")
    job = job_store.get(job_id)
    return PreprocessStatusResponse(
        job_id=job_id,
        status=job["status"],
        created_at=job["created_at"],
        started_at=job["started_at"],
        finished_at=job["finished_at"],
        progress=job["progress"],
        message=job["message"],
        output_location=job["output_location"],
        error=job["error"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_model_path(model_id: str) -> str:
    models_dir = os.path.dirname(os.path.abspath(settings.model_path))
    return os.path.join(models_dir, f"{model_id}.joblib")


def _load_model_cached(model_id: str):
    if model_id not in MODEL_CACHE:
        path = _resolve_model_path(model_id)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        MODEL_CACHE[model_id] = load_bundle(path)
    return MODEL_CACHE[model_id]


@app.get("/v1/models", response_model=ModelResponse)
def models_list():
    try:
        models_dir = os.path.dirname(os.path.abspath(settings.model_path))
        models = [
            f.removesuffix(".joblib")
            for f in os.listdir(models_dir)
            if f.endswith(".joblib")
        ]
        return ModelResponse(models=models, num_models=len(models))
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Models directory not found. Error: {e}",
        )


@app.get("/v1/models/{model_id}/metrics", response_model=ModelMetricsResponse)
def get_model_metrics(model_id: str):
    model_path = _resolve_model_path(model_id)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found.")

    model_dir = os.path.dirname(model_path)
    results_path = os.path.join(model_dir, "results.json")

    if not os.path.exists(results_path):
        raise HTTPException(
            status_code=404,
            detail=f"Metrics not found for model '{model_id}'. Train with /v1/train to generate results.json.",
        )

    with open(results_path) as f:
        data = json.load(f)

    return ModelMetricsResponse(
        model_type=data.get("model_type"),
        total_epochs=data.get("total_epochs"),
        total_nights=data.get("total_nights"),
        overall=data.get("overall"),
        folds=data.get("folds"),
        training_config=data.get("training_config"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _run_training_job(
    train_id: str,
    npz_directory: str,
    model_out: str,
    fs: float,
    n_splits: int,
    model_type: str = "lda",
) -> None:
    """Synchronous fallback executed via BackgroundTasks when ARQ is unavailable."""
    job = job_store.get(train_id)
    job["status"] = JobStatus.running
    job["started_at"] = datetime.now()
    job["message"] = f"Training {model_type} classifier"
    job_store.set(train_id, job)
    logger.info(
        "Training %s started — model_type=%s npz_dir=%s", train_id, model_type, npz_directory
    )

    try:
        X, Y, night_ids = load_nightly_npz(npz_directory)
        logger.info(
            "Training %s: loaded %d epochs from %d nights",
            train_id, X.shape[0], len(set(night_ids)),
        )

        out_dir = os.path.dirname(model_out)
        bundle, results = train_classifier(
            X, Y, night_ids, fs=fs, n_splits=n_splits, model_type=model_type, out_dir=out_dir
        )

        save_bundle(model_out, bundle)
        results_path = os.path.join(out_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        _job = job_store.get(train_id)
        _job["status"] = JobStatus.succeeded
        _job["progress"] = 100
        _job["finished_at"] = datetime.now()
        _job["output_location"] = out_dir
        _job["message"] = f"Training complete ({model_type})"
        _job["results"] = results
        job_store.set(train_id, _job)
        logger.info("Training %s succeeded: model_out=%s", train_id, model_out)

    except Exception as e:
        _job = job_store.get(train_id)
        _job["status"] = JobStatus.failed
        _job["finished_at"] = datetime.now()
        _job["error"] = ErrorDetail(
            code="TRAINING_ERROR",
            message=str(e),
            details={"traceback": traceback.format_exc()},
        )
        _job["message"] = f"Failed: {e}"
        job_store.set(train_id, _job)
        logger.error("Training %s failed: %s", train_id, e)


@app.post("/v1/train")
async def train_model(request: TrainConfigRequest, background_task: BackgroundTasks):
    if request.session_id:
        session_data = job_store.get(f"{SESSION_KEY_PREFIX}{request.session_id}")
        if session_data is None:
            raise HTTPException(
                status_code=404, detail=f"Session '{request.session_id}' not found."
            )
        npz_dir = session_data.get("npz_dir")
        if npz_dir is None:
            raise HTTPException(
                status_code=400,
                detail="Preprocessing has not completed for this session. Wait for the preprocess job to finish.",
            )
    elif request.npz_dir:
        npz_dir = request.npz_dir
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'session_id' (after preprocessing) or 'npz_dir'.",
        )

    if not os.path.exists(npz_dir):
        raise HTTPException(status_code=400, detail="NPZ directory not found.")

    import glob as _glob
    night_files = [
        f for f in _glob.glob(os.path.join(npz_dir, "*.npz"))
        if "sleep_edf_all" not in os.path.basename(f)
    ]
    if len(night_files) == 0:
        raise HTTPException(status_code=400, detail="No .npz night files found in npz_dir.")
    if len(night_files) < request.n_splits:
        raise HTTPException(
            status_code=400,
            detail=(
                f"n_splits={request.n_splits} requires at least {request.n_splits} nights, "
                f"but only {len(night_files)} found."
            ),
        )

    if request.model_out is None:
        model_out = os.path.join(tempfile.mkdtemp(prefix=OUTPUT_PREFIX), "model.joblib")
    else:
        model_out = request.model_out

    model_type = request.model_type.value if hasattr(request.model_type, "value") else str(request.model_type)
    train_id = str(uuid.uuid4())

    job_store.set(train_id, {
        "npz_dir": npz_dir,
        "status": JobStatus.queued,
        "model_out": model_out,
        "model_type": model_type,
        "fs": request.fs,
        "n_splits": request.n_splits,
        "created_at": datetime.now(),
        "progress": 0,
        "started_at": None,
        "finished_at": None,
        "message": None,
        "output_location": model_out,
        "error": None,
    })

    if _arq_pool:
        await _arq_pool.enqueue_job(
            "train_task",
            train_id, npz_dir, model_out, request.fs, request.n_splits, model_type,
            _job_id=f"train-{train_id}",
        )
        logger.info("Training %s enqueued via ARQ (model_type=%s)", train_id, model_type)
    else:
        background_task.add_task(
            _run_training_job, train_id, npz_dir, model_out, request.fs, request.n_splits, model_type
        )
        logger.info("Training %s dispatched via BackgroundTasks (model_type=%s)", train_id, model_type)

    return TrainingJobCreated(
        job_id=train_id,
        status=JobStatus.queued,
        status_url=f"/v1/train/{train_id}",
    )


@app.get("/v1/train/{train_id}")
def get_training_status(train_id: str):
    if not job_store.exists(train_id):
        raise HTTPException(status_code=404, detail="train_id not found")
    job = job_store.get(train_id)
    return TrainingStatusResponse(
        npz_dir=job["npz_dir"],
        job_id=train_id,
        status=job["status"],
        created_at=job["created_at"],
        started_at=job["started_at"],
        finished_at=job["finished_at"],
        progress=job["progress"],
        message=job["message"],
        output_location=job["output_location"],
        results=job.get("results"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket streaming — chunk-based real-time inference
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/v1/stream")
async def stream_inference(websocket: WebSocket, model_id: str):
    """Real-time EEG sleep-stage inference via WebSocket.

    Chunk-based protocol (recommended):
      Client sends: {"chunk": [<float>, ...], "fs": 100.0, "session_id": "<id>"}
      Server replies (buffering): {"buffered": 250, "needed": 2750}
      Server replies (on full epoch): {"stage": "N2", "confidence": 0.87, "latency_ms": 4.2, "epoch_idx": 0}

    Legacy full-epoch protocol (backward compatible):
      Client sends: {"epoch": [<3000 floats>], "timestamp": <optional float>}
      Server replies: {"stage": "N2", "confidence": 0.87, "latency_ms": 4.2}
    """
    await websocket.accept()
    logger.info("WebSocket /v1/stream connected: model_id=%s", model_id)

    try:
        bundle = _load_model_cached(model_id)
    except FileNotFoundError:
        await websocket.send_json({
            "error": "MODEL_NOT_FOUND",
            "message": f"Model '{model_id}' not found. Train a model first via POST /v1/train.",
        })
        await websocket.close(code=1008)
        return

    expected_samples = int(bundle.fs * 30)  # 3000 at 100 Hz
    active_session_id: Optional[str] = None

    try:
        while True:
            data = await websocket.receive_json()

            if "chunk" in data:
                # ── Chunk-based streaming (realistic) ──────────────────────
                chunk = data["chunk"]
                session_id = data.get("session_id", str(id(websocket)))
                active_session_id = session_id

                if session_id not in _session_buffers:
                    _session_buffers[session_id] = EpochBuffer(
                        epoch_samples=expected_samples
                    )

                buf = _session_buffers[session_id]
                epoch = buf.push(chunk)

                if epoch is None:
                    await websocket.send_json({
                        "buffered": buf.buffered_samples,
                        "needed": buf.samples_needed,
                    })
                    continue

                epoch_idx = buf.epoch_count - 1

            elif "epoch" in data:
                # ── Legacy full-epoch format (backward compatible) ──────────
                epoch_data = data.get("epoch")
                if not isinstance(epoch_data, list) or len(epoch_data) != expected_samples:
                    got = len(epoch_data) if isinstance(epoch_data, list) else 0
                    await websocket.send_json({
                        "error": "INVALID_EPOCH_LENGTH",
                        "message": f"Expected {expected_samples} samples, got {got}.",
                    })
                    continue
                epoch = np.array(epoch_data, dtype=np.float32)
                epoch_idx = 0

            else:
                await websocket.send_json({
                    "error": "INVALID_MESSAGE",
                    "message": "Message must contain 'chunk' (streaming) or 'epoch' (full-epoch).",
                })
                continue

            # ── Inference ───────────────────────────────────────────────────
            t0 = time.perf_counter()
            try:
                features = extract_features_epoch(epoch, fs=bundle.fs).reshape(1, -1)
                proba = bundle.pipeline.predict_proba(features)[0]
                label = int(np.argmax(proba))
                confidence = float(np.max(proba))
                stage = bundle.label_map[label]
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)

                response = {
                    "stage": stage,
                    "label": label,
                    "confidence": round(confidence, 4),
                    "latency_ms": latency_ms,
                    "epoch_idx": epoch_idx,
                }
                if "timestamp" in data:
                    response["timestamp"] = data["timestamp"]

                await websocket.send_json(response)

            except Exception as e:
                logger.exception("Prediction failed for model_id=%s", model_id)
                await websocket.send_json({
                    "error": "PREDICTION_FAILED",
                    "message": str(e),
                })

    except WebSocketDisconnect:
        logger.info("WebSocket /v1/stream disconnected: model_id=%s", model_id)
        if active_session_id and active_session_id in _session_buffers:
            del _session_buffers[active_session_id]
            logger.info("Cleaned up EpochBuffer for session: %s", active_session_id)
