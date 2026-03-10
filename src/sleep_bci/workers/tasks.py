"""ARQ task functions for durable background job processing.

These async functions are executed by the ARQ worker process.
They wrap the existing sync preprocessing/training logic and update
job_store for status polling from the API.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import traceback
from datetime import datetime
from functools import partial
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


async def _run_sync(func, *args, **kwargs):
    """Run a blocking function in the default thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))


async def preprocess_task(
    ctx: Dict[str, Any],
    job_id: str,
    raw_dir: str,
    out_dir: str,
    spec_dict: Dict[str, Any],
    combine: bool,
    session_id: Optional[str] = None,
) -> None:
    """ARQ task: run EDF preprocessing pipeline and update job status."""
    from sleep_bci.api.job_store import JobStore
    from sleep_bci.api.schemas import JobStatus, ErrorDetail
    from sleep_bci.preprocessing.core import PreprocessSpec, preprocess_sleep_edf
    from sleep_bci.preprocessing.combine import combine_nights
    from sleep_bci.config import settings

    job_store = JobStore(settings.redis_url)
    SESSION_KEY_PREFIX = "session:"

    job = job_store.get(job_id) or {}
    job["status"] = JobStatus.running
    job["started_at"] = datetime.now()
    job["message"] = "Preprocessing started"
    job["progress"] = 0
    job_store.set(job_id, job)

    # Reconstruct PreprocessSpec (JSON serialization converts tuples to lists)
    spec_dict = dict(spec_dict)
    if isinstance(spec_dict.get("bandpass_hz"), list):
        spec_dict["bandpass_hz"] = tuple(spec_dict["bandpass_hz"])
    spec = PreprocessSpec(**spec_dict)

    def on_progress(idx: int, total: int, night_id: str) -> None:
        scale = 90 if combine else 100
        _job = job_store.get(job_id) or {}
        _job["progress"] = (idx + 1) / total * scale
        _job["message"] = f"Processing night {idx + 1}/{total} ({night_id})"
        job_store.set(job_id, _job)

    try:
        kept, skipped = await _run_sync(
            preprocess_sleep_edf, raw_dir, out_dir, spec, on_progress
        )

        if combine:
            _job = job_store.get(job_id) or {}
            _job["progress"] = 90
            _job["message"] = "Combining nights..."
            job_store.set(job_id, _job)
            combined_path = os.path.join(out_dir, "sleep_edf_all.npz")
            await _run_sync(combine_nights, out_dir, combined_path)

        _job = job_store.get(job_id) or {}
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

        logger.info(
            "preprocess_task %s succeeded: kept=%d skipped=%d out_dir=%s",
            job_id, kept, skipped, out_dir,
        )

    except Exception as e:
        _job = job_store.get(job_id) or {}
        _job["status"] = JobStatus.failed
        _job["finished_at"] = datetime.now()
        _job["error"] = ErrorDetail(
            code="PREPROCESSING_ERROR",
            message=str(e),
            details={"traceback": traceback.format_exc()},
        )
        _job["message"] = f"Failed: {e}"
        job_store.set(job_id, _job)
        logger.error("preprocess_task %s failed: %s", job_id, e)

    finally:
        UPLOAD_PREFIX = "sleep-bci-upload-"
        upload_dir_prefix = os.path.join(tempfile.gettempdir(), UPLOAD_PREFIX)
        if raw_dir.startswith(upload_dir_prefix):
            try:
                shutil.rmtree(raw_dir)
                logger.info("Cleaned up temp upload dir: %s", raw_dir)
            except Exception as cleanup_err:
                logger.warning(
                    "Failed to clean up temp dir %s: %s", raw_dir, cleanup_err
                )


async def train_task(
    ctx: Dict[str, Any],
    train_id: str,
    npz_directory: str,
    model_out: str,
    fs: float,
    n_splits: int,
    model_type: str = "lda",
) -> None:
    """ARQ task: train a sleep-stage classifier and update job status."""
    from sleep_bci.api.job_store import JobStore
    from sleep_bci.api.schemas import JobStatus, ErrorDetail
    from sleep_bci.model.train import load_nightly_npz, train_classifier
    from sleep_bci.model.artifacts import save_bundle
    from sleep_bci.config import settings

    job_store = JobStore(settings.redis_url)

    job = job_store.get(train_id) or {}
    job["status"] = JobStatus.running
    job["started_at"] = datetime.now()
    job["message"] = f"Training {model_type} classifier"
    job_store.set(train_id, job)

    logger.info(
        "train_task %s started — model_type=%s npz_dir=%s",
        train_id, model_type, npz_directory,
    )

    try:
        X, Y, night_ids = await _run_sync(load_nightly_npz, npz_directory)
        logger.info(
            "train_task %s: loaded %d epochs from %d nights",
            train_id, X.shape[0], len(set(night_ids)),
        )

        out_dir = os.path.dirname(model_out)
        bundle, results = await _run_sync(
            train_classifier, X, Y, night_ids, fs, n_splits, model_type, out_dir
        )

        await _run_sync(save_bundle, model_out, bundle)

        results_path = os.path.join(out_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        _job = job_store.get(train_id) or {}
        _job["status"] = JobStatus.succeeded
        _job["progress"] = 100
        _job["finished_at"] = datetime.now()
        _job["output_location"] = out_dir
        _job["message"] = f"Training complete ({model_type})"
        _job["results"] = results
        job_store.set(train_id, _job)

        logger.info("train_task %s succeeded: model_out=%s", train_id, model_out)

    except Exception as e:
        _job = job_store.get(train_id) or {}
        _job["status"] = JobStatus.failed
        _job["finished_at"] = datetime.now()
        _job["error"] = ErrorDetail(
            code="TRAINING_ERROR",
            message=str(e),
            details={"traceback": traceback.format_exc()},
        )
        _job["message"] = f"Failed: {e}"
        job_store.set(train_id, _job)
        logger.error("train_task %s failed: %s", train_id, e)
