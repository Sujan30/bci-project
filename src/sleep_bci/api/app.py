import logging
import os
import shutil
import tempfile
import traceback
import uuid
from datetime import datetime
from typing import List

from fastapi import FastAPI, File, HTTPException, BackgroundTasks, UploadFile

from schemas import (
    PreprocessStatusResponse,
    ErrorDetail,
    JobCreatedResponse,
    PreprocessRequest,
    PreprocessingConfig,
    JobStatus,
    DryRunFileInfo,
    DryRunResponse,
    UploadResponse,
)
from sleep_bci.preprocessing.core import (
    PreprocessSpec,
    preprocess_sleep_edf,
    discover_and_validate,
)
from sleep_bci.preprocessing.combine import combine_nights

logger = logging.getLogger(__name__)

UPLOAD_PREFIX = "sleep-bci-upload-"
OUTPUT_PREFIX = "sleep-bci-output-"

app = FastAPI()


"""""
Current API endpoints:
1. endpoint to preprocess data
2. endpoint to upload new data to help calibrate the model
3. /stream so we can use the model that we trained
4. /epochs to basically split the data into epochs
5. /train to train the data
6. /change-feature to basically play around with the feature



"""

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.post("/upload", response_model=UploadResponse)
async def upload_edf_files(files: List[UploadFile] = File(...)):
    # 1. Validate all filenames end with .edf
    for f in files:
        if not f.filename or not f.filename.lower().endswith(".edf"):
            raise HTTPException(
                status_code=400,
                detail=f"File '{f.filename}' is not an EDF file. All files must have a .edf extension.",
            )

    filenames = [f.filename for f in files]

    # 2. Validate at least one PSG and one Hypnogram file present
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

    # 3. Create temp directory
    tmp_dir = tempfile.mkdtemp(prefix=UPLOAD_PREFIX)

    try:
        # 4. Save each uploaded file (stream in chunks for large PSG files)
        for f in files:
            dest = os.path.join(tmp_dir, f.filename)
            with open(dest, "wb") as out:
                while chunk := await f.read(1024 * 1024):  # 1 MB chunks
                    out.write(chunk)

        # 5. Discover and validate pairs
        pairs = discover_and_validate(tmp_dir)

        # 6. Build response
        file_info = [
            DryRunFileInfo(
                psg_file=os.path.basename(psg),
                hypnogram_file=os.path.basename(hyp),
            )
            for psg, hyp in pairs
        ]

        return UploadResponse(
            session_id=str(uuid.uuid4()),
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


JOBS = {}


def _build_spec(config: PreprocessingConfig) -> PreprocessSpec:
    """Map API PreprocessingConfig to core PreprocessSpec."""
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
):
    job = JOBS[job_id]
    job["status"] = JobStatus.running
    job["started_at"] = datetime.now()
    job["message"] = "Preprocessing started"
    job["progress"] = 0

    def on_progress(idx: int, total: int, night_id: str) -> None:
        scale = 90 if combine else 100
        job["progress"] = (idx + 1) / total * scale
        job["message"] = f"Processing night {idx + 1}/{total} ({night_id})"

    try:
        kept, skipped = preprocess_sleep_edf(
            raw_dir, out_dir, spec, on_progress=on_progress
        )

        if combine:
            job["progress"] = 90
            job["message"] = "Combining nights..."
            combined_path = os.path.join(out_dir, "sleep_edf_all.npz")
            combine_nights(out_dir, combined_path)

        job["status"] = JobStatus.succeeded
        job["progress"] = 100
        job["message"] = f"Done. Kept {kept}, skipped {skipped}."
        job["finished_at"] = datetime.now()
        job["output_location"] = out_dir
    except Exception as e:
        job["status"] = JobStatus.failed
        job["finished_at"] = datetime.now()
        job["error"] = ErrorDetail(
            code="PREPROCESSING_ERROR",
            message=str(e),
            details={"traceback": traceback.format_exc()},
        )
        job["message"] = f"Failed: {e}"
    finally:
        # Clean up temp upload directories after processing
        upload_dir_prefix = os.path.join(tempfile.gettempdir(), UPLOAD_PREFIX)
        if raw_dir.startswith(upload_dir_prefix):
            try:
                shutil.rmtree(raw_dir)
                logger.info("Cleaned up temp upload dir: %s", raw_dir)
            except Exception as cleanup_err:
                logger.warning("Failed to clean up temp dir %s: %s", raw_dir, cleanup_err)


#pre process the dataset, and return a job id so we can keep track of it
@app.post("/preprocess")
def preprocess_data(request: PreprocessRequest, background_task: BackgroundTasks):
    raw_dir = request.dataset.raw_dir
    out_dir = request.output.out_dir
    combine = request.output.combine

    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix=OUTPUT_PREFIX)

    # --- Validation ---

    # 1. Check raw directory exists
    if not os.path.exists(raw_dir):
        raise HTTPException(status_code=400, detail="Raw directory path not found")

    # 2. Build spec from config
    spec = _build_spec(request.preprocessing_config)

    # 3. Discover and validate pairs (also validates PSG/Hyp files exist)
    try:
        pairs = discover_and_validate(raw_dir, spec)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 4. Dry-run: return validation results immediately
    if request.dry_run:
        files = [
            DryRunFileInfo(
                psg_file=os.path.basename(psg),
                hypnogram_file=os.path.basename(hyp),
            )
            for psg, hyp in pairs
        ]
        return DryRunResponse(
            valid=True,
            matched_pairs=len(pairs),
            files=files,
            raw_dir=raw_dir,
            out_dir=out_dir,
            preprocessing_config=request.preprocessing_config,
            message=f"Validation passed. {len(pairs)} PSG/Hypnogram pairs matched.",
        )

    # 5. Output directory
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to make an output directory here {out_dir}. Error {e}",
        )

    # 6. Create job id
    job_id = str(uuid.uuid4())

    # 7. Add job to the dictionary of jobs
    JOBS[job_id] = {
        "status": JobStatus.queued,
        "created_at": datetime.now(),
        "progress": 0,
        "raw_dir": raw_dir,
        "out_dir": out_dir,
        "started_at": None,
        "finished_at": None,
        "message": None,
        "output_location": None,
        "error": None,
    }

    background_task.add_task(run_preprocess_job, job_id, raw_dir, out_dir, spec, combine)

    response = JobCreatedResponse(
        job_id=job_id, status=JobStatus.queued, status_url=f"/preprocess/{job_id}"
    )

    return response


@app.get("/preprocess/{job_id}")
def get_preprocessing_status(job_id: str):

    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="job_id not found")

    job = JOBS[job_id]
    response = PreprocessStatusResponse(
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
    return response

