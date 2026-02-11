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
    TrainConfigRequest, 
    TrainingJobCreated,
    TrainingStatusResponse
)
from sleep_bci.preprocessing.core import (
    PreprocessSpec,
    preprocess_sleep_edf,
    discover_and_validate,
)
from sleep_bci.preprocessing.combine import combine_nights
from sleep_bci.model.train import load_nightly_npz, train_lda
from sleep_bci.model.artifacts import save_bundle

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
        "output_location": out_dir,
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

training_data = {}


#helper function for background task for training

def update_training(
        train_id: str,
        npz_directory: str,
        model_out: str,
        fs: float, 
        n_splits: int

):
    job = training_data[train_id]
    job["status"] = JobStatus.running
    job["started_at"] = datetime.now()
    job["message"] = f"now working on training {train_id}"
    logger.info("Training %s started — npz_dir=%s, model_out=%s", train_id, npz_directory, model_out)

    try:
        logger.info("Training %s: loading nightly NPZ files...", train_id)
        X, Y, night_ids = load_nightly_npz(npz_directory)
        logger.info("Training %s: loaded %d epochs from %d nights", train_id, X.shape[0], len(set(night_ids)))
        bundle = train_lda(X, Y, night_ids, fs=fs, n_splits=n_splits)
        logger.info("Training %s: saving model to %s", train_id, model_out)
        save_bundle(model_out, bundle)
        job["status"] = JobStatus.succeeded
        job['progress'] = 100
        job['finished_at'] = datetime.now()
        job['output_location'] = model_out
        job['message'] = "Training complete"
        logger.info("Training %s succeeded", train_id)
    except Exception as e:
        job["status"] = JobStatus.failed
        job["finished_at"] = datetime.now()
        job["error"] = ErrorDetail(
            code="TRAINING_ERROR",
            message=str(e),
            details={"traceback": traceback.format_exc()},
        )
        job["message"] = f"Failed: {e}"
        logger.error("Training %s failed: %s", train_id, e)
        

    

@app.post("/train")
def train_model(request: TrainConfigRequest, background_task: BackgroundTasks):
    
    # step 1. validate

    if not (os.path.exists(request.npz_dir)):
        raise HTTPException(status_code=400,detail= "NPZ directory not found")

    import glob as _glob
    night_files = [
        f for f in _glob.glob(os.path.join(request.npz_dir, "*.npz"))
        if "sleep_edf_all" not in os.path.basename(f)
    ]
    if len(night_files) == 0:
        raise HTTPException(status_code=400, detail="No .npz night files found in npz_dir")
    if len(night_files) < request.n_splits:
        raise HTTPException(
            status_code=400,
            detail=f"n_splits={request.n_splits} but only {len(night_files)} night(s) found. "
                   f"n_splits must be <= number of nights.",
        )

    if request.model_out is None:
        model_out = os.path.join(tempfile.mkdtemp(prefix=OUTPUT_PREFIX), "model.joblib")
    else:
        model_out = request.model_out
    
    train_id = str(uuid.uuid4())

    training_data[train_id]= {
        "npz_dir": request.npz_dir,
        "status": JobStatus.queued,
        "model_out": model_out,
        "fs": request.fs,
        "n_splits": request.n_splits,
        "created_at": datetime.now(),
        "progress": 0,
        "started_at": None,
        "finished_at": None,
        "message": None,
        "output_location": model_out,
        "error": None,
    }
    
    background_task.add_task(update_training, train_id, request.npz_dir, model_out, request.fs, request.n_splits)

    return TrainingJobCreated(
        training_id=train_id, 
        status = JobStatus.queued,
        status_url=f"/train/{train_id}"
    )
    


@app.get("/training/{train_id}")
def get_training_status(train_id: str):
    if train_id not in training_data:
        raise HTTPException(status_code=404, detail="train_id not found")
    job = training_data[train_id]
    return TrainingStatusResponse(
        npz_dir=job["npz_dir"],
        training_id=train_id,
        status=job['status'],
        created_at=job['created_at'],
        started_at=job['started_at'],
        finished_at=job['finished_at'],
        progress=job['progress'],
        message=job['message'],
        output_location=job['output_location']
    )
    
    





    

    






