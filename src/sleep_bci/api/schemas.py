"""
API Schemas for Sleep BCI Service

This file defines all request and response models used by the API.
These schemas serve as the API contract and are versioned implicitly
via route versioning (e.g., /v1/preprocess).
"""

from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
import os


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"

class PreprocessingConfig(BaseModel):
    channel: str = Field(
        default="EEG Fpz-Cz",
        description="EEG channel to extract edf file"
    )
    epochs: int = Field(
        default = 30,
        ge=1,
        description="splitting the edf data into epochs>=1"
    )
    bandpass: List[float] = Field(
        default=[0.3, 30.0],
        description="Bandpass filter range [low, high] in Hz."
    )

    notch: Optional[float] = Field(
        default=None,
        description="Optional notch filter frequency in Hz. Omit if bandpass already excludes it."
    )

    @field_validator("notch", mode="before")
    @classmethod
    def _notch_must_be_positive_or_none(cls, v):
        if v is not None and v <= 0:
            raise ValueError("notch must be a positive frequency (e.g. 50 or 60) or null/omitted")
        return v

class DatasetDescription(BaseModel):
    type: str = Field(
        default="local_edf",
        description="The dataset type rn using local_edf"
    )
    raw_dir: str = Field(
        ...,
        description="the path to the raw data"
    )
    
class OutputDir(BaseModel):
    out_dir: Optional[str] = Field(
        default=None,
        examples=[None],
        description="Output directory for processed files. Auto-generated if omitted."
    )
    combine: bool = Field(
        default=True,
        description="Whether to generate combined dataset artifact."
    )

    @field_validator("out_dir", mode="before")
    @classmethod
    def _out_dir_must_be_absolute_or_none(cls, v):
        if v is not None and not os.path.isabs(v):
            raise ValueError(
                f"out_dir must be an absolute path (e.g. /tmp/output) or null/omitted, got '{v}'"
            )
        return v

class PreprocessRequest(BaseModel):
    dataset: DatasetDescription
    output: OutputDir
    preprocessing_config: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig
    )

    dry_run: bool = Field(
        default=False,
        description="If true, validate inputs without writing output."
    )

    api_version: str = Field(
        default="v1",
        description="API contract version."
    )

class DryRunFileInfo(BaseModel):
    psg_file: str
    hypnogram_file: str


class DryRunResponse(BaseModel):
    valid: bool
    matched_pairs: int
    files: List[DryRunFileInfo]
    raw_dir: str
    out_dir: str
    preprocessing_config: PreprocessingConfig
    message: str


class JobCreatedResponse(BaseModel):
    job_id: str
    status: JobStatus
    status_url: str

class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None

class UploadResponse(BaseModel):
    session_id: str
    raw_dir: str
    matched_pairs: int
    files: List[DryRunFileInfo]
    message: str


class PreprocessStatusResponse(BaseModel):
    job_id: str
    status: JobStatus

    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    progress: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Progress percentage (0-100)."
    )

    message: Optional[str] = None

    output_location: Optional[str] = None
    error: Optional[ErrorDetail] = None



class TrainConfigRequest(BaseModel):
    npz_dir: str = Field(
        ...,
        description="The directory where your pre preprocessed .npz files live"
    )
    model_out: Optional[str] = Field(
        default=None,
        examples=[None],
        description="Where the training output will leave"
    )
    fs: float = Field(
        default=100.0,
        gt=0,
        description="Sampling frequency in Hz (must match preprocessing)"
    )
    n_splits: int = Field(
        default=5,
        ge=2,
        description="the number of cross validation folds"
    )

    @field_validator("model_out", mode="before")
    @classmethod
    def _model_out_must_be_absolute_or_none(cls, v):
        if v is not None and not os.path.isabs(v):
            raise ValueError(
                f"model_out must be an absolute path (e.g. /tmp/model.joblib) or null/omitted, got '{v}'"
            )
        return v

class TrainingJobCreated(BaseModel):
    training_id: str
    status: JobStatus
    status_url: str

class TrainingStatusResponse(BaseModel):
    npz_dir: str
    training_id: str
    status: JobStatus
   
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None

    progress: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="The progress of training the model"
    )
    message: Optional[str] = None
    output_location : Optional[str] = None

    error: Optional[ErrorDetail] = None




