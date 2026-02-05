"""
API Schemas for Sleep BCI Service

This file defines all request and response models used by the API.
These schemas serve as the API contract and are versioned implicitly
via route versioning (e.g., /v1/preprocess).
"""

from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime


class JobStatus(str, Enum):
    queued = "queued",
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
        default=60.0,
        description="Optional notch filter frequency in Hz."
    )

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
    out_dir: str = Field(
        ...,
        description="the directory where the processed files will win"
    )
    combine: bool = Field(
        default=True,
        description="Whether to generate combined dataset artifact."
    )

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

class JobCreatedResponse(BaseModel):
    job_id: str
    status: JobStatus
    status_url: str

class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None

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



