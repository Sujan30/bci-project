# Sleep-BCI API Specification v1

**Version**: 1.0.0
**Base URL**: `http://localhost:8000`
**Protocol**: HTTP/1.1, WebSocket
**Authentication**: None (v1), API Key planned for v2

---

## Table of Contents

1. [Overview](#overview)
2. [Endpoints](#endpoints)
3. [WebSocket Streaming](#websocket-streaming)
4. [Request/Response Schemas](#requestresponse-schemas)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)
7. [Async Job Model](#async-job-model)

---

## Overview

### Design Principles

1. **Async-First**: Long-running operations (preprocess, train) return immediately with job ID
2. **Polling Pattern**: Clients poll status endpoints to check job progress
3. **Versioned Routes**: All endpoints prefixed with `/v1/` for future compatibility
4. **Idempotent**: Repeated requests with same parameters return same job ID (planned)

### API Flow

```
┌──────────┐
│  Client  │
└────┬─────┘
     │ POST /v1/preprocess
     ├─────────────────────────────────────────┐
     │                                         │
     ▼                                         ▼
┌─────────────┐    ┌───────────────────────────────┐
│  Job Queue  │───▶│  Background Worker (FastAPI)  │
└─────────────┘    └───────────────────────────────┘
     │                         │
     │ GET /v1/preprocess/{id} │ (writes to disk)
     ▼                         ▼
┌─────────────────────────────────────────┐
│  Job Status (Redis/in-memory)           │
│  - queued → running → succeeded/failed  │
└─────────────────────────────────────────┘
```

---

## Endpoints

### 1. Health Check

**GET** `/`

**Description**: Verify server is running

**Response**:
```json
{
  "message": "Hello, World!",
  "version": "1.0.0", 
  "status": "healthy"
}
```

**Status Codes**:
- `200 OK`: Server is healthy

---

### 2. Upload EDF Files

**POST** `/upload`

**Description**: Upload raw EDF files for validation (does not process, just validates pairs)

**Request**:
- Content-Type: `multipart/form-data`
- Body: Multiple files with `.edf` extension

**Example**:
```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@SC4001E0-PSG.edf" \
  -F "files=@SC4001EC-Hypnogram.edf"
```

**Response** (200 OK):
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "raw_dir": "/tmp/sleep-bci-upload-abc123",
  "matched_pairs": 1,
  "files": [
    {
      "psg_file": "SC4001E0-PSG.edf",
      "hypnogram_file": "SC4001EC-Hypnogram.edf"
    }
  ],
  "message": "Uploaded 2 files. 1 PSG/Hypnogram pairs matched."
}
```

**Error Responses**:
- `400 Bad Request`: Invalid file format or missing required files
  ```json
  {
    "detail": "No PSG file found. At least one file matching '*PSG.edf' is required."
  }
  ```

**Status Codes**:
- `200 OK`: Files uploaded and validated
- `400 Bad Request`: Validation failed
- `413 Payload Too Large`: File size exceeds limit (100MB per file)

---

### 3. Preprocess Data (Async Job)

**POST** `/v1/preprocess`

**Description**: Start preprocessing job to convert raw EDF files into feature-ready NPZ format

**Request Body**:
```json
{
  "dataset": {
    "type": "local_edf",
    "raw_dir": "/app/data/raw/sleep-cassette"
  },
  "preprocessing_config": {
    "channel": "EEG Fpz-Cz",
    "epochs": 30,
    "bandpass": [0.5, 30.0],
    "notch": null
  },
  "output": {
    "out_dir": null,
    "combine": true
  },
  "dry_run": false,
  "api_version": "v1"
}
```

**Response** (200 OK):
```json
{
  "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "queued",
  "status_url": "/v1/preprocess/3fa85f64-5717-4562-b3fc-2c963f66afa6"
}
```

**Dry Run Response** (if `dry_run: true`):
```json
{
  "valid": true,
  "matched_pairs": 3,
  "files": [
    {"psg_file": "SC4001E0-PSG.edf", "hypnogram_file": "SC4001EC-Hypnogram.edf"},
    {"psg_file": "SC4002E0-PSG.edf", "hypnogram_file": "SC4002EC-Hypnogram.edf"},
    {"psg_file": "SC4003E0-PSG.edf", "hypnogram_file": "SC4003EC-Hypnogram.edf"}
  ],
  "raw_dir": "/app/data/raw/sleep-cassette",
  "out_dir": "/tmp/sleep-bci-output-xyz",
  "preprocessing_config": { ... },
  "message": "Validation passed. 3 PSG/Hypnogram pairs matched."
}
```

**Status Codes**:
- `200 OK`: Job created or dry-run succeeded
- `400 Bad Request`: Invalid parameters or missing data

---

### 4. Get Preprocessing Status

**GET** `/v1/preprocess/{job_id}`

**Description**: Poll preprocessing job status

**Path Parameters**:
- `job_id` (string, required): Job identifier from POST response

**Response** (200 OK - Running):
```json
{
  "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "running",
  "created_at": "2026-02-15T12:00:00Z",
  "started_at": "2026-02-15T12:00:05Z",
  "finished_at": null,
  "progress": 66.7,
  "message": "Processing night 2/3 (SC4002E0)",
  "output_location": "/app/data/processed",
  "error": null
}
```

**Response** (200 OK - Succeeded):
```json
{
  "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "succeeded",
  "created_at": "2026-02-15T12:00:00Z",
  "started_at": "2026-02-15T12:00:05Z",
  "finished_at": "2026-02-15T12:02:30Z",
  "progress": 100.0,
  "message": "Done. Kept 3, skipped 0.",
  "output_location": "/app/data/processed",
  "error": null
}
```

**Response** (200 OK - Failed):
```json
{
  "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "status": "failed",
  "created_at": "2026-02-15T12:00:00Z",
  "started_at": "2026-02-15T12:00:05Z",
  "finished_at": "2026-02-15T12:00:45Z",
  "progress": 33.3,
  "message": "Failed: Channel 'EEG Fpz-Cz' not found in SC4002E0-PSG.edf",
  "output_location": null,
  "error": {
    "code": "PREPROCESSING_ERROR",
    "message": "Channel 'EEG Fpz-Cz' not found in SC4002E0-PSG.edf",
    "details": {
      "traceback": "Traceback (most recent call last):\n..."
    }
  }
}
```

**Status Codes**:
- `200 OK`: Job found (check `status` field)
- `404 Not Found`: Job ID does not exist

---

### 5. Train Classifier (Async Job)

**POST** `/train`

**Description**: Start training job to build LDA classifier from preprocessed NPZ files

**Request Body**:
```json
{
  "npz_dir": "/app/data/processed",
  "model_out": null,
  "fs": 100.0,
  "n_splits": 5
}
```

**Field Descriptions**:
- `npz_dir`: Directory containing `*.npz` files (output from preprocessing)
- `model_out`: Path to save trained model (auto-generated if null)
- `fs`: Sampling frequency in Hz (must match preprocessing)
- `n_splits`: Number of cross-validation folds (must be ≤ number of nights)

**Response** (200 OK):
```json
{
  "training_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "status": "queued",
  "status_url": "/training/7c9e6679-7425-40de-944b-e07fc1f90ae7"
}
```

**Status Codes**:
- `200 OK`: Training job created
- `400 Bad Request`: Invalid parameters (e.g., `n_splits > num_nights`)

---

### 6. Get Training Status

**GET** `/training/{training_id}`

**Description**: Poll training job status

**Response** (200 OK - Succeeded):
```json
{
  "npz_dir": "/app/data/processed",
  "training_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "status": "succeeded",
  "created_at": "2026-02-15T12:05:00Z",
  "started_at": "2026-02-15T12:05:02Z",
  "finished_at": "2026-02-15T12:06:15Z",
  "progress": 100.0,
  "message": "Training complete",
  "output_location": "/app/models",
  "error": null,
  "results": {
    "n_splits": 5,
    "fs": 100.0,
    "total_epochs": 8281,
    "total_nights": 3,
    "folds": [
      {
        "fold": 1,
        "balanced_accuracy": 0.6647,
        "macro_f1": 0.6495,
        "classification_report": { ... }
      },
      ...
    ],
    "overall": {
      "balanced_accuracy_mean": 0.536,
      "balanced_accuracy_std": 0.1287,
      "macro_f1_mean": 0.5333,
      "macro_f1_std": 0.1162
    }
  }
}
```

**Status Codes**:
- `200 OK`: Training job found
- `404 Not Found`: Training ID does not exist

---

### 7. Batch Prediction (Planned v1.1)

**POST** `/v1/predict`

**Description**: Synchronous batch inference (single or multiple epochs)

**Request Body**:
```json
{
  "model_id": "lda_pipeline",
  "epochs": [
    [0.1, -0.2, 0.3, ...],  // 3000 samples @ 100 Hz
    [0.2, 0.1, -0.1, ...]
  ]
}
```

**Response** (200 OK):
```json
{
  "predictions": [
    {"stage": "N2", "label": 2, "confidence": 0.87},
    {"stage": "W", "label": 0, "confidence": 0.92}
  ],
  "model_id": "lda_pipeline",
  "inference_time_ms": 12.5
}
```

**Status**: Not yet implemented (planned for v1.1)

---

## WebSocket Streaming

### Endpoint: `/v1/stream`

**Protocol**: WebSocket (ws:// or wss://)
**Description**: Real-time streaming inference for continuous EEG data

### Connection

```python
import websockets
import json

async with websockets.connect("ws://localhost:8000/v1/stream?model_id=lda_pipeline") as ws:
    # Connection established
```

**Query Parameters**:
- `model_id` (string, required): Model filename (without extension)

### Message Format

**Client → Server** (Epoch Data):
```json
{
  "epoch": [0.1, -0.2, 0.3, ...],  // Array of 3000 floats (30s @ 100 Hz)
  "timestamp": 1234567890.123       // Optional: Unix timestamp
}
```

**Server → Client** (Prediction):
```json
{
  "stage": "N2",
  "label": 2,
  "confidence": 0.87,
  "latency_ms": 45.2,
  "timestamp": 1234567890.123
}
```

**Server → Client** (Error):
```json
{
  "error": "INVALID_EPOCH_LENGTH",
  "message": "Expected 3000 samples, got 2500"
}
```

### Error Codes

- `INVALID_EPOCH_LENGTH`: Wrong number of samples
- `MODEL_NOT_FOUND`: Model ID does not exist
- `PREDICTION_FAILED`: Internal error during inference

### Example Client

```python
import asyncio
import websockets
import json
import numpy as np

async def stream_predictions():
    uri = "ws://localhost:8000/v1/stream?model_id=lda_pipeline"

    async with websockets.connect(uri) as ws:
        # Simulate 10 epochs
        for i in range(10):
            epoch = np.random.randn(3000).tolist()
            await ws.send(json.dumps({
                "epoch": epoch,
                "timestamp": time.time()
            }))

            response = json.loads(await ws.recv())
            print(f"Epoch {i}: {response['stage']} (confidence: {response['confidence']:.2f})")

asyncio.run(stream_predictions())
```

---

## Request/Response Schemas

### PreprocessingConfig

```json
{
  "channel": "EEG Fpz-Cz",
  "epochs": 30,
  "bandpass": [0.5, 30.0],
  "notch": null
}
```

**Fields**:
- `channel` (string): EEG channel name (must exist in EDF)
- `epochs` (integer, ≥1): Epoch duration in seconds
- `bandpass` (array[float, float]): Bandpass filter [low, high] in Hz
- `notch` (float | null): Notch filter frequency (50 or 60 Hz)

**Validation**:
- `epochs` must be ≥1
- `bandpass[0]` < `bandpass[1]`
- `notch` must be positive or null

---

### JobStatus Enum

```json
"queued" | "running" | "succeeded" | "failed"
```

**State Transitions**:
```
queued → running → succeeded
                ↘ failed
```

---

### ErrorDetail

```json
{
  "code": "PREPROCESSING_ERROR",
  "message": "Channel 'EEG Fpz-Cz' not found",
  "details": {
    "traceback": "...",
    "file": "SC4002E0-PSG.edf"
  }
}
```

**Common Error Codes**:
- `PREPROCESSING_ERROR`: Failure during EDF processing
- `TRAINING_ERROR`: Failure during model training
- `VALIDATION_ERROR`: Invalid input parameters
- `FILE_NOT_FOUND`: Missing data files
- `MODEL_NOT_FOUND`: Model file does not exist

---

## Error Handling

### HTTP Error Responses

All errors follow FastAPI's default format:

```json
{
  "detail": "Error message here"
}
```

**Common Status Codes**:
- `400 Bad Request`: Invalid input (validation failed)
- `404 Not Found`: Resource not found (job ID, model ID)
- `413 Payload Too Large`: File upload exceeds limit
- `422 Unprocessable Entity`: Pydantic validation error
- `500 Internal Server Error`: Unexpected server error

### Validation Errors (422)

```json
{
  "detail": [
    {
      "loc": ["body", "preprocessing_config", "epochs"],
      "msg": "ensure this value is greater than or equal to 1",
      "type": "value_error.number.not_ge"
    }
  ]
}
```

---

## Rate Limiting

**Status**: Not implemented in v1

**Planned for v2**:
- 100 requests/minute per IP
- 10 concurrent jobs per user
- Response headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`

---

## Async Job Model

### Design Pattern

Sleep-BCI uses the **Async Request-Reply** pattern for long-running operations:

1. **Client** submits job → receives `job_id` immediately
2. **Server** processes job in background (FastAPI `BackgroundTasks`)
3. **Client** polls status endpoint until `status ∈ {succeeded, failed}`
4. **Server** stores job state in Redis (planned) or in-memory dict (current)

### Benefits

- Non-blocking API (server doesn't time out)
- Client can disconnect and reconnect
- Multiple clients can monitor same job
- Horizontal scaling (with Redis backend)

### Polling Recommendations

**Exponential Backoff**:
```python
import time

job_id = create_job()
delay = 1  # Start with 1 second

while True:
    status = get_status(job_id)
    if status["status"] in ["succeeded", "failed"]:
        break

    time.sleep(delay)
    delay = min(delay * 1.5, 30)  # Cap at 30 seconds
```

**Estimated Durations**:
- Preprocessing: 15-30s per night (3 nights → ~1 min)
- Training: 30-60s for 3 nights with 2-fold CV

---

## Future Endpoints (v1.1+)

| Endpoint | Method | Description | Status |
|----------|--------|-------------|--------|
| `/v1/models` | GET | List available models | Planned |
| `/v1/models/{id}` | GET | Get model metadata | Planned |
| `/v1/models/{id}` | DELETE | Delete model | Planned |
| `/v1/predict` | POST | Batch synchronous inference | Planned |
| `/v1/health` | GET | Detailed health check | Planned |
| `/metrics` | GET | Prometheus metrics | Planned |
| `/v1/calibrate` | POST | User-specific calibration | Planned |

---

## Versioning Strategy

- **Current**: v1 (routes prefixed with `/v1/`)
- **Breaking changes**: Increment version (v2, v3, ...)
- **Non-breaking changes**: Update within same version
- **Deprecation**: 6-month notice before removing old version

---

## OpenAPI/Swagger Documentation

Interactive API documentation available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

---

## Support

**Issues**: https://github.com/yourusername/sleep-bci/issues
**Discussions**: https://github.com/yourusername/sleep-bci/discussions

---

**Last Updated**: 2026-02-15
**API Version**: 1.0.0
