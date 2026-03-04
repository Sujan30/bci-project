# Sleep BCI API Implementation

## Summary

The Sleep BCI API is now fully functional and can be started with the `sleepbci-serve` command.

## What Was Fixed

1. **Created `/src/sleep_bci/api/main.py`** - Entry point module that `pyproject.toml` was expecting
2. **Fixed imports** - Updated `app.py` to use proper module paths (`sleep_bci.api.schemas`)
3. **Added v1 prefix** - Updated endpoints to use `/v1/preprocess` path as requested

## Implemented Endpoints

### 1. POST /v1/preprocess
Creates an async preprocessing job.

**Request:**
```json
{
  "dataset": {
    "raw_dir": "/path/to/edf/files"
  },
  "preprocessing_config": {
    "channel": "EEG Fpz-Cz",
    "epochs": 30,
    "bandpass": [0.5, 30],
    "notch": 50
  },
  "output": {
    "out_dir": "/path/to/output",
    "combine": false
  },
  "dry_run": false
}
```

**Response:**
```json
{
  "job_id": "612ba1dc-0f22-4e61-9986-7d0a9256dc0c",
  "status": "queued",
  "status_url": "/v1/preprocess/612ba1dc-0f22-4e61-9986-7d0a9256dc0c"
}
```

### 2. GET /v1/preprocess/{job_id}
Gets the status and progress of a preprocessing job.

**Response:**
```json
{
  "job_id": "612ba1dc-0f22-4e61-9986-7d0a9256dc0c",
  "status": "running",
  "created_at": "2026-02-14T13:00:11.702860",
  "started_at": "2026-02-14T13:00:11.703529",
  "finished_at": null,
  "progress": 66.67,
  "message": "Processing night 2/3 (SC4002E0)",
  "output_location": "/tmp/test-preprocess-output",
  "error": null
}
```

**Status values:**
- `queued` - Job is waiting to start
- `running` - Job is currently processing
- `succeeded` - Job completed successfully
- `failed` - Job encountered an error

## How to Run

### Start the server:
```bash
sleepbci-serve
```

The server will start on `http://0.0.0.0:8000`

### Test the API:
```bash
# Run the test script
./test_api.sh

# Or test individual endpoints:
curl -X POST http://localhost:8000/v1/preprocess \
  -H "Content-Type: application/json" \
  -d @request.json

curl http://localhost:8000/v1/preprocess/{job_id}
```

## Additional Endpoints

The API also includes:
- `POST /upload` - Upload EDF files
- `POST /train` - Train a model
- `GET /training/{train_id}` - Get training status

## Architecture

- **FastAPI** - Modern async web framework
- **Background Tasks** - Jobs run asynchronously without blocking
- **Progress Tracking** - Real-time progress updates via callbacks
- **Error Handling** - Detailed error messages with tracebacks
- **Temp Directory Management** - Automatic cleanup of temporary files

## Testing

A test script is provided at `test_api.sh` that demonstrates:
1. Server health check
2. Creating a preprocessing job
3. Polling job status
4. Viewing progress and results