# Sleep-BCI: EEG Sleep Stage Classifier

[![CI](https://github.com/Sujan30/bci-project/actions/workflows/ci.yml/badge.svg)](https://github.com/Sujan30/bci-project/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Research-grade pipeline for EEG sleep stage classification: raw polysomnography (PSG) data → feature extraction → REST + WebSocket inference API.

---

## What's New (March 2026)

| Area | Change |
|------|--------|
| **Streaming** | WebSocket now accepts 250-sample chunks instead of full 3000-sample epochs — realistic device behavior |
| **Workers** | ARQ worker queue (via Redis) replaces in-process BackgroundTasks for fault-tolerant job execution |
| **Models** | RandomForest baseline added alongside LDA; both use `class_weight='balanced'` for N1 imbalance |
| **Artifacts** | Training now saves `confusion_matrix.png` + `confusion_matrix.json` + per-class F1 scores |
| **Health** | `GET /v1/health` returns `uptime_s`, `redis_connected`, `model_loaded` — no more "we are flowing!" |
| **Observability** | Structured JSON logs, per-request latency (`duration_ms`), `X-Request-ID` header on every response |
| **Data hygiene** | Raw EDF files excluded from git; `data/README.md` with checksums and class-distribution manifest |
| **Integration test** | `pytest tests/test_integration.py -m integration` — full pipeline without mocks |

---

## Limitations

- **Single EEG channel**: uses only "EEG Fpz-Cz"; multi-channel fusion not implemented
- **Small dataset**: 3 nights / 8,281 epochs — not clinical-grade
- **Baseline accuracy**: 53.6% balanced accuracy (2-fold CV); N1 recall near zero (~2.7% of epochs)
- **LDA is a baseline**: production systems use CNNs/RNNs
- **No online learning**: model does not update incrementally

---

## Quick Start (Docker)

```bash
# 1. Clone
git clone https://github.com/Sujan30/bci-project && cd bci-project

# 2. Download 3 nights of Sleep-EDF data from PhysioNet
bash scripts/download_sample_data.sh 3

# 3. Start API + Redis
docker-compose up api

# 4. (Optional) Start ARQ worker for durable job processing
docker-compose up worker
```

Verify it's running:
```bash
curl http://localhost:8000/v1/health
# {"status":"ok","uptime_s":4.1,"redis_connected":true,"model_loaded":false,"job_store_backend":"redis"}
```

---

## How to Test the New Features

### 1. Chunk-Based WebSocket Streaming

The `/v1/stream` endpoint now accepts 250-sample chunks instead of full epochs.
A model must be trained first (see step below), then:

```bash
# Fast test: sends 250-sample chunks, auto-assembles 30s epochs
python scripts/ws_stream_test.py

# With real 30s delays between epochs (true real-time simulation)
python scripts/ws_stream_test.py --realtime

# Legacy full-epoch mode (still works for backward compatibility)
python scripts/ws_stream_test.py --legacy

# Custom chunk size
python scripts/ws_stream_test.py --chunk-size 500
```

Python client example:
```python
import asyncio, json, numpy as np, websockets

async def stream():
    uri = "ws://localhost:8000/v1/stream?model_id=lda_pipeline"
    chunk = np.random.randn(250).tolist()  # 2.5s of EEG at 100 Hz

    async with websockets.connect(uri) as ws:
        for i in range(12):  # 12 × 250 = 3000 samples = one 30s epoch
            await ws.send(json.dumps({"chunk": chunk, "fs": 100.0, "session_id": "demo"}))
            resp = json.loads(await ws.recv())
            if "stage" in resp:
                print(resp)  # {"stage": "N2", "confidence": 0.87, "latency_ms": 3.1, "epoch_idx": 0}
            else:
                print(f"Buffering: {resp['buffered']}/3000 samples")

asyncio.run(stream())
```

---

### 2. Train with RandomForest (class-balanced)

```bash
curl -X POST http://localhost:8000/v1/train \
  -H "Content-Type: application/json" \
  -d '{
    "npz_dir": "/app/data/processed",
    "model_out": null,
    "fs": 100.0,
    "n_splits": 2,
    "model_type": "random_forest"
  }'
# model_type: "lda" (default) or "random_forest"
```

After training completes, retrieve metrics:
```bash
curl http://localhost:8000/v1/models/lda_pipeline/metrics
# Returns per-class F1, balanced accuracy, training config, folds
```

Confusion matrix artifacts are saved to the same directory as the model:
- `confusion_matrix.png` — visual heatmap
- `confusion_matrix.json` — raw matrix + normalized values

---

### 3. Health Check with Uptime

```bash
curl http://localhost:8000/v1/health
# {
#   "status": "ok",
#   "uptime_s": 142.3,
#   "redis_connected": true,
#   "model_loaded": true,
#   "job_store_backend": "redis"
# }
```

Check `X-Request-ID` tracing header:
```bash
curl -v http://localhost:8000/v1/health 2>&1 | grep X-Request-ID
# < X-Request-ID: 4a7c1f83-9d2e-4b5a-...
```

---

### 4. ARQ Worker Queue

When Redis is running, jobs are dispatched to the ARQ worker instead of running in-process:

```bash
# Start the worker (separate terminal or docker-compose up worker)
arq sleep_bci.workers.settings.WorkerSettings

# Or via Docker
docker-compose up worker
```

Jobs automatically route to ARQ when Redis is available. When Redis is unavailable, they fall back to FastAPI `BackgroundTasks` — no code changes needed.

---

### 5. Run the Integration Test

```bash
# Requires no external services (uses in-memory job store)
pytest tests/test_integration.py -v -m integration
```

This test covers the full pipeline without mocks:
1. Generates synthetic NPZ data
2. Trains a model via `POST /v1/train`
3. Polls until `succeeded`
4. Verifies model appears in `GET /v1/models`
5. Streams 12 chunks via WebSocket → asserts a valid prediction

---

### 6. Data Integrity Check

```bash
# Verify processed NPZ files haven't changed
sha256sum -c data/processed/checksums.sha256
```

---

## Full Pipeline Walkthrough

```bash
# 1. Download data
bash scripts/download_sample_data.sh 3

# 2. Preprocess (via API)
curl -X POST http://localhost:8000/v1/preprocess \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": {"raw_dir": "/absolute/path/to/data/raw"},
    "preprocessing_config": {"channel": "EEG Fpz-Cz", "epochs": 30, "bandpass": [0.3, 30.0]},
    "output": {"combine": true},
    "dry_run": false
  }'
# → {"job_id": "abc-123", "status": "queued", "status_url": "/v1/preprocess/abc-123"}

# 3. Poll until done
curl http://localhost:8000/v1/preprocess/abc-123

# 4. Train
curl -X POST http://localhost:8000/v1/train \
  -H "Content-Type: application/json" \
  -d '{"npz_dir": "/path/to/data/processed", "n_splits": 2, "model_type": "lda"}'

# 5. Check training metrics
curl http://localhost:8000/v1/models/lda_pipeline/metrics

# 6. Stream real-time inference
python scripts/ws_stream_test.py
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Client Layer                              │
│    curl / Python SDK / WebSocket client                          │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                    FastAPI Server                                │
│  POST /v1/preprocess     POST /v1/train     WS /v1/stream        │
│  GET  /v1/preprocess/{id}  GET /v1/train/{id}                    │
│  GET  /v1/models  GET /v1/models/{id}/metrics  GET /v1/health    │
└──────────┬─────────────────────────────────────────┬────────────┘
           │ enqueue_job (ARQ / Redis)               │ BackgroundTasks
           ▼                                         │ (fallback, no Redis)
┌──────────────────────┐                            │
│  ARQ Worker Process  │◄───────────────────────────┘
│  preprocess_task     │
│  train_task          │
└──────────┬───────────┘
           ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Storage Layer                               │
│  Redis (job queue + status) │ File system (models, NPZ, results) │
└──────────────────────────────────────────────────────────────────┘
```

**WebSocket chunk buffering** (`EpochBuffer`):
```
Client sends:  [250 samples] → [250] → [250] → ... × 12
                                                      ↓ (3000 samples = 1 epoch)
                                               Feature extraction → LDA → {"stage": "N2"}
```

---

## Project Structure

```
bci-project/
├── src/sleep_bci/
│   ├── api/
│   │   ├── app.py          # FastAPI app (all endpoints)
│   │   ├── schemas.py      # Request/response models
│   │   └── job_store.py    # Redis-backed job state
│   ├── features/
│   │   └── bandpower.py    # 7D feature extraction (δ, θ, α, β, γ + ratios)
│   ├── model/
│   │   ├── train.py        # train_classifier() — LDA + RandomForest
│   │   └── artifacts.py    # ModelBundle dataclass + save/load
│   ├── preprocessing/
│   │   └── core.py         # EDF → filtered epochs → NPZ
│   ├── stream/
│   │   └── buffer.py       # EpochBuffer — chunk accumulation
│   └── workers/
│       ├── tasks.py        # ARQ async task functions
│       └── settings.py     # ARQ WorkerSettings
├── tests/
│   ├── test_api.py
│   ├── test_train.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_redis.py
│   └── test_integration.py  # End-to-end (no mocks)
├── scripts/
│   ├── download_sample_data.sh  # PhysioNet downloader
│   └── ws_stream_test.py        # WebSocket chunk streaming client
├── data/
│   ├── raw/              # EDF files (not in git — use download script)
│   ├── processed/        # NPZ artifacts + checksums.sha256 + manifest.json
│   └── README.md         # Dataset manifest
├── models/               # Trained .joblib files
├── docker-compose.yml    # api + worker + redis services
└── .github/workflows/
    └── ci.yml            # Unit tests + integration tests (with Redis service)
```

---

## Testing

```bash
# Unit tests (no external services needed)
pytest -m "not integration" --cov=sleep_bci -q

# Integration test (full pipeline, no mocks)
pytest tests/test_integration.py -v -m integration

# All tests
pytest -q
```

**Current**: 60 passing (59 unit + 1 integration).

---

## Results (3-Night Baseline)

| Metric | LDA | RandomForest |
|--------|-----|--------------|
| Balanced Accuracy | 53.6% ± 12.9% | TBD |
| Macro F1 | 53.3% ± 11.6% | TBD |
| N1 F1 | ~0% | Higher (class_weight='balanced') |
| Total Epochs | 8,281 | 8,281 |

---

## Dataset

**Source**: Sleep-EDF Expanded (Cassette subset) — Kemp et al., PhysioNet
**DOI**: [10.13026/C2SC7Q](https://doi.org/10.13026/C2SC7Q)

Download: `bash scripts/download_sample_data.sh 3`
See `data/README.md` for full manifest, class distributions, and checksums.

---

## Contact

**Author**: Sujan Nandikol Sunilkumar
**Email**: nandikolsujan@gmail.com
**LinkedIn**: [linkedin.com/in/suqjan](https://linkedin.com/in/suqjan)
**GitHub**: [@sujan30](https://github.com/sujan30)
