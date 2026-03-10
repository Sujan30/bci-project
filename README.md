# Sleep-BCI: EEG Sleep Stage Classifier

[![CI](https://github.com/Sujan30/bci-project/actions/workflows/ci.yml/badge.svg)](https://github.com/Sujan30/bci-project/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Research-grade pipeline for EEG sleep stage classification: from raw polysomnography (PSG) data to a REST + WebSocket inference API.

---

## Limitations

> **Read this before evaluating the project:**

- **Single EEG channel**: uses only "EEG Fpz-Cz"; multi-channel fusion is not implemented.
- **Small dataset**: 3 nights / 8,281 epochs from Sleep-EDF Expanded (3 subjects).
- **Baseline accuracy**: 53.6% balanced accuracy (2-fold GroupKFold CV) — not clinical-grade; N1 recall is near zero due to class imbalance (~2.7% of epochs).
- **LDA baseline**: Linear Discriminant Analysis is a proof-of-concept; production systems use CNNs or attention-based RNNs.
- **No online learning**: model does not update incrementally.
- **Jobs run in-process**: when Redis is unavailable, jobs use FastAPI BackgroundTasks (not fault-tolerant).

---

## What This Project Does

```
EDF Files → Preprocessing (MNE) → Bandpower Features → LDA / RandomForest → REST API + WebSocket
```

**Pipeline stages**:
1. **Preprocessing**: Load EDF → bandpass filter (0.3–30 Hz) → 30-second epochs → extract channel
2. **Feature extraction**: Bandpower (δ, θ, α, β, γ) + ratio features → 7D feature vector per epoch
3. **Training**: LDA or RandomForest classifier with GroupKFold CV (subject-independent splits)
4. **Inference**: FastAPI REST + chunk-based WebSocket streaming (2.5s chunks → 30s epoch → prediction)

**Supported sleep stages**: W (Wake), N1, N2, N3, REM

---

## Results (3-Night Proof-of-Concept)

**Trained on 3 nights from Sleep-EDF Expanded** (SC4001, SC4002, SC4011):

| Metric | Score |
|--------|-------|
| Balanced Accuracy | **53.6% ± 12.9%** |
| Macro F1 | **53.3% ± 11.6%** |
| Total Epochs | 8,281 |
| Cross-Validation | 2-fold GroupKFold |

N1 class is severely underrepresented (58–109 epochs per night, ~2.7% of data) and is misclassified in nearly all folds.

---

## 60-Second Quickstart (Docker)

```bash
# 1. Clone
git clone https://github.com/Sujan30/bci-project && cd bci-project

# 2. Download sample data (3 nights from Sleep-EDF via PhysioNet)
bash scripts/download_sample_data.sh 3

# 3. Start API server + Redis
docker-compose up api

# 4. Open interactive docs
# http://localhost:8000/docs
```

**Verify the API is running**:
```bash
curl http://localhost:8000/v1/health
# {"status":"ok","uptime_s":4.2,"redis_connected":true,"model_loaded":false,"job_store_backend":"redis"}
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Client Layer                              │
│   (curl, Python SDK, WebSocket client, Lab Streaming Layer)      │
└─────────────────────────┬────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│                    FastAPI Server                                │
│  POST /v1/preprocess  POST /v1/train  WS /v1/stream             │
│  GET  /v1/preprocess/{id}  GET /v1/train/{id}                   │
│  GET  /v1/models  GET /v1/models/{id}/metrics  GET /v1/health   │
└──────────┬─────────────────────────────────────────┬────────────┘
           │ enqueue_job (ARQ)                        │ BackgroundTasks
           ▼                                          │ (fallback)
┌──────────────────────┐                             │
│  ARQ Worker Process  │◄────────────────────────────┘
│  (preprocessing,     │
│   training)          │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Storage Layer                               │
│  Redis (job queue + status) │ File system (models, NPZ, results) │
└──────────────────────────────────────────────────────────────────┘
```

**Key design decisions**:
- **GroupKFold CV**: splits on subject night, not epochs — prevents data leakage
- **ARQ worker queue**: durable job execution via Redis (falls back to BackgroundTasks)
- **Chunk-based WebSocket**: accepts 250-sample chunks; buffers to 3000 samples before inference
- **MNE**: industry-standard EEG library for loading and filtering EDF files

---

## Installation & Usage

### Option 1: Docker (Recommended)

```bash
# Start API + Redis
docker-compose up api

# Start worker (for durable job processing)
docker-compose up worker

# Development mode with hot reload
docker-compose --profile dev up dev
```

### Option 2: Local Python

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# Optional: start Redis (for persistent job store)
redis-server --daemonize yes

# Start API
./start.sh
```

---

## Getting the Data

The `data/raw/` directory is excluded from git. Download from PhysioNet:

```bash
bash scripts/download_sample_data.sh 3   # 3 nights (~300 MB)
bash scripts/download_sample_data.sh 5   # 5 nights (~500 MB)
```

> PhysioNet requires a free account. Register at: https://physionet.org/register/

**Dataset**: Sleep-EDF Expanded (Cassette subset) — Kemp et al., PhysioNet
**DOI**: [10.13026/C2SC7Q](https://doi.org/10.13026/C2SC7Q)

---

## API Usage Examples

### 1. Preprocess EDF Files

```bash
curl -X POST http://localhost:8000/v1/preprocess \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": {"raw_dir": "/app/data/raw"},
    "preprocessing_config": {
      "channel": "EEG Fpz-Cz",
      "epochs": 30,
      "bandpass": [0.3, 30.0]
    },
    "output": {"out_dir": null, "combine": true},
    "dry_run": false
  }'

# Response
{"job_id": "abc-123", "status": "queued", "status_url": "/v1/preprocess/abc-123"}

# Poll status
curl http://localhost:8000/v1/preprocess/abc-123
```

### 2. Train Classifier

```bash
curl -X POST http://localhost:8000/v1/train \
  -H "Content-Type: application/json" \
  -d '{
    "npz_dir": "/app/data/processed",
    "model_out": null,
    "fs": 100.0,
    "n_splits": 2,
    "model_type": "lda"
  }'

# model_type: "lda" or "random_forest"
# Response
{"job_id": "train-789", "status": "queued", "status_url": "/v1/train/train-789"}
```

### 3. Get Training Metrics

```bash
curl http://localhost:8000/v1/models/lda_pipeline/metrics
# Returns per-class F1, balanced accuracy, confusion matrix path, training config
```

### 4. Real-Time Streaming Inference (WebSocket — chunk-based)

```python
import asyncio, json, numpy as np, websockets

async def stream():
    uri = "ws://localhost:8000/v1/stream?model_id=lda_pipeline"
    async with websockets.connect(uri) as ws:
        # Send 12 chunks of 250 samples (= one 30s epoch at 100 Hz)
        chunk = np.random.randn(250).tolist()
        for i in range(12):
            await ws.send(json.dumps({
                "chunk": chunk,
                "fs": 100.0,
                "session_id": "session-001",
            }))
            resp = json.loads(await ws.recv())
            if "stage" in resp:
                print(resp)  # {"stage": "N2", "confidence": 0.87, "latency_ms": 4.2, "epoch_idx": 0}

asyncio.run(stream())
```

Or use the test script:
```bash
python scripts/ws_stream_test.py           # chunk-based (default)
python scripts/ws_stream_test.py --legacy  # full-epoch format
```

---

## Testing

```bash
# Unit + API tests (no Redis required)
pytest -m "not integration" --cov=sleep_bci

# End-to-end integration test (Redis must be running)
pytest tests/test_integration.py -v -m integration
```

**Current**: 53 passing unit tests, 6 skipped. Integration test covers the full pipeline.

---

## Project Structure

```
bci-project/
├── src/sleep_bci/
│   ├── api/          # FastAPI app, schemas, job store
│   ├── features/     # Bandpower feature extraction
│   ├── model/        # Training (LDA + RandomForest), artifacts
│   ├── preprocessing/# EDF loading, filtering, epoching
│   ├── stream/       # EpochBuffer for chunk-based streaming
│   └── workers/      # ARQ task functions and worker settings
├── tests/            # Unit + integration tests
├── scripts/
│   ├── download_sample_data.sh  # PhysioNet data downloader
│   └── ws_stream_test.py        # WebSocket streaming client
├── data/
│   ├── raw/          # EDF input files (not in git — use download script)
│   ├── processed/    # NPZ feature artifacts (not in git)
│   └── README.md     # Dataset manifest + checksums
├── models/           # Trained classifiers (.joblib)
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Roadmap

### Completed
- [x] End-to-end EDF → model → API pipeline
- [x] Docker + Redis support
- [x] Chunk-based WebSocket streaming with EpochBuffer
- [x] ARQ durable worker queue
- [x] RandomForest baseline (class-balanced)
- [x] Confusion matrix + per-class F1 artifacts
- [x] End-to-end integration test
- [x] Data manifest + checksums

### Planned
- [ ] Multi-channel feature fusion (improve N1 recall)
- [ ] CNN / attention-based model baseline
- [ ] Prometheus metrics + Grafana dashboard
- [ ] Online learning (incremental model updates)
- [ ] LSL (Lab Streaming Layer) live data integration

---

## References

- Kemp et al. (2000). *Sleep-EDF Database*. PhysioNet. DOI: 10.13026/C2SC7Q
- Rechtschaffen & Kales (1968). Manual sleep staging rules.
- [YASA](https://github.com/raphaelvallat/yasa) — advanced sleep analysis toolkit
- [MNE-Python](https://mne.tools/) — core EEG signal processing library

---

## Contact

**Author**: Sujan Nandikol Sunilkumar
**Email**: nandikolsujan@gmail.com
**LinkedIn**: [linkedin.com/in/suqjan](https://linkedin.com/in/suqjan)
**GitHub**: [@sujan30](https://github.com/sujan30)
