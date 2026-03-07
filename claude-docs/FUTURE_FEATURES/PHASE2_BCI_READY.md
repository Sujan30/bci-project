# Phase 2: BCI-Ready (1-2 weeks)

**Objective**: Demonstrate real-time BCI understanding

### 2.1 Streaming Inference Endpoint ⏱️ 6h | Complexity: M

**Files to create**:
- `src/sleep_bci/api/streaming.py`
- `src/sleep_bci/stream/buffer.py`
- `tests/test_streaming.py`

**Acceptance Criteria**:
- [ ] WebSocket endpoint `/v1/stream`
- [ ] Client sends 30s EEG epochs → receives predictions
- [ ] Handles multiple concurrent streams
- [ ] Graceful error handling
- [ ] Latency <100ms (logged)

**Implementation**:
```python
# src/sleep_bci/api/streaming.py
from fastapi import WebSocket

@app.websocket("/v1/stream")
async def stream_inference(websocket: WebSocket, model_id: str):
    await websocket.accept()
    bundle = load_bundle(f"models/{model_id}.joblib")

    while True:
        # Receive epoch data (JSON or binary)
        data = await websocket.receive_json()
        epoch = np.array(data["epoch"])

        # Extract features + predict
        feats = extract_features_epoch(epoch, fs=bundle.fs)
        pred = bundle.pipeline.predict([feats])[0]

        # Send prediction
        await websocket.send_json({
            "stage": bundle.label_map[pred],
            "confidence": float(max(bundle.pipeline.predict_proba([feats])[0]))
        })
```

**Test Client**:
```python
# examples/stream_client.py
import asyncio, websockets, json

async def stream():
    uri = "ws://localhost:8000/v1/stream?model_id=lda_pipeline"
    async with websockets.connect(uri) as ws:
        for epoch in load_epochs():
            await ws.send(json.dumps({"epoch": epoch.tolist()}))
            response = await ws.recv()
            print(response)
```

---

### 2.2 Redis Job Storage ⏱️ 5h | Complexity: M

**Files to modify**:
- `src/sleep_bci/api/app.py` (replace JOBS dict)
- `src/sleep_bci/api/storage.py` (new)
- `docker-compose.yml` (add redis service)

**Acceptance Criteria**:
- [ ] Jobs persist across server restarts
- [ ] Redis backend with clean interface
- [ ] Fallback to in-memory if Redis unavailable
- [ ] TTL for completed jobs (24h)

**Implementation**:
```python
# src/sleep_bci/api/storage.py
import redis, json
from typing import Optional

class JobStorage:
    def __init__(self, redis_url: Optional[str] = None):
        self.redis = redis.from_url(redis_url) if redis_url else None
        self.mem = {}  # Fallback

    def set(self, job_id: str, data: dict, ttl: int = 86400):
        if self.redis:
            self.redis.setex(f"job:{job_id}", ttl, json.dumps(data))
        else:
            self.mem[job_id] = data

    def get(self, job_id: str) -> Optional[dict]:
        if self.redis:
            val = self.redis.get(f"job:{job_id}")
            return json.loads(val) if val else None
        return self.mem.get(job_id)
```

**docker-compose.yml**:
```yaml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  api:
    environment:
      - REDIS_URL=redis://redis:6379
```

---

### 2.3 Structured Logging ⏱️ 3h | Complexity: S

**Files to modify**:
- `src/sleep_bci/api/app.py`
- `src/sleep_bci/preprocessing/core.py`
- `src/sleep_bci/model/train.py`
- `src/sleep_bci/logging.py` (new)

**Acceptance Criteria**:
- [ ] All print() replaced with logger
- [ ] JSON-formatted logs (production)
- [ ] Human-readable logs (dev)
- [ ] Request ID tracking
- [ ] Configurable log level

**Implementation**:
```python
# src/sleep_bci/logging.py
import logging, sys

def setup_logging(level: str = "INFO", json_logs: bool = False):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    if json_logs:
        # Add structlog here
        pass

# src/sleep_bci/api/app.py
from sleep_bci.logging import setup_logging

setup_logging(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Replace all print() with logger.info/debug/error
```

---

### 2.4 API Tests (Full Coverage) ⏱️ 4h | Complexity: M

**Files to create**:
- `tests/test_api_upload.py`
- `tests/test_api_streaming.py`
- `tests/test_api_errors.py`

**Acceptance Criteria**:
- [ ] All endpoints tested
- [ ] Error cases covered
- [ ] File upload tested
- [ ] WebSocket tested
- [ ] Job polling tested

---

### 2.5 Configuration Management ⏱️ 2h | Complexity: S

**Files to create**:
- `src/sleep_bci/config.py`

**Acceptance Criteria**:
- [ ] Pydantic BaseSettings
- [ ] Env vars with defaults
- [ ] Validated at startup
- [ ] Single source of truth

**Implementation**:
```python
# src/sleep_bci/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    port: int = 8000
    host: str = "0.0.0.0"
    log_level: str = "INFO"
    redis_url: str | None = None
    model_path: str = "models/lda_pipeline.joblib"
    max_upload_size: int = 100 * 1024 * 1024  # 100MB

    class Config:
        env_file = ".env"

settings = Settings()
```
