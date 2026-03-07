# BCI Realism: From Offline EDF to Live Device Streaming

**Objective**: Transform sleep-bci from batch offline classifier → real-time BCI system

---

## Current State: Offline EDF Processing

### What We Have

```
┌──────────────┐
│  EDF Files   │ (recorded overnight, 8+ hours)
└──────┬───────┘
       │ Load entire file
       ▼
┌──────────────┐
│ Preprocessing│ (batch: all epochs at once)
└──────┬───────┘
       │ Write to disk
       ▼
┌──────────────┐
│ Feature Ext. │ (batch: extract_features_batch)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ LDA Predict  │ (batch: all epochs)
└──────────────┘
```

**Characteristics**:
- ✅ Simple: Load → Process → Save
- ✅ Reproducible: Same input → Same output
- ✅ No timing constraints
- ❌ Not real-time (takes minutes to process 1 night)
- ❌ No live feedback
- ❌ Requires pre-recorded data

---

## Target State: Real-Time Streaming BCI

### What We Need

```
┌──────────────────────┐
│  Live EEG Device     │ (LSL stream: 100 samples/sec)
└──────────┬───────────┘
           │ Network stream (TCP/UDP)
           ▼
┌──────────────────────┐
│  Ring Buffer         │ (30s sliding window)
│  [oldest ... newest] │
└──────────┬───────────┘
           │ Every 1s (or 30s)
           ▼
┌──────────────────────┐
│  Online Preprocessing│ (bandpass, artifact rejection)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Feature Extraction  │ (<50ms latency)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  LDA Predict         │ (<10ms)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Output (WebSocket)  │ → Dashboard, closed-loop therapy
└──────────────────────┘
```

**Characteristics**:
- ✅ Real-time: Predictions within 100ms of data arrival
- ✅ Continuous: Runs indefinitely
- ✅ Live feedback: Results available immediately
- ⚠️ Complex: Buffer management, drift, artifacts
- ⚠️ Timing-critical: <1s latency for usability

---

## Key Differences: Offline vs Real-Time

| Aspect | Offline (Current) | Real-Time (Target) |
|--------|-------------------|--------------------|
| **Data Source** | EDF files (pre-recorded) | LSL stream (live device) |
| **Processing** | Batch (all epochs at once) | Streaming (epoch-by-epoch) |
| **Timing** | No constraints | <100ms latency |
| **Buffer** | Load entire file into memory | Sliding window (30s) |
| **Artifacts** | Offline rejection | Online detection + handling |
| **Normalization** | Global (all data) | Adaptive (recent history) |
| **Output** | File (NPZ, JSON) | WebSocket, UDP, or callback |
| **Failure Mode** | Retry entire file | Skip/interpolate bad epochs |

---

## Evolution Roadmap: 4 Stages

### Stage 1: Simulated Streaming (Current → 1 week)

**Goal**: Prove real-time inference without hardware

**Implementation**:
```python
# src/sleep_bci/stream/simulate.py (already exists)

def simulate_stream(npz_path: str, model_path: str, realtime: bool = True):
    """Stream pre-recorded data at realistic speed"""
    data = np.load(npz_path)
    X, y = data["X"], data["y"]
    bundle = load_bundle(model_path)

    for epoch, true_label in zip(X, y):
        if realtime:
            time.sleep(30)  # Simulate 30s epoch

        # Extract features + predict (same as offline)
        feats = extract_features_epoch(epoch, fs=bundle.fs)
        pred = bundle.pipeline.predict([feats])[0]

        print(f"True: {true_label} | Pred: {pred}")
```

**Additions Needed**:
- [x] Basic simulation exists (`sleepbci-simulate`)
- [ ] Add WebSocket output (instead of print)
- [ ] Add latency measurement
- [ ] Add buffer visualization

**Outcome**: WebSocket client receives predictions every 30s

---

### Stage 2: Buffer Management (1-2 weeks)

**Goal**: Handle incoming sample-by-sample data with sliding windows

**Challenge**: EEG devices send samples at 100 Hz (every 10ms), but we need 30s epochs (3000 samples)

**Solution**: Ring Buffer

```python
# src/sleep_bci/stream/buffer.py

import numpy as np
from collections import deque

class RingBuffer:
    """Thread-safe ring buffer for streaming EEG"""

    def __init__(self, capacity: int, n_channels: int = 1):
        self.capacity = capacity  # e.g., 3000 for 30s @ 100Hz
        self.buffer = deque(maxlen=capacity)
        self.n_channels = n_channels

    def push(self, sample: np.ndarray):
        """Add single sample (or mini-batch)"""
        self.buffer.append(sample)

    def get_epoch(self) -> np.ndarray | None:
        """Extract full epoch if buffer is full"""
        if len(self.buffer) < self.capacity:
            return None
        return np.array(list(self.buffer))

    def is_full(self) -> bool:
        return len(self.buffer) == self.capacity
```

**Usage**:
```python
buffer = RingBuffer(capacity=3000, n_channels=1)

# As samples arrive...
for sample in lsl_stream:
    buffer.push(sample)

    if buffer.is_full():
        epoch = buffer.get_epoch()
        feats = extract_features_epoch(epoch, fs=100)
        pred = model.predict([feats])
```

**Sliding Window Strategy**:
- **Non-overlapping** (simpler): New prediction every 30s (3000 new samples)
- **Overlapping** (smoother): New prediction every 1s (100 new samples, 29s overlap)

**Trade-offs**:
| Strategy | Latency | Compute | Smoothness |
|----------|---------|---------|------------|
| 30s non-overlap | 30s | Low | Jumpy |
| 1s overlap | 1s | High (30x more predictions) | Smooth |

**Recommendation**: Start with 30s, optimize to 5-10s overlap later
