# BCI LSL Integration & Hardware Guide

---

## Stage 3: LSL Integration (2-3 weeks)

**Goal**: Connect to real EEG device via Lab Streaming Layer

**What is LSL?**
- Industry-standard protocol for streaming time-series data
- Used by: OpenBCI, Emotiv, NeuroSky, g.tec, ANT Neuro
- Language bindings: Python, C++, MATLAB, Unity

**Architecture**:
```
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│  EEG Device  │──LSL──│  Inlet       │──────▶│  Your Code   │
│  (Outlet)    │ (TCP) │  (Consumer)  │ Queue │  (Process)   │
└──────────────┘       └──────────────┘       └──────────────┘
```

**Implementation**:

```python
# src/sleep_bci/stream/lsl_client.py

import pylsl
import numpy as np

class LSLClient:
    """Receive EEG data from LSL outlet"""

    def __init__(self, stream_name: str = "SleepEEG", channel: str = "EEG Fpz-Cz"):
        # Find LSL stream
        print(f"Looking for stream: {stream_name}...")
        streams = pylsl.resolve_stream('name', stream_name)
        if not streams:
            raise RuntimeError(f"No LSL stream found with name '{stream_name}'")

        self.inlet = pylsl.StreamInlet(streams[0])
        self.fs = self.inlet.info().nominal_srate()
        self.channel = channel

    def pull_sample(self) -> tuple[np.ndarray, float]:
        """Pull single sample (blocking)"""
        sample, timestamp = self.inlet.pull_sample()
        return np.array(sample), timestamp

    def pull_chunk(self, max_samples: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        """Pull batch of samples (non-blocking)"""
        samples, timestamps = self.inlet.pull_chunk(max_samples=max_samples)
        return np.array(samples), np.array(timestamps)
```

**Full Pipeline**:

```python
# src/sleep_bci/stream/live_inference.py

from sleep_bci.stream.lsl_client import LSLClient
from sleep_bci.stream.buffer import RingBuffer
from sleep_bci.features.bandpower import extract_features_epoch
from sleep_bci.model.artifacts import load_bundle

def live_inference_loop(model_path: str, stream_name: str = "SleepEEG"):
    # Setup
    lsl = LSLClient(stream_name)
    buffer = RingBuffer(capacity=int(30 * lsl.fs))  # 30s window
    model = load_bundle(model_path)

    print(f"Listening to {stream_name} @ {lsl.fs} Hz")

    # Main loop
    while True:
        sample, timestamp = lsl.pull_sample()  # Blocking
        buffer.push(sample)

        if buffer.is_full():
            epoch = buffer.get_epoch()

            # Apply preprocessing (online filtering)
            epoch = apply_online_filter(epoch, lsl.fs)

            # Predict
            feats = extract_features_epoch(epoch.reshape(1, -1), fs=lsl.fs)
            pred = model.pipeline.predict([feats])[0]
            stage = model.label_map[pred]

            print(f"[{timestamp:.2f}] Predicted: {stage}")

            # Optional: Send to WebSocket/dashboard
            # await websocket.send_json({"stage": stage, "timestamp": timestamp})
```

**Testing Without Hardware**:

```python
# src/sleep_bci/stream/lsl_simulator.py

import pylsl
import numpy as np
import time

class LSLSimulator:
    """Simulate EEG device by publishing LSL stream from NPZ file"""

    def __init__(self, stream_name: str = "SleepEEG", fs: float = 100.0):
        info = pylsl.StreamInfo(stream_name, 'EEG', 1, fs, 'float32', 'sim123')
        self.outlet = pylsl.StreamOutlet(info)
        self.fs = fs

    def stream_from_npz(self, npz_path: str, realtime: bool = True):
        data = np.load(npz_path)
        X = data["X"]  # (n_epochs, 1, 3000)

        for epoch in X:
            for sample in epoch[0]:
                self.outlet.push_sample([float(sample)])
                if realtime:
                    time.sleep(1 / self.fs)  # Simulate 100 Hz
```

**Usage**:
```bash
# Terminal 1: Start simulator
python -c "from sleep_bci.stream.lsl_simulator import LSLSimulator; \
           sim = LSLSimulator(); \
           sim.stream_from_npz('data/processed/SC4001E0.npz')"

# Terminal 2: Run live inference
python -m sleep_bci.stream.live_inference --model models/lda_pipeline.joblib
```

---

## Stage 4: Production Features (3-4 weeks)

**Goal**: Handle real-world edge cases

### 4.1 Online Artifact Rejection

**Problem**: Eye blinks, muscle artifacts, electrode noise corrupt EEG

**Solution**: Real-time artifact detection

```python
def detect_artifact(epoch: np.ndarray, threshold: float = 100.0) -> bool:
    """Simple artifact detection"""
    # Check for amplitude clipping
    if np.max(np.abs(epoch)) > threshold:
        return True

    # Check for flat line (electrode disconnected)
    if np.std(epoch) < 1.0:
        return True

    return False

# In inference loop:
if detect_artifact(epoch):
    print("Artifact detected, skipping epoch")
    continue  # Don't predict
```

**Advanced**: Use ICA-based artifact removal (mne.preprocessing.ICA)

---

### 4.2 Adaptive Normalization

**Problem**: EEG amplitude drifts over time (electrode impedance, skin conductance)

**Solution**: Exponential moving average normalization

```python
class AdaptiveScaler:
    def __init__(self, alpha: float = 0.01):
        self.mean = 0.0
        self.std = 1.0
        self.alpha = alpha  # Learning rate

    def update(self, epoch: np.ndarray):
        """Update statistics with new epoch"""
        self.mean = (1 - self.alpha) * self.mean + self.alpha * np.mean(epoch)
        self.std = (1 - self.alpha) * self.std + self.alpha * np.std(epoch)

    def transform(self, epoch: np.ndarray) -> np.ndarray:
        """Normalize epoch using current statistics"""
        return (epoch - self.mean) / (self.std + 1e-8)
```

---

### 4.3 Latency Optimization

**Target**: <100ms end-to-end latency

**Breakdown**:
- Data acquisition: 0-30ms (depends on LSL buffering)
- Preprocessing (filtering): 5-10ms
- Feature extraction: 10-20ms
- Model inference: 5-10ms
- Output transmission: 1-5ms

**Total**: ~50ms (comfortable margin)

**Optimization Tips**:
```python
# 1. Pre-allocate arrays
buffer = np.zeros((1, 3000), dtype=np.float32)

# 2. Use scipy.signal.sosfilt (faster than filtfilt)
import scipy.signal
sos = scipy.signal.butter(4, [0.3, 30], btype='band', fs=100, output='sos')
filtered = scipy.signal.sosfilt(sos, epoch)

# 3. Vectorize feature extraction
def extract_features_fast(epoch: np.ndarray, fs: float) -> np.ndarray:
    # Use numpy broadcasting, avoid loops
    ...

# 4. Profile hot paths
import cProfile
cProfile.run('live_inference_loop()')
```

---

### 4.4 Threading Model

**Problem**: Blocking LSL reads prevent parallel processing

**Solution**: Producer-Consumer pattern

```python
import threading
import queue

class StreamingPipeline:
    def __init__(self, model_path: str):
        self.lsl = LSLClient()
        self.buffer = RingBuffer(3000)
        self.model = load_bundle(model_path)
        self.queue = queue.Queue(maxsize=10)

    def producer_thread(self):
        """Pull samples from LSL"""
        while True:
            sample, timestamp = self.lsl.pull_sample()
            self.buffer.push(sample)
            if self.buffer.is_full():
                epoch = self.buffer.get_epoch()
                self.queue.put((epoch, timestamp))

    def consumer_thread(self):
        """Process epochs"""
        while True:
            epoch, timestamp = self.queue.get()
            feats = extract_features_epoch(epoch, fs=self.lsl.fs)
            pred = self.model.pipeline.predict([feats])[0]
            print(f"Prediction: {pred}")

    def run(self):
        threading.Thread(target=self.producer_thread, daemon=True).start()
        threading.Thread(target=self.consumer_thread, daemon=True).start()
        threading.Event().wait()  # Run forever
```

---

## Hardware Integration Guide

### Recommended Devices (by Budget)

| Device | Price | Channels | Sample Rate | LSL Support |
|--------|-------|----------|-------------|-------------|
| **OpenBCI Cyton** | $200 | 8 | 250 Hz | ✅ Native |
| **Muse S** | $400 | 4 | 256 Hz | ⚠️ Via BrainFlow |
| **g.tec g.USBamp** | $5000+ | 16+ | 512 Hz | ✅ Native |
| **Emotiv EPOC X** | $850 | 14 | 128 Hz | ✅ Native |

**Budget Option**: Use **OpenBCI Cyton** ($200) + **OpenBCI GUI** (free) for LSL streaming

---

### Setup: OpenBCI → LSL → sleep-bci

**Step 1: Install OpenBCI GUI**
```bash
# Download from: https://openbci.com/downloads
# Start GUI, select "Live" mode
```

**Step 2: Enable LSL Streaming**
```
1. Open OpenBCI GUI
2. Click "Networking" → "LSL"
3. Enable "Stream data via LSL"
4. Set stream name: "SleepEEG"
```

**Step 3: Test LSL Connection**
```python
import pylsl

# Find streams
streams = pylsl.resolve_streams()
for s in streams:
    print(f"Found: {s.name()} @ {s.nominal_srate()} Hz")
```

**Step 4: Run sleep-bci**
```bash
python -m sleep_bci.stream.live_inference --model models/lda_pipeline.joblib --stream SleepEEG
```

---

## Testing & Validation

### Latency Measurement

```python
import time

def measure_latency(model_path: str, n_epochs: int = 100):
    latencies = []

    for _ in range(n_epochs):
        epoch = np.random.randn(1, 3000)

        start = time.perf_counter()
        feats = extract_features_epoch(epoch, fs=100)
        pred = model.pipeline.predict([feats])
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # ms

    print(f"Mean latency: {np.mean(latencies):.2f} ms")
    print(f"P95 latency: {np.percentile(latencies, 95):.2f} ms")
```

**Target**: <100ms mean, <150ms p95

---

### End-to-End Test

```python
def test_live_pipeline():
    # 1. Start LSL simulator
    sim = LSLSimulator()
    threading.Thread(target=lambda: sim.stream_from_npz("data/processed/SC4001E0.npz")).start()

    # 2. Wait for stream
    time.sleep(2)

    # 3. Run inference for 60s
    pipeline = StreamingPipeline("models/lda_pipeline.joblib")
    pipeline.run(duration=60)

    # 4. Verify predictions were generated
    assert len(pipeline.predictions) > 0
```

---

## Resume-Worthy Metrics

**Before** (offline):
> "Processed EEG data and trained sleep stage classifier"

**After** (real-time):
> "Engineered real-time BCI pipeline processing live EEG streams at 100 Hz with <50ms prediction latency, supporting continuous online classification via LSL protocol"

**Key Numbers**:
- Streaming rate: 100 Hz
- Latency: <50ms (p50), <100ms (p95)
- Throughput: 100 epochs/sec (if 1s sliding window)
- Buffer size: 3000 samples (30s @ 100 Hz)

---

## Common Pitfalls

### 1. **Clock Drift**
- **Problem**: LSL timestamps ≠ system time
- **Solution**: Use LSL's `local_clock()` for synchronization

### 2. **Sample Drops**
- **Problem**: Network/CPU spikes cause missed samples
- **Solution**: Interpolate short gaps (<100ms), reject long gaps

### 3. **Filter Initialization**
- **Problem**: IIR filters have transient response (first few samples are wrong)
- **Solution**: Pre-fill filter state with neutral values, or use FIR

### 4. **Feature Mismatch**
- **Problem**: Offline model trained on different features than online
- **Solution**: Save feature extraction params with model (fs, bandpass, etc.)

---

## Next Steps

1. **Implement Stage 2** (Buffer) - 1 week
2. **Install pylsl** - `pip install pylsl`
3. **Test with simulator** - Prove <100ms latency
4. **Buy OpenBCI** ($200) - If serious about hardware
5. **Implement Stage 3** (LSL) - 2 weeks
6. **Deploy live demo** - Record video for resume

---

## Resources

### Libraries
- **pylsl**: https://github.com/labstreaminglayer/liblsl-Python
- **BrainFlow**: https://brainflow.org/ (alternative to LSL)
- **MNE-Python**: https://mne.tools/stable/auto_tutorials/preprocessing/50_artifact_correction_ssp.html

### Tutorials
- LSL Quickstart: https://labstreaminglayer.readthedocs.io/
- OpenBCI + LSL: https://docs.openbci.com/Software/CompatibleThirdPartySoftware/LSL/
- Real-time EEG Processing: https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html

---

**Key Takeaway**: Moving from offline → real-time is 80% engineering (buffers, threading, latency) and 20% algorithms. Focus on **proving <100ms latency** with simulated data first, then add hardware.
