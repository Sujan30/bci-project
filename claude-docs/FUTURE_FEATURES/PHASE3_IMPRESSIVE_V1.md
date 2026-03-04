# Phase 3: Impressive v1 (1 month)

**Objective**: Senior-level differentiation

### 3.1 Live Streaming Simulation ⏱️ 8h | Complexity: L

**Files to create**:
- `src/sleep_bci/stream/lsl_simulator.py`
- `examples/live_demo.py`
- `docs/lsl-integration.md`

**Acceptance Criteria**:
- [ ] LSL outlet publishing EEG data
- [ ] API consuming LSL stream
- [ ] Buffer management (sliding window)
- [ ] Visualization (matplotlib animation)
- [ ] <50ms latency demonstrated

**Implementation**:
```python
# src/sleep_bci/stream/lsl_simulator.py
import pylsl
import numpy as np

class EEGSimulator:
    def __init__(self, fs=100, n_channels=1):
        info = pylsl.StreamInfo('SleepEEG', 'EEG', n_channels, fs, 'float32', 'sim123')
        self.outlet = pylsl.StreamOutlet(info)

    def stream_from_npz(self, npz_path: str):
        data = np.load(npz_path)
        X = data["X"]  # (n_epochs, 1, 3000)

        for epoch in X:
            for sample in epoch[0]:
                self.outlet.push_sample([float(sample)])
                time.sleep(1/100)  # 100Hz
```

---

### 3.2 Web Dashboard ⏱️ 16h | Complexity: L

**Files to create**:
- `frontend/` (React or vanilla JS)
- `src/sleep_bci/api/static.py`

**Acceptance Criteria**:
- [ ] Real-time visualization (Chart.js)
- [ ] Job monitoring UI
- [ ] Model upload/management
- [ ] WebSocket live predictions
- [ ] Deployed with FastAPI staticfiles

**Tech Stack**: React + Vite OR vanilla JS + htmx (simpler)

---

### 3.3 Model Comparison ⏱️ 6h | Complexity: M

**Files to create**:
- `src/sleep_bci/model/xgboost_train.py`
- `notebooks/model_comparison.ipynb`

**Acceptance Criteria**:
- [ ] Train XGBoost classifier
- [ ] Compare LDA vs XGBoost vs Random Forest
- [ ] Metrics table in README
- [ ] Feature importance plots

---

### 3.4 Prometheus Metrics ⏱️ 4h | Complexity: M

**Files to create**:
- `src/sleep_bci/api/metrics.py`

**Acceptance Criteria**:
- [ ] /metrics endpoint (Prometheus format)
- [ ] Request latency histogram
- [ ] Job queue depth
- [ ] Model inference time
- [ ] Error rate counter

---

### 3.5 Notebook Analysis ⏱️ 4h | Complexity: S

**Files to create**:
- `notebooks/01_data_exploration.ipynb`
- `notebooks/02_feature_analysis.ipynb`
- `notebooks/03_model_evaluation.ipynb`

**Acceptance Criteria**:
- [ ] EDA on Sleep-EDF
- [ ] Feature importance
- [ ] Confusion matrix analysis
- [ ] Cross-validation visualization

---

## Dependencies & Tools

### Add to pyproject.toml:
```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "pytest-asyncio",
    "httpx",
    "ruff",
    "mypy",
]
streaming = [
    "websockets",
    "redis",
    "pylsl",  # LSL integration
]
analysis = [
    "jupyter",
    "matplotlib",
    "seaborn",
    "pandas",
]
```
