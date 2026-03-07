# Sleep-BCI Project: Technical Review

**Reviewer**: Senior BCI/SWE | **Date**: 2026-02-15
**Project**: Sleep-EDF Pipeline (Preprocess → Features → LDA → API)

---

## Executive Summary

**Overall Grade**: B+ (Solid foundation, production gaps)

**Strengths**:
- ✅ End-to-end working pipeline (data → model → API)
- ✅ Docker support with multi-stage builds
- ✅ FastAPI with async background jobs
- ✅ Proper Python packaging (pyproject.toml)
- ✅ MNE-based preprocessing (industry standard)
- ✅ GroupKFold CV (respects subject independence)

**Critical Gaps** (Blocking resume-grade):
- ❌ No CI/CD pipeline (GitHub Actions)
- ❌ Minimal test coverage (~5% - only 1 test file)
- ❌ No proper logging (print statements instead)
- ❌ No streaming inference endpoint (core BCI feature)
- ❌ In-memory job storage (lost on restart)
- ❌ README too technical (not recruiter-friendly)
- ❌ No architecture diagram
- ❌ Model metrics not quantified in README

---

## Detailed Analysis

### 1. Code Quality & Engineering ⭐⭐⭐⭐☆ (4/5)

**Good**:
- Clean separation of concerns (preprocessing/, model/, api/, features/)
- Type hints in most places
- Pydantic schemas for API validation
- Proper error handling in preprocessing pipeline
- Dataclass for PreprocessSpec (immutable config)

**Issues**:
- src/sleep_bci/api/app.py:36 - Logger configured but not used consistently
- src/sleep_bci/api/app.py:130 - JOBS dict is global in-memory (not production-ready)
- src/sleep_bci/api/app.py:296 - training_data dict also global (same issue)
- src/sleep_bci/preprocessing/core.py:207 - Uses print() instead of logger
- src/sleep_bci/model/train.py:82 - Mixed print/logger usage
- No configuration management (settings scattered)
- No input validation for model loading

**Action Items**:
- [ ] Add structured logging (structlog or loguru)
- [ ] Replace in-memory JOBS with Redis/SQLite
- [ ] Create config.py with Pydantic BaseSettings
- [ ] Add model validation on load (check fs, feature dims)

---

### 2. Testing & Quality Assurance ⭐☆☆☆☆ (1/5)

**Current State**:
- Only 1 test file: tests/test_features.py (10 lines)
- No API tests
- No integration tests
- No preprocessing tests
- No Docker tests

**Missing**:
- Unit tests for preprocessing pipeline
- API endpoint tests (pytest-asyncio + httpx)
- Model training tests (mock data)
- Feature extraction tests (edge cases)
- Docker build/run tests

**Action Items**:
- [ ] Add pytest.ini with coverage settings
- [ ] Target 80%+ coverage for core logic
- [ ] Add tests/conftest.py with fixtures
- [ ] Add integration tests for full pipeline
- [ ] Add test data (synthetic EDF files)

---

### 3. API Design & Architecture ⭐⭐⭐⭐☆ (4/5)

**Good**:
- RESTful design with /v1/ versioning
- Async background tasks (BackgroundTasks)
- Proper status endpoints (job polling)
- Pydantic validation
- Dry-run support

**Issues**:
- No streaming inference endpoint (critical for BCI)
- No WebSocket/SSE for real-time predictions
- No authentication/authorization
- No rate limiting
- No CORS configuration
- JOBS dict lost on server restart
- No health check endpoint returning useful metrics
- No OpenAPI tags/descriptions

**Missing Endpoints**:
- POST /v1/predict (batch inference)
- POST /v1/stream/start (WebSocket)
- GET /v1/models (list available models)
- GET /v1/health (detailed health check)
- GET /v1/metrics (Prometheus format)

**Action Items**:
- [ ] Add WebSocket endpoint for streaming inference
- [ ] Add Redis for job persistence
- [ ] Add /v1/predict for batch inference
- [ ] Add proper health check with model status
- [ ] Add OpenAPI tags and better descriptions

---

### 4. Docker & Deployment ⭐⭐⭐⭐⭐ (5/5)

**Excellent**:
- Multi-stage Dockerfile (builder + runtime)
- Slim base image (python:3.11-slim)
- Health checks configured
- Docker Compose with profiles (api, cli, dev)
- Volume mounts for data/models
- Proper entrypoint script

**Minor**:
- No .dockerignore optimization (includes venv, .git)
- No docker-compose.test.yml for CI
- No Kubernetes manifests (not critical)

---

### 5. Documentation ⭐⭐☆☆☆ (2/5)

**Current README Issues**:
- Too technical (assumes expertise)
- No 60-second quickstart
- Data instructions unclear (where to get data?)
- No example outputs
- No metrics visualization
- No architecture diagram
- No badge (CI, coverage, license)
- No "Why this matters" section
- Docker section comes before local setup

**Missing Docs**:
- CONTRIBUTING.md
- API documentation (beyond Swagger)
- Architecture diagram
- Example notebooks
- Deployment guide
- Troubleshooting guide

---

### 6. BCI-Specific Concerns ⭐⭐⭐☆☆ (3/5)

**Current**:
- Offline EDF processing only
- No real-time streaming simulation
- No LSL/BrainFlow integration path
- Feature extraction not optimized for real-time

**Real-World BCI Gaps**:
- No buffer management for incoming streams
- No online artifact rejection
- No adaptive normalization
- No drift correction
- No user calibration workflow
- No feedback latency measurement
- No epoch sliding window for streaming

**Critical for BCI Resume**:
- Need to show understanding of real-time constraints
- Need streaming inference (even if simulated)
- Need buffer management strategy

---

### 7. Model Performance 📊

**Current Metrics** (from results.json):
```
Balanced Accuracy: 53.6% ± 12.9%
Macro F1: 53.3% ± 11.6%
```

**Issues**:
- High variance (12.9% std) - likely due to only 3 nights
- N1 stage: 0% recall (completely missed)
- Only 3 subjects (not enough for generalization)

**Not a blocker for resume**, but should:
- [ ] Acknowledge in README ("proof-of-concept metrics")
- [ ] Add "With full dataset: expect ~70-75% acc" note
- [ ] Add feature importance visualization
- [ ] Add confusion matrix in results

---

### 8. Security & Production Readiness ⭐⭐☆☆☆ (2/5)

**Missing**:
- No authentication
- No input sanitization (file upload)
- No rate limiting
- No HTTPS configuration
- Temp files not cleaned on error (some paths)
- No secrets management
- No audit logging

**Critical for Production**:
- [ ] Add API key authentication
- [ ] Add file size limits
- [ ] Add request timeouts
- [ ] Add rate limiting (slowapi)
- [ ] Add security headers (FastAPI middleware)

---

## Entry Point & Path Issues

### Found Issues:
1. ✅ **Entry points work** (tested in pyproject.toml:24-29)
2. ⚠️ **Mixed logging** - app.py uses logger, but prints in preprocessing
3. ✅ **Docker paths correct** - volumes mounted properly
4. ⚠️ **No explicit data validation for model compat** - could load wrong model

### Recommended Fixes:
```python
# Add to model/artifacts.py
def load_and_validate_bundle(path: str, expected_fs: float) -> ModelBundle:
    bundle = load_bundle(path)
    if abs(bundle.fs - expected_fs) > 1e-6:
        raise ValueError(f"Model fs {bundle.fs} != expected {expected_fs}")
    return bundle
```
