"""
End-to-end integration test: full Sleep-BCI pipeline without mocks.

Covers:
  1. Generate synthetic NPZ data
  2. POST /v1/train → enqueue training job
  3. Poll /v1/train/{job_id} until succeeded
  4. GET /v1/models → verify model appears
  5. WebSocket /v1/stream → send 12 chunks of 250 samples (= 1 epoch)
  6. Assert prediction received with a valid sleep stage label

Run:
    pytest tests/test_integration.py -v -m integration

Or with the full suite:
    pytest -m "not integration"   # skip integration tests
    pytest -m integration          # run only integration tests
"""
import os
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sleep_bci.api.app import app
from sleep_bci.config import settings


@pytest.mark.integration
def test_full_pipeline_e2e(tmp_path):
    """Full pipeline: generate data → train → verify model → stream chunks → predict."""
    client = TestClient(app)

    # ── Step 1: Generate synthetic NPZ data (2 nights, 6 epochs each) ────────
    sfreq = 100.0
    n_samples = 3000  # 30s at 100 Hz
    n_epochs_per_night = 6
    n_nights = 2  # minimum for n_splits=2

    for night_id in range(n_nights):
        rng = np.random.default_rng(seed=night_id)
        X = rng.standard_normal((n_epochs_per_night, 1, n_samples)).astype(np.float32)
        y = rng.integers(0, 5, n_epochs_per_night).astype(np.int32)
        np.savez(
            tmp_path / f"night{night_id:02d}.npz",
            X=X,
            y=y,
            sfreq=sfreq,
            channel="EEG Fpz-Cz",
            epoch_sec=30,
            bandpass=np.array([0.3, 30.0]),
            notch_hz=-1.0,
            dataset="integration-test",
        )

    npz_dir = str(tmp_path)

    # Determine models directory (where .joblib files live)
    models_dir = os.path.dirname(os.path.abspath(settings.model_path))
    model_id = "test_integration_e2e"
    model_out = os.path.join(models_dir, f"{model_id}.joblib")

    try:
        # ── Step 2: POST /v1/train ─────────────────────────────────────────
        train_resp = client.post(
            "/v1/train",
            json={
                "npz_dir": npz_dir,
                "model_out": model_out,
                "fs": 100.0,
                "n_splits": 2,
                "model_type": "lda",
            },
        )
        assert train_resp.status_code == 200, f"Train POST failed: {train_resp.text}"
        job_id = train_resp.json()["job_id"]
        assert job_id, "Expected job_id in response"

        # ── Step 3: Poll /v1/train/{job_id} until succeeded (60s timeout) ──
        deadline = time.monotonic() + 60
        final_status = None

        while time.monotonic() < deadline:
            status_resp = client.get(f"/v1/train/{job_id}")
            assert status_resp.status_code == 200, f"Status GET failed: {status_resp.text}"
            final_status = status_resp.json()["status"]
            if final_status in ("succeeded", "failed"):
                break
            time.sleep(0.2)

        assert final_status == "succeeded", (
            f"Training job did not succeed within timeout. Final status: {final_status}"
        )

        # ── Step 4: GET /v1/models — verify model appears ──────────────────
        models_resp = client.get("/v1/models")
        assert models_resp.status_code == 200, f"Models GET failed: {models_resp.text}"
        models_data = models_resp.json()
        assert model_id in models_data["models"], (
            f"Model '{model_id}' not found in listing: {models_data['models']}"
        )

        # ── Step 5: WebSocket streaming — 12 chunks × 250 samples = 1 epoch ─
        chunk_size = 250
        n_chunks = 12  # 12 × 250 = 3000 = one complete epoch
        assert chunk_size * n_chunks == 3000, "Chunk math must equal one full epoch"

        rng = np.random.default_rng(seed=99)
        fake_signal = rng.standard_normal(chunk_size).tolist()

        prediction_received = None

        with client.websocket_connect(f"/v1/stream?model_id={model_id}") as ws:
            for i in range(n_chunks):
                ws.send_json({
                    "chunk": fake_signal,
                    "fs": 100.0,
                    "session_id": "e2e-integration-test",
                })
                resp = ws.receive_json()

                if "stage" in resp:
                    # Got a prediction — buffer is full
                    prediction_received = resp
                elif "error" in resp:
                    pytest.fail(f"WebSocket error on chunk {i}: {resp}")
                # Otherwise it's a buffering status message — continue

        # ── Step 6: Assert valid prediction was received ───────────────────
        assert prediction_received is not None, (
            "No prediction received after 12 chunks (3000 samples = 1 epoch)"
        )
        valid_stages = {"W", "N1", "N2", "N3", "REM"}
        assert prediction_received["stage"] in valid_stages, (
            f"Invalid stage label: {prediction_received['stage']!r}. "
            f"Expected one of {valid_stages}"
        )
        assert "confidence" in prediction_received, "Prediction missing 'confidence' field"
        assert "latency_ms" in prediction_received, "Prediction missing 'latency_ms' field"
        assert 0.0 <= prediction_received["confidence"] <= 1.0, (
            f"Confidence out of range: {prediction_received['confidence']}"
        )

    finally:
        # Cleanup test model artifacts
        for path in [model_out, os.path.join(models_dir, "results.json")]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
        for fname in ["confusion_matrix.json", "confusion_matrix.png"]:
            p = os.path.join(models_dir, fname)
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
