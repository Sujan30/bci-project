import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO

from sleep_bci.api.schemas import JobStatus
from sleep_bci.api.app import MODEL_CACHE


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, World!"}


# ---------------------------------------------------------------------------
# /upload
# ---------------------------------------------------------------------------

def test_upload_rejects_non_edf(client):
    response = client.post(
        "/v1/upload",
        files=[("files", ("data.txt", BytesIO(b"bad"), "text/plain"))],
    )
    assert response.status_code == 400
    assert "not an EDF file" in response.json()["detail"]


def test_upload_missing_psg(client):
    response = client.post(
        "/v1/upload",
        files=[("files", ("SC4001E0-PSG-Hypnogram.edf", BytesIO(b"fake"), "application/octet-stream"))],
    )
    assert response.status_code == 400
    assert "PSG" in response.json()["detail"]


def test_upload_missing_hypnogram(client):
    response = client.post(
        "/v1/upload",
        files=[("files", ("SC4001E0-PSG.edf", BytesIO(b"fake"), "application/octet-stream"))],
    )
    assert response.status_code == 400
    assert "Hypnogram" in response.json()["detail"]


def test_upload_valid_pair(client, tmp_path):
    """Upload matching PSG + Hypnogram pair; mock discover_and_validate to skip real EDF parsing."""
    psg_name = "SC4001E0-PSG.edf"
    hyp_name = "SC4001E0-Hypnogram.edf"

    fake_pairs = [(f"/tmp/{psg_name}", f"/tmp/{hyp_name}")]

    with patch("sleep_bci.api.app.discover_and_validate", return_value=fake_pairs):
        response = client.post(
            "/v1/upload",
            files=[
                ("files", (psg_name, BytesIO(b"fake-psg"), "application/octet-stream")),
                ("files", (hyp_name, BytesIO(b"fake-hyp"), "application/octet-stream")),
            ],
        )

    assert response.status_code == 200
    body = response.json()
    assert body["matched_pairs"] == 1
    assert "session_id" in body


# ---------------------------------------------------------------------------
# /v1/preprocess
# ---------------------------------------------------------------------------

def test_preprocess_missing_raw_dir(client):
    payload = {
        "dataset": {"raw_dir": "/nonexistent/path/that/does/not/exist"},
        "output": {},
    }
    response = client.post("/v1/preprocess", json=payload)
    assert response.status_code == 400
    assert "not found" in response.json()["detail"].lower()


def test_preprocess_dry_run(client, tmp_path):
    """Dry run should return matched pairs without launching a background job."""
    psg_path = str(tmp_path / "SC4001E0-PSG.edf")
    hyp_path = str(tmp_path / "SC4001E0-Hypnogram.edf")
    open(psg_path, "w").close()
    open(hyp_path, "w").close()

    fake_pairs = [(psg_path, hyp_path)]

    with patch("sleep_bci.api.app.discover_and_validate", return_value=fake_pairs):
        response = client.post(
            "/v1/preprocess",
            json={
                "dataset": {"raw_dir": str(tmp_path)},
                "output": {},
                "dry_run": True,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["matched_pairs"] == 1


def test_preprocess_creates_job(client, tmp_path):
    """A non-dry-run request should create a job and return job_id."""
    psg_path = str(tmp_path / "SC4001E0-PSG.edf")
    hyp_path = str(tmp_path / "SC4001E0-Hypnogram.edf")
    open(psg_path, "w").close()
    open(hyp_path, "w").close()

    fake_pairs = [(psg_path, hyp_path)]
    out_dir = str(tmp_path / "output")

    with patch("sleep_bci.api.app.discover_and_validate", return_value=fake_pairs), \
         patch("sleep_bci.api.app.run_preprocess_job"):
        response = client.post(
            "/v1/preprocess",
            json={
                "dataset": {"raw_dir": str(tmp_path)},
                "output": {"out_dir": out_dir},
                "dry_run": False,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert "job_id" in body
    assert body["status"] == JobStatus.queued


# ---------------------------------------------------------------------------
# /v1/preprocess/{job_id}
# ---------------------------------------------------------------------------

def test_get_preprocess_status_not_found(client):
    response = client.get("/v1/preprocess/does-not-exist")
    assert response.status_code == 404


def test_get_preprocess_status_found(client, tmp_path):
    """After creating a job, its status endpoint should return 200."""
    psg_path = str(tmp_path / "SC4001E0-PSG.edf")
    hyp_path = str(tmp_path / "SC4001E0-Hypnogram.edf")
    open(psg_path, "w").close()
    open(hyp_path, "w").close()

    fake_pairs = [(psg_path, hyp_path)]
    out_dir = str(tmp_path / "output")

    with patch("sleep_bci.api.app.discover_and_validate", return_value=fake_pairs), \
         patch("sleep_bci.api.app.run_preprocess_job"):
        create_resp = client.post(
            "/v1/preprocess",
            json={
                "dataset": {"raw_dir": str(tmp_path)},
                "output": {"out_dir": out_dir},
                "dry_run": False,
            },
        )

    job_id = create_resp.json()["job_id"]
    status_resp = client.get(f"/v1/preprocess/{job_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["job_id"] == job_id


# ---------------------------------------------------------------------------
# /train
# ---------------------------------------------------------------------------

def test_train_missing_npz_dir(client):
    response = client.post(
        "/v1/train",
        json={"npz_dir": "/nonexistent/npz/dir", "n_splits": 2},
    )
    assert response.status_code == 400
    assert "not found" in response.json()["detail"].lower()


def test_train_no_npz_files(client, tmp_path):
    response = client.post(
        "/v1/train",
        json={"npz_dir": str(tmp_path), "n_splits": 2},
    )
    assert response.status_code == 400
    assert "No .npz" in response.json()["detail"]


def test_train_n_splits_too_large(client, sample_npz_dir):
    """n_splits > number of nights should be rejected."""
    response = client.post(
        "/v1/train",
        json={"npz_dir": sample_npz_dir, "n_splits": 100},
    )
    assert response.status_code == 400
    assert "n_splits" in response.json()["detail"]


def test_train_creates_job(client, sample_npz_dir):
    with patch("sleep_bci.api.app.update_training"):
        response = client.post(
            "/v1/train",
            json={"npz_dir": sample_npz_dir, "n_splits": 2},
        )

    assert response.status_code == 200
    body = response.json()
    assert "job_id" in body
    assert body["status"] == JobStatus.queued


# ---------------------------------------------------------------------------
# /v1/train/{job_id}
# ---------------------------------------------------------------------------

def test_get_training_status_not_found(client):
    response = client.get("/v1/train/does-not-exist")
    assert response.status_code == 404


def test_get_training_status_found(client, sample_npz_dir):
    with patch("sleep_bci.api.app.update_training"):
        create_resp = client.post(
            "/v1/train",
            json={"npz_dir": sample_npz_dir, "n_splits": 2},
        )

    job_id = create_resp.json()["job_id"]
    status_resp = client.get(f"/v1/train/{job_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["job_id"] == job_id


# ---------------------------------------------------------------------------
# /v1/stream (WebSocket)
# ---------------------------------------------------------------------------

REAL_NPZ = "data/processed/SC4001E0.npz"


def test_ws_model_not_found(client):
    with client.websocket_connect("/v1/stream?model_id=nonexistent") as ws:
        data = ws.receive_json()
        assert data["error"] == "MODEL_NOT_FOUND"


def test_ws_invalid_epoch_length(client, trained_bundle):
    MODEL_CACHE["test"] = trained_bundle
    with client.websocket_connect("/v1/stream?model_id=test") as ws:
        ws.send_json({"epoch": [0.0] * 2500})
        data = ws.receive_json()
        assert data["error"] == "INVALID_EPOCH_LENGTH"
        assert "2500" in data["message"]
        # connection stays open — send a second bad epoch to confirm
        ws.send_json({"epoch": [0.0] * 2500})
        data2 = ws.receive_json()
        assert data2["error"] == "INVALID_EPOCH_LENGTH"


def test_ws_stream_real_epoch(client, trained_bundle):
    """Classify one real EEG epoch from a Sleep-EDF recording."""
    if not os.path.exists(REAL_NPZ):
        pytest.skip("Real NPZ data not available")

    npz = np.load(REAL_NPZ)
    epoch = npz["X"][0][0].tolist()  # first real 30s epoch, shape (3000,)

    MODEL_CACHE["test"] = trained_bundle
    with client.websocket_connect("/v1/stream?model_id=test") as ws:
        ws.send_json({"epoch": epoch, "timestamp": 1234567890.0})
        resp = ws.receive_json()

    assert resp["stage"] in ("W", "N1", "N2", "N3", "REM")
    assert 0 <= resp["label"] <= 4
    assert 0.0 <= resp["confidence"] <= 1.0
    assert resp["latency_ms"] >= 0
    assert resp["timestamp"] == 1234567890.0


def test_ws_stream_multiple_real_epochs(client, trained_bundle):
    """Simulate streaming: classify 5 consecutive real epochs in one connection."""
    if not os.path.exists(REAL_NPZ):
        pytest.skip("Real NPZ data not available")

    npz = np.load(REAL_NPZ)
    MODEL_CACHE["test"] = trained_bundle

    with client.websocket_connect("/v1/stream?model_id=test") as ws:
        for i in range(5):
            epoch = npz["X"][i][0].tolist()
            ws.send_json({"epoch": epoch})
            resp = ws.receive_json()
            assert resp["stage"] in ("W", "N1", "N2", "N3", "REM"), f"Epoch {i}: {resp}"
            assert "confidence" in resp and "latency_ms" in resp


def test_ws_stream_without_timestamp(client, trained_bundle):
    """'timestamp' field should be absent from response when not sent."""
    if not os.path.exists(REAL_NPZ):
        pytest.skip("Real NPZ data not available")

    npz = np.load(REAL_NPZ)
    epoch = npz["X"][0][0].tolist()

    MODEL_CACHE["test"] = trained_bundle
    with client.websocket_connect("/v1/stream?model_id=test") as ws:
        ws.send_json({"epoch": epoch})  # no timestamp key
        resp = ws.receive_json()

    assert "timestamp" not in resp
    assert resp["stage"] in ("W", "N1", "N2", "N3", "REM")