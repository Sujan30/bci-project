import pytest
import numpy as np
import mne
from fastapi.testclient import TestClient
from sleep_bci.api.app import app, JOBS, training_data, MODEL_CACHE


@pytest.fixture(autouse=True)
def clear_job_state():
    """Reset shared job dicts between tests to prevent state leakage."""
    JOBS.clear()
    training_data.clear()
    yield
    JOBS.clear()
    training_data.clear()


@pytest.fixture(autouse=True)
def clear_model_cache():
    """Prevent MODEL_CACHE state from leaking between tests."""
    MODEL_CACHE.clear()
    yield
    MODEL_CACHE.clear()


@pytest.fixture
def trained_bundle(sample_npz_dir):
    """Train a real LDA pipeline from the sample NPZ fixture. Returns ModelBundle."""
    from sleep_bci.model.train import load_nightly_npz, train_lda
    X, y, night_ids = load_nightly_npz(sample_npz_dir)
    bundle, _ = train_lda(X, y, night_ids, fs=100.0, n_splits=2)
    return bundle


@pytest.fixture
def client():
    """FastAPI TestClient for API tests."""
    return TestClient(app)


@pytest.fixture
def sample_npz_dir(tmp_path):
    """Create a directory with minimal nightly .npz files for training tests."""
    sfreq = 100.0
    n_samples = 3000  # 30s at 100 Hz

    for night_id in range(3):
        n_epochs = 20
        X = np.random.randn(n_epochs, 1, n_samples).astype(np.float32)
        y = np.random.randint(0, 5, n_epochs).astype(np.int32)
        np.savez(
            tmp_path / f"night{night_id:02d}.npz",
            X=X,
            y=y,
            sfreq=sfreq,
            channel="EEG Fpz-Cz",
            epoch_sec=30,
            bandpass=np.array([0.3, 30.0]),
            notch_hz=-1.0,
            dataset="test",
        )

    return str(tmp_path)
