import os
import numpy as np
import pytest

from sleep_bci.model.train import load_nightly_npz, train_lda
from sleep_bci.model.artifacts import ModelBundle, save_bundle, load_bundle


# ---------------------------------------------------------------------------
# load_nightly_npz
# ---------------------------------------------------------------------------

def test_load_nightly_npz_missing_dir():
    with pytest.raises(FileNotFoundError):
        load_nightly_npz("/nonexistent/processed/dir")


def test_load_nightly_npz_empty_dir(tmp_path):
    with pytest.raises(ValueError, match="No nightly .npz files"):
        load_nightly_npz(str(tmp_path))


def test_load_nightly_npz_valid(sample_npz_dir):
    X, y, night_ids = load_nightly_npz(sample_npz_dir)
    assert X.ndim == 3              # (epochs, channels, samples)
    assert y.ndim == 1
    assert night_ids.ndim == 1
    assert X.shape[0] == y.shape[0] == night_ids.shape[0]


def test_load_nightly_npz_ignores_combined(sample_npz_dir):
    """Files named 'sleep_edf_all.npz' should be ignored."""
    combined_path = os.path.join(sample_npz_dir, "sleep_edf_all.npz")
    np.savez(combined_path, X=np.zeros((5, 1, 3000)), y=np.zeros(5))

    X, y, _ = load_nightly_npz(sample_npz_dir)
    # epoch count should match only the 3 night files (3 × 20 = 60)
    assert X.shape[0] == 60


def test_load_nightly_npz_truncates_3001_samples(tmp_path):
    """NPZ files with 3001 samples should be silently truncated to 3000."""
    X = np.random.randn(5, 1, 3001).astype(np.float32)
    y = np.zeros(5, dtype=np.int32)
    np.savez(tmp_path / "night00.npz", X=X, y=y)

    X_out, _, _ = load_nightly_npz(str(tmp_path))
    assert X_out.shape[-1] == 3000


# ---------------------------------------------------------------------------
# train_lda
# ---------------------------------------------------------------------------

def test_train_lda_returns_bundle_and_results(sample_npz_dir):
    X, y, night_ids = load_nightly_npz(sample_npz_dir)
    bundle, results = train_lda(X, y, night_ids, fs=100.0, n_splits=3)

    assert isinstance(bundle, ModelBundle)
    assert bundle.fs == 100.0
    assert "overall" in results
    assert "folds" in results
    assert len(results["folds"]) == 3


def test_train_lda_overall_metrics_present(sample_npz_dir):
    X, y, night_ids = load_nightly_npz(sample_npz_dir)
    _, results = train_lda(X, y, night_ids, fs=100.0, n_splits=3)

    overall = results["overall"]
    for key in ("balanced_accuracy_mean", "balanced_accuracy_std", "macro_f1_mean", "macro_f1_std"):
        assert key in overall
        assert isinstance(overall[key], float)


def test_train_lda_bundle_can_predict(sample_npz_dir):
    from sleep_bci.features.bandpower import extract_features_epoch
    X, y, night_ids = load_nightly_npz(sample_npz_dir)
    bundle, _ = train_lda(X, y, night_ids, fs=100.0, n_splits=3)

    sample_epoch = X[0]
    feat = extract_features_epoch(sample_epoch, fs=100.0).reshape(1, -1)
    pred = bundle.pipeline.predict(feat)
    assert pred.shape == (1,)


# ---------------------------------------------------------------------------
# ModelBundle save / load round-trip
# ---------------------------------------------------------------------------

def test_save_and_load_bundle(tmp_path, sample_npz_dir):
    X, y, night_ids = load_nightly_npz(sample_npz_dir)
    bundle, _ = train_lda(X, y, night_ids, fs=100.0, n_splits=3)

    model_path = str(tmp_path / "model.joblib")
    save_bundle(model_path, bundle)

    assert os.path.exists(model_path)

    loaded = load_bundle(model_path)
    assert isinstance(loaded, ModelBundle)
    assert loaded.fs == bundle.fs
    assert loaded.label_map == bundle.label_map