import numpy as np
import pytest
from sleep_bci.features.bandpower import bandpower, extract_features_epoch, extract_features_batch


def test_extract_features_epoch_shape():
    epoch = np.random.randn(1, 3000).astype(np.float32)
    feats = extract_features_epoch(epoch, fs=100.0)
    assert feats.shape == (7,)


def test_extract_features_epoch_finite():
    epoch = np.random.randn(1, 3000).astype(np.float32)
    feats = extract_features_epoch(epoch, fs=100.0)
    assert np.isfinite(feats).all()


def test_extract_features_epoch_1d_input():
    """1-D signal (no channel dim) should also return (7,)."""
    epoch = np.random.randn(3000).astype(np.float32)
    feats = extract_features_epoch(epoch, fs=100.0)
    assert feats.shape == (7,)
    assert np.isfinite(feats).all()


def test_extract_features_epoch_dtype():
    epoch = np.random.randn(1, 3000).astype(np.float32)
    feats = extract_features_epoch(epoch, fs=100.0)
    assert feats.dtype == np.float32


def test_extract_features_epoch_nonnegative_powers():
    """Band-power values (first 5 features) must be >= 0."""
    epoch = np.random.randn(1, 3000).astype(np.float32)
    feats = extract_features_epoch(epoch, fs=100.0)
    assert (feats[:5] >= 0).all()


def test_bandpower_returns_float():
    sig = np.random.randn(3000)
    result = bandpower(sig, fs=100.0, low_hz=0.5, high_hz=4.0)
    assert isinstance(result, float)
    assert result >= 0


def test_bandpower_delta_dominant():
    """A signal built from low-frequency sine waves should have high delta power."""
    t = np.linspace(0, 30, 3000)
    sig = np.sin(2 * np.pi * 1.0 * t)  # 1 Hz — inside delta band
    delta = bandpower(sig, fs=100.0, low_hz=0.5, high_hz=4.0)
    beta = bandpower(sig, fs=100.0, low_hz=15.0, high_hz=30.0)
    assert delta > beta


def test_extract_features_batch_shape():
    X = np.random.randn(10, 1, 3000).astype(np.float32)
    feats = extract_features_batch(X, fs=100.0)
    assert feats.shape == (10, 7)


def test_extract_features_batch_finite():
    X = np.random.randn(5, 1, 3000).astype(np.float32)
    feats = extract_features_batch(X, fs=100.0)
    assert np.isfinite(feats).all()