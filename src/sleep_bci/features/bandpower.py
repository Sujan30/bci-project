from __future__ import annotations
import numpy as np
from scipy.signal import welch


def bandpower(epoch_1d: np.ndarray, fs: float, low_hz: float, high_hz: float) -> float:
    freq, psd = welch(epoch_1d, fs=fs, nperseg=int(4 * fs))
    mask = (freq >= low_hz) & (freq <= high_hz)
    return float(np.trapezoid(psd[mask], freq[mask]))


def extract_features_epoch(epoch: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """Feature vector for one epoch.

    epoch: (n_channels, n_samples) or (n_samples,)
    returns: (7,)
    """
    sig = epoch[0] if epoch.ndim == 2 else epoch

    delta = bandpower(sig, fs, 0.5, 4)
    theta = bandpower(sig, fs, 4, 8)
    alpha = bandpower(sig, fs, 8, 12)
    sigma = bandpower(sig, fs, 12, 15)
    beta  = bandpower(sig, fs, 15, 30)

    total = delta + theta + alpha + sigma + beta + 1e-12
    return np.array(
        [
            delta, theta, alpha, sigma, beta,
            delta / total,
            theta / (alpha + beta + 1e-6),
        ],
        dtype=np.float32,
    )


def extract_features_batch(X: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """Batch extraction for training. X: (n_epochs, n_channels, n_samples)"""
    feats = [extract_features_epoch(ep, fs=fs) for ep in X]
    return np.vstack(feats)
