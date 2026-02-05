import numpy as np
from sleep_bci.features.bandpower import extract_features_epoch


def test_extract_features_epoch():
    epoch = np.random.randn(1, 3000).astype(np.float32)
    feats = extract_features_epoch(epoch, fs=100.0)
    assert feats.shape == (7,)
    assert np.isfinite(feats).all()
