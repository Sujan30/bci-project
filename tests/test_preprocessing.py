import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from sleep_bci.preprocessing.core import (
    PreprocessSpec,
    STAGE_MAP,
    discover_sleep_edf_pairs,
    apply_standard_preprocessing,
    discover_and_validate,
)


# ---------------------------------------------------------------------------
# PreprocessSpec defaults
# ---------------------------------------------------------------------------

def test_preprocess_spec_defaults():
    spec = PreprocessSpec()
    assert spec.channel == "EEG Fpz-Cz"
    assert spec.epoch_sec == 30
    assert spec.bandpass_hz == (0.3, 30.0)
    assert spec.notch_hz is None
    assert spec.target_sfreq == 100.0


def test_preprocess_spec_custom():
    spec = PreprocessSpec(channel="EEG Pz-Oz", epoch_sec=20, bandpass_hz=(1.0, 40.0))
    assert spec.channel == "EEG Pz-Oz"
    assert spec.epoch_sec == 20
    assert spec.bandpass_hz == (1.0, 40.0)


# ---------------------------------------------------------------------------
# STAGE_MAP
# ---------------------------------------------------------------------------

def test_stage_map_covers_all_stages():
    expected_keys = {
        "Sleep stage W",
        "Sleep stage 1",
        "Sleep stage 2",
        "Sleep stage 3",
        "Sleep stage 4",
        "Sleep stage R",
    }
    assert set(STAGE_MAP.keys()) == expected_keys


def test_stage_map_n3_merged():
    """Stages 3 and 4 should both map to label 3 (N3)."""
    assert STAGE_MAP["Sleep stage 3"] == STAGE_MAP["Sleep stage 4"] == 3


# ---------------------------------------------------------------------------
# discover_sleep_edf_pairs — filesystem errors
# ---------------------------------------------------------------------------

def test_discover_nonexistent_dir():
    with pytest.raises(FileNotFoundError):
        discover_sleep_edf_pairs("/nonexistent/path/xyz")


def test_discover_no_psg_files(tmp_path):
    with pytest.raises(ValueError, match="No PSG EDF files"):
        discover_sleep_edf_pairs(str(tmp_path))


def test_discover_no_hypnogram_files(tmp_path):
    (tmp_path / "SC4001E0-PSG.edf").touch()
    with pytest.raises(ValueError, match="No Hypnogram EDF files"):
        discover_sleep_edf_pairs(str(tmp_path))


def test_discover_unmatched_pairs(tmp_path):
    """PSG and Hypnogram files with completely different prefixes should raise."""
    (tmp_path / "AAAA01-PSG.edf").touch()
    (tmp_path / "BBBB99-Hypnogram.edf").touch()
    with pytest.raises(ValueError, match="could not match"):
        discover_sleep_edf_pairs(str(tmp_path))


def test_discover_matched_pair(tmp_path):
    """Matching PSG/Hypnogram files should return one pair."""
    (tmp_path / "SC4001E0-PSG.edf").touch()
    (tmp_path / "SC4001E0-Hypnogram.edf").touch()
    pairs = discover_sleep_edf_pairs(str(tmp_path))
    assert len(pairs) == 1
    psg, hyp = pairs[0]
    assert psg.endswith("PSG.edf")
    assert hyp.endswith("Hypnogram.edf")


# ---------------------------------------------------------------------------
# apply_standard_preprocessing
# ---------------------------------------------------------------------------

def test_apply_standard_preprocessing_bandpass():
    """Preprocessing should call filter on the raw object."""
    mock_raw = MagicMock()
    mock_raw.info = {"sfreq": 100.0}
    spec = PreprocessSpec()

    apply_standard_preprocessing(mock_raw, spec)

    mock_raw.filter.assert_called_once_with(0.3, 30.0, verbose="ERROR")
    mock_raw.notch_filter.assert_not_called()
    mock_raw.resample.assert_not_called()


def test_apply_standard_preprocessing_notch():
    mock_raw = MagicMock()
    mock_raw.info = {"sfreq": 100.0}
    spec = PreprocessSpec(notch_hz=50.0)

    apply_standard_preprocessing(mock_raw, spec)

    mock_raw.notch_filter.assert_called_once_with(freqs=[50.0], verbose="ERROR")


def test_apply_standard_preprocessing_resample():
    """When sfreq differs from target, resample should be called."""
    mock_raw = MagicMock()
    mock_raw.info = {"sfreq": 256.0}
    spec = PreprocessSpec(target_sfreq=100.0)

    apply_standard_preprocessing(mock_raw, spec)

    mock_raw.resample.assert_called_once_with(100.0, verbose="ERROR")


def test_apply_standard_preprocessing_no_resample_when_matching():
    """When sfreq already matches target, resample should NOT be called."""
    mock_raw = MagicMock()
    mock_raw.info = {"sfreq": 100.0}
    spec = PreprocessSpec(target_sfreq=100.0)

    apply_standard_preprocessing(mock_raw, spec)

    mock_raw.resample.assert_not_called()


# ---------------------------------------------------------------------------
# discover_and_validate
# ---------------------------------------------------------------------------

def test_discover_and_validate_max_files(tmp_path):
    """max_files should limit the number of returned pairs."""
    for i in range(3):
        (tmp_path / f"SC400{i}E0-PSG.edf").touch()
        (tmp_path / f"SC400{i}E0-Hypnogram.edf").touch()

    pairs = discover_and_validate(str(tmp_path), max_files=2)
    assert len(pairs) == 2