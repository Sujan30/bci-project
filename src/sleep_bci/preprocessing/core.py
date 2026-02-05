from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
from tqdm import tqdm


@dataclass(frozen=True)
class PreprocessSpec:
    dataset: str = "sleep-edf-expanded (sleep-cassette)"
    channel: str = "EEG Fpz-Cz"
    epoch_sec: int = 30
    bandpass_hz: Tuple[float, float] = (0.3, 30.0)
    notch_hz: Optional[float] = 60.0
    target_sfreq: Optional[float] = 100.0


# Default stage mapping for Sleep-EDF hypnograms.
STAGE_MAP: Dict[str, int] = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  # merge 3+4 into N3
    "Sleep stage R": 4,
}


def _fail_fast_dir(path: str, name: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} does not exist: {path}")


def discover_sleep_edf_pairs(raw_dir: str) -> List[Tuple[str, str]]:
    """Return list of (psg_edf, hypnogram_edf) pairs."""
    _fail_fast_dir(raw_dir, "raw_dir")

    psg_files = sorted(glob.glob(os.path.join(raw_dir, "*PSG.edf")))
    hyp_files = sorted(glob.glob(os.path.join(raw_dir, "*Hypnogram.edf")))

    if len(psg_files) == 0:
        sample = os.listdir(raw_dir)[:25]
        raise ValueError(
            f"No PSG EDF files found in {raw_dir}. Expected '*PSG.edf'. Sample: {sample}"
        )
    if len(hyp_files) == 0:
        sample = os.listdir(raw_dir)[:25]
        raise ValueError(
            f"No Hypnogram EDF files found in {raw_dir}. Expected '*Hypnogram.edf'. Sample: {sample}"
        )

    hyp_map: Dict[str, str] = {}
    for hf in hyp_files:
        key = os.path.basename(hf).split("-")[0]
        hyp_map[key] = hf

    pairs: List[Tuple[str, str]] = []
    for pf in psg_files:
        key = os.path.basename(pf).split("-")[0]
        hf = hyp_map.get(key)
        if hf:
            pairs.append((pf, hf))

    if len(pairs) == 0:
        raise ValueError(
            "Found PSG and Hypnogram files but could not match them by prefix. "
            "Check filenames."
        )
    return pairs


def load_raw_channel(edf_path: str, channel: str) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    if channel not in raw.ch_names:
        raise ValueError(f"Channel '{channel}' not in {edf_path}. Available: {raw.ch_names}")
    raw.pick([channel])
    return raw


def apply_standard_preprocessing(raw: mne.io.BaseRaw, spec: PreprocessSpec) -> mne.io.BaseRaw:
    raw.filter(spec.bandpass_hz[0], spec.bandpass_hz[1], verbose="ERROR")
    if spec.notch_hz is not None:
        raw.notch_filter(freqs=[spec.notch_hz], verbose="ERROR")
    if spec.target_sfreq is not None and abs(raw.info["sfreq"] - spec.target_sfreq) > 1e-6:
        raw.resample(spec.target_sfreq, verbose="ERROR")
    return raw


def load_hypnogram_annotations(hyp_path: str) -> mne.Annotations:
    hyp = mne.io.read_raw_edf(hyp_path, preload=False, verbose="ERROR")
    if hyp.annotations is None:
        raise ValueError(f"No annotations found in hypnogram file: {hyp_path}")
    return hyp.annotations


def epoch_and_label(
    raw: mne.io.BaseRaw, annotations: mne.Annotations, spec: PreprocessSpec
) -> Tuple[np.ndarray, np.ndarray]:
    """Epoch raw into fixed windows and label each epoch by hypnogram stage."""
    raw.set_annotations(annotations, emit_warning=False)

    epochs = mne.make_fixed_length_epochs(
        raw, duration=float(spec.epoch_sec), preload=True, verbose="ERROR"
    )
    X = epochs.get_data()  # (n_epochs, 1, n_samples)

    expected = int(spec.epoch_sec * raw.info["sfreq"])
    if X.shape[-1] > expected:
        X = X[..., :expected]
    elif X.shape[-1] < expected:
        pad = expected - X.shape[-1]
        X = np.pad(X, ((0, 0), (0, 0), (0, pad)), mode="constant")

    y = np.full((X.shape[0],), -1, dtype=np.int32)

    # Label by annotation covering the epoch start time
    for i in range(X.shape[0]):
        t0 = epochs.events[i, 0] / raw.info["sfreq"]
        desc = None
        for onset, dur, d in zip(annotations.onset, annotations.duration, annotations.description):
            if onset <= t0 < onset + dur:
                desc = d
                break
        if desc in STAGE_MAP:
            y[i] = STAGE_MAP[desc]

    keep = y >= 0
    return X[keep], y[keep]


def preprocess_sleep_edf(
    raw_dir: str,
    out_dir: str,
    spec: Optional[PreprocessSpec] = None,
    dry_run: bool = False,
    max_files: Optional[int] = None,
) -> Tuple[int, int]:
    """Preprocess Sleep-EDF EDF files into per-night NPZ artifacts.

    Returns: (kept_nights, skipped_nights)
    """
    spec = spec or PreprocessSpec()
    os.makedirs(out_dir, exist_ok=True)

    pairs = discover_sleep_edf_pairs(raw_dir)
    if max_files is not None:
        pairs = pairs[:max_files]

    if dry_run:
        print(f"[DRY RUN] Matched {len(pairs)} PSG/Hyp pairs. First 3:")
        for p, h in pairs[:3]:
            print(" -", os.path.basename(p), "|", os.path.basename(h))
        return 0, 0

    kept = skipped = 0
    for psg_path, hyp_path in tqdm(pairs, desc="Preprocessing nights"):
        night_id = os.path.basename(psg_path).split("-")[0]
        out_path = os.path.join(out_dir, f"{night_id}.npz")
        try:
            raw = load_raw_channel(psg_path, spec.channel)
            raw = apply_standard_preprocessing(raw, spec)
            ann = load_hypnogram_annotations(hyp_path)
            X, y = epoch_and_label(raw, ann, spec)
            if X.shape[0] == 0:
                raise ValueError("No labeled epochs after mapping; check STAGE_MAP.")
            np.savez(
                out_path,
                X=X.astype(np.float32),
                y=y.astype(np.int32),
                sfreq=float(raw.info["sfreq"]),
                channel=spec.channel,
                epoch_sec=int(spec.epoch_sec),
                bandpass=np.array(spec.bandpass_hz, dtype=np.float32),
                notch_hz=-1.0 if spec.notch_hz is None else float(spec.notch_hz),
                dataset=spec.dataset,
            )
            kept += 1
        except Exception as e:
            skipped += 1
            print(f"[SKIP] {night_id}: {e}")

    if kept == 0:
        raise RuntimeError("Preprocessing finished but produced 0 nights. Inspect logs above.")
    return kept, skipped
