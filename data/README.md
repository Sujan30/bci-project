# Data Directory

## Dataset: Sleep-EDF Expanded (Cassette Subset)

**Source**: PhysioNet — Sleep-EDF Expanded Database
**DOI**: [10.13026/C2SC7Q](https://doi.org/10.13026/C2SC7Q)
**Citation**: Kemp B, Zwinderman AH, Tuk B, Kamphuisen HAC, Oberye JJL (2000). *Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG.* IEEE-BME 47(9):1185-1194.

## Obtaining the Data

Raw EDF files are **not tracked in git** (too large, binary). Download them:

```bash
bash scripts/download_sample_data.sh 3  # downloads 3 nights (~300 MB)
```

PhysioNet requires a free account for access. Register at: https://physionet.org/register/

## Directory Structure

```
data/
├── raw/            # EDF input files (not tracked in git)
│   ├── SC4001E0-PSG.edf          # Night 1: Polysomnography
│   ├── SC4001EC-Hypnogram.edf    # Night 1: Sleep stage annotations
│   ├── SC4002E0-PSG.edf          # Night 2: Polysomnography
│   ├── SC4002EC-Hypnogram.edf    # Night 2: Sleep stage annotations
│   ├── SC4011E0-PSG.edf          # Night 3: Polysomnography
│   └── SC4011EH-Hypnogram.edf    # Night 3: Sleep stage annotations
├── processed/      # NPZ feature artifacts (generated, not tracked in git)
│   ├── SC4001E0.npz              # Night 1: extracted epochs
│   ├── SC4002E0.npz              # Night 2: extracted epochs
│   ├── SC4011E0.npz              # Night 3: extracted epochs
│   └── checksums.sha256          # Integrity verification
└── README.md       # This file
```

## File Manifest

| File | PSG Pair | N Epochs | Shape (X) | Class Distribution |
|------|----------|----------|-----------|--------------------|
| SC4001E0.npz | SC4001E0-PSG.edf + SC4001EC-Hypnogram.edf | 2650 | (2650, 1, 3000) | W:1997 N1:58 N2:250 N3:220 REM:125 |
| SC4002E0.npz | SC4002E0-PSG.edf + SC4002EC-Hypnogram.edf | 2829 | (2829, 1, 3000) | W:1885 N1:59 N2:373 N3:297 REM:215 |
| SC4011E0.npz | SC4011E0-PSG.edf + SC4011EH-Hypnogram.edf | 2802 | (2802, 1, 3000) | W:1856 N1:109 N2:562 N3:105 REM:170 |
| **Total** | 3 nights | **8281** | — | W:5738 N1:226 N2:1185 N3:622 REM:510 |

## Processing Parameters

| Parameter | Value |
|-----------|-------|
| Channel | EEG Fpz-Cz |
| Epoch duration | 30 seconds |
| Sampling frequency | 100 Hz |
| Samples per epoch | 3000 |
| Bandpass filter | 0.3–30.0 Hz |
| Notch filter | None (disabled) |

## Orphaned Files (No PSG Match)

The following hypnogram files exist in the dataset but have no matching PSG recording
in the 3-night subset. They are excluded from processing:

- `SC4022EJ-Hypnogram.edf` — no matching SC4022E0-PSG.edf
- `SC4031EC-Hypnogram.edf` — no matching SC4031E0-PSG.edf

## Class Imbalance Note

N1 is severely underrepresented (~2.7% of epochs). This class imbalance
causes poor N1 recall in the baseline LDA model. The RandomForest model
uses `class_weight='balanced'` to mitigate this.

## Git History Note

Raw EDF files were committed to the repository in early commits before
`.gitignore` was corrected. The `.gitignore` now properly excludes them.
If you need to reduce repository size, use `git filter-repo` to purge
the binary files from history (requires confirmation; not done automatically).

## Verifying Integrity

```bash
# From project root
sha256sum -c data/processed/checksums.sha256
```
