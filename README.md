# sleep-bci (EEG → features → classifier → API)

Industry-style refactor of a Sleep-EDF EEG sleep-stage classifier into a reusable pipeline:

**EEG Stream → Preprocessing → Feature Extraction → Classifier API → Output State**

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Data (not included)

Place Sleep-EDF (sleep-cassette) EDF files under:

```
data/raw/sleep-cassette/
  SC4xxxE0-PSG.edf
  SC4xxxEC-Hypnogram.edf
```

## Preprocess

```bash
sleepbci-preprocess --raw_dir data/raw/sleep-cassette --out_dir data/processed/nightly
```

## Train

```bash
sleepbci-train --processed_dir data/processed/nightly --model_out models/lda_pipeline.joblib
```

## Serve API

```bash
sleepbci-serve --model_path models/lda_pipeline.joblib
# open http://127.0.0.1:8000/docs
```

## Simulate streaming

```bash
sleepbci-simulate --processed_dir data/processed/nightly --model_path models/lda_pipeline.joblib --max_epochs 50
```

## Label mapping

W=0, N1=1, N2=2, N3=3, REM=4
