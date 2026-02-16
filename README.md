# sleep-bci (EEG → features → classifier → API)

Industry-style refactor of a Sleep-EDF EEG sleep-stage classifier into a reusable pipeline:

**EEG Stream → Preprocessing → Feature Extraction → Classifier API → Output State**

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Docker Usage

### Prerequisites
- Docker and Docker Compose installed

### Quick Start (API Server)

Start the API server:
```bash
docker-compose up api
```

Visit http://localhost:8000/docs to see the API documentation.

### Running CLI Commands

Preprocess data:
```bash
docker-compose run --rm cli sleepbci-preprocess \
  --raw_dir /app/data/raw/ \
  --out_dir /app/data/results
```

Train model:

if the number of nights is less than 5, you MUST add a n_splits argument that is less than the number of nights of data you have

```bash
docker-compose run --rm cli sleepbci-train \
  --processed_dir /app/data/results \
  --model_out /app/models/lda_pipeline.joblib \
  --n_splits=2 
```

Combine datasets:
```bash
docker-compose run --rm cli sleepbci-combine \
  --processed_dir /app/data/processed/nightly \
  --out_path /app/data/processed/combined.npz
```

Simulate streaming:
```bash
docker-compose run --rm cli sleepbci-simulate \
  --processed_dir /app/data/processed/nightly \
  --model_path /app/models/lda_pipeline.joblib \
  --max_epochs 50
```

### Development Mode

Run with hot code reloading:
```bash
docker-compose --profile dev up dev
```

Code changes in `src/` will automatically reload the server.

### Using Docker Directly

Build image:
```bash
docker build -t sleep-bci:latest .
```

Run API:
```bash
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  sleep-bci:latest
```

Run CLI command:
```bash
docker run --rm \
  -v $(pwd)/data:/app/data \
  sleep-bci:latest sleepbci-preprocess --help
```

### Environment Variables

Copy `.env.example` to `.env` and customize:
- `PORT`: API server port (default: 8000)
- `HOST`: API server host (default: 0.0.0.0)
- `LOG_LEVEL`: Logging level (default: info)

Example:
```bash
cp .env.example .env
# Edit .env with your preferred values
docker-compose up api
```

## Data (not included)

Or you could get the data from here:
https://physionet.org/content/sleep-edfx/1.0.0/

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


## Quick example start

```bash
sleepbci-serve #start the server
open http://localhost:8000/docs #open the server docs
./test_api.sh
```