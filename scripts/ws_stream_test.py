"""
Realistic streaming test: replay Sleep-EDF epochs as 250-sample chunks
through the /v1/stream WebSocket endpoint.

This simulates a real EEG device client: raw EEG arrives in small
incremental buffers (e.g. 2.5s at 100 Hz = 250 samples), the server
accumulates chunks into a 30s epoch (3000 samples), then classifies.

Prerequisites:
  1. Train a model:
       python scripts/ws_stream_test.py --train-first
     OR:
       POST /v1/train {"npz_dir": "data/processed", "n_splits": 2, "model_out": "models/lda_pipeline.joblib"}

  2. Start the server:
       uvicorn sleep_bci.api.app:app --reload

  3. Run:
       python scripts/ws_stream_test.py            # chunk-based (default)
       python scripts/ws_stream_test.py --realtime  # 30s delay between epochs
       python scripts/ws_stream_test.py --chunk-size 500  # 500-sample chunks
       python scripts/ws_stream_test.py --legacy    # full-epoch format (backward compat)
"""
import argparse
import asyncio
import json
import os
import sys
import time

import numpy as np
import websockets

NPZ_PATH = "data/processed/SC4001E0.npz"
SERVER_URL = "ws://localhost:8000/v1/stream"
MODEL_ID = "lda_pipeline"
N_EPOCHS = 20
LABEL_MAP = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def train_model():
    from sleep_bci.model.train import load_nightly_npz, train_classifier
    from sleep_bci.model.artifacts import save_bundle
    os.makedirs("models", exist_ok=True)
    print("Training model from data/processed/ ...")
    X, y, ids = load_nightly_npz("data/processed")
    bundle, results = train_classifier(X, y, ids, fs=100.0, n_splits=2, model_type="lda")
    save_bundle("models/lda_pipeline.joblib", bundle)
    bal_acc = results["overall"]["balanced_accuracy_mean"]
    print(f"Model saved to models/lda_pipeline.joblib  (CV balanced_acc={bal_acc:.3f})")


async def stream_chunks(chunk_size: int, realtime: bool):
    """Send 30s epochs as incremental chunks (realistic streaming)."""
    if not os.path.exists(NPZ_PATH):
        sys.exit(f"NPZ not found: {NPZ_PATH}. Run preprocessing first.")

    data = np.load(NPZ_PATH)
    X, y = data["X"], data["y"]
    n_epochs = min(N_EPOCHS, len(X))
    epoch_samples = X.shape[-1]  # 3000
    session_id = f"ws-stream-{os.getpid()}"

    url = f"{SERVER_URL}?model_id={MODEL_ID}"
    print(f"Connecting: {url}")
    print(f"Mode: chunk-based ({chunk_size} samples/chunk, {chunk_size / 100:.1f}s per chunk)")
    print(f"Streaming {n_epochs} epochs  (realtime={realtime})")
    print()
    print(f"{'#':>3}  {'True':>4}  {'Pred':>4}  {'Conf':>6}  {'ms':>7}  {'Chunks':>6}  Match")
    print("─" * 55)

    async with websockets.connect(url) as ws:
        for epoch_i in range(n_epochs):
            raw_epoch = X[epoch_i][0]  # shape (3000,)
            true_label = LABEL_MAP[int(y[epoch_i])]
            n_chunks = int(np.ceil(epoch_samples / chunk_size))

            # Send all chunks for this epoch
            for chunk_i in range(n_chunks):
                start = chunk_i * chunk_size
                end = min(start + chunk_size, epoch_samples)
                chunk = raw_epoch[start:end].tolist()

                await ws.send(json.dumps({
                    "chunk": chunk,
                    "fs": 100.0,
                    "session_id": session_id,
                }))

                resp = json.loads(await ws.recv())

                if "error" in resp:
                    print(f"  {epoch_i:2d}  ERROR: {resp}")
                    break

                # Last chunk — should return a prediction
                if chunk_i == n_chunks - 1:
                    if "stage" not in resp:
                        print(f"  {epoch_i:2d}  WARNING: expected prediction, got: {resp}")
                        break
                    pred = resp["stage"]
                    conf = resp["confidence"]
                    lat = resp["latency_ms"]
                    mark = "✓" if pred == true_label else "✗"
                    print(
                        f"  {epoch_i:2d}  {true_label:>4}  {pred:>4}  {conf:>6.3f}  "
                        f"{lat:>5.1f}ms  {n_chunks:>5}x  {mark}"
                    )

            if realtime and epoch_i < n_epochs - 1:
                print(f"       [waiting 30s for next epoch...]")
                await asyncio.sleep(30.0)


async def stream_legacy(realtime: bool):
    """Send complete 30s epochs (backward-compatible full-epoch format)."""
    if not os.path.exists(NPZ_PATH):
        sys.exit(f"NPZ not found: {NPZ_PATH}. Run preprocessing first.")

    data = np.load(NPZ_PATH)
    X, y = data["X"], data["y"]
    n_epochs = min(N_EPOCHS, len(X))
    delay = 30.0 if realtime else 0.0

    url = f"{SERVER_URL}?model_id={MODEL_ID}"
    print(f"Connecting: {url}")
    print(f"Mode: legacy full-epoch (3000 samples at once, realtime={realtime})")
    print()
    print(f"{'#':>3}  {'True':>4}  {'Pred':>4}  {'Conf':>6}  {'ms':>7}  Match")
    print("─" * 42)

    async with websockets.connect(url) as ws:
        for i in range(n_epochs):
            epoch = X[i][0].tolist()
            true_label = LABEL_MAP[int(y[i])]

            await ws.send(json.dumps({"epoch": epoch, "timestamp": float(i * 30)}))
            resp = json.loads(await ws.recv())

            if "error" in resp:
                print(f"  {i:2d}  ERROR: {resp}")
                continue

            pred = resp["stage"]
            conf = resp["confidence"]
            lat = resp["latency_ms"]
            mark = "✓" if pred == true_label else "✗"
            print(f"  {i:2d}  {true_label:>4}  {pred:>4}  {conf:>6.3f}  {lat:>5.1f}ms  {mark}")

            if delay and i < n_epochs - 1:
                print(f"       [waiting {delay:.0f}s for next epoch...]")
                await asyncio.sleep(delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream EEG epochs through the Sleep-BCI WebSocket API.")
    parser.add_argument(
        "--chunk-size", type=int, default=250,
        help="Samples per chunk for chunk-based streaming (default: 250 = 2.5s at 100 Hz)",
    )
    parser.add_argument(
        "--realtime", action="store_true",
        help="Add 30s delay between epochs (true real-time simulation)",
    )
    parser.add_argument(
        "--legacy", action="store_true",
        help="Use legacy full-epoch format instead of chunk-based streaming",
    )
    parser.add_argument(
        "--train-first", action="store_true",
        help="Train and save model before streaming",
    )
    args = parser.parse_args()

    if args.train_first:
        train_model()

    if args.legacy:
        asyncio.run(stream_legacy(args.realtime))
    else:
        asyncio.run(stream_chunks(args.chunk_size, args.realtime))
