"""
IRL test: replay real Sleep-EDF epochs through /v1/stream.

This simulates a real EEG device client: each epoch is a genuine 30-second
EEG recording that would arrive from hardware in real time. The server
classifies it and returns the predicted sleep stage.

The only difference from live hardware: we send epochs back-to-back
(--realtime flag adds the real 30-second wait between epochs).

Prerequisites:
  1. Train a model:
       python scripts/ws_stream_test.py --train-first
     OR manually:
       POST /train {"npz_dir":"data/processed","n_splits":2,"model_out":"models/lda_pipeline.joblib"}

  2. Start the server:
       uvicorn sleep_bci.api.app:app --reload

  3. pip install websockets

  4. Run:
       python scripts/ws_stream_test.py            # fast replay
       python scripts/ws_stream_test.py --realtime  # true 30s per epoch
"""
import argparse, asyncio, json, os, sys, time
import numpy as np
import websockets

NPZ_PATH   = "data/processed/SC4001E0.npz"
SERVER_URL = "ws://localhost:8000/v1/stream"
MODEL_ID   = "lda_pipeline"
N_EPOCHS   = 20
LABEL_MAP  = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def train_model():
    import os
    from sleep_bci.model.train import load_nightly_npz, train_lda
    from sleep_bci.model.artifacts import save_bundle
    os.makedirs("models", exist_ok=True)
    print("Training model from data/processed/ ...")
    X, y, ids = load_nightly_npz("data/processed")
    bundle, results = train_lda(X, y, ids, fs=100.0, n_splits=2)
    save_bundle("models/lda_pipeline.joblib", bundle)
    acc = results.get("mean_balanced_accuracy")
    acc_str = f"{acc:.3f}" if acc is not None else "?"
    print(f"Model saved to models/lda_pipeline.joblib  (CV balanced_acc={acc_str})")


async def stream(realtime: bool):
    if not os.path.exists(NPZ_PATH):
        sys.exit(f"NPZ not found: {NPZ_PATH}")

    data  = np.load(NPZ_PATH)
    X, y  = data["X"], data["y"]
    n     = min(N_EPOCHS, len(X))
    delay = 30.0 if realtime else 0.0

    url = f"{SERVER_URL}?model_id={MODEL_ID}"
    print(f"Connecting: {url}")
    print(f"Streaming {n} real EEG epochs  (realtime={realtime})\n")
    print(f"{'#':>3}  {'True':>4}  {'Pred':>4}  {'Conf':>6}  {'ms':>7}  Match")
    print("─" * 42)

    async with websockets.connect(url) as ws:
        for i in range(n):
            epoch = X[i][0].tolist()
            true  = LABEL_MAP[int(y[i])]
            t0    = time.perf_counter()

            await ws.send(json.dumps({"epoch": epoch, "timestamp": float(i * 30)}))
            resp  = json.loads(await ws.recv())

            if "error" in resp:
                print(f"  {i:2d}  ERROR: {resp}")
                continue

            pred  = resp["stage"]
            conf  = resp["confidence"]
            lat   = resp["latency_ms"]
            mark  = "✓" if pred == true else "✗"
            print(f"  {i:2d}  {true:>4}  {pred:>4}  {conf:>6.3f}  {lat:>5.1f}ms  {mark}")

            if delay and i < n - 1:
                print(f"       [waiting {delay:.0f}s for next epoch...]")
                await asyncio.sleep(delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--realtime", action="store_true",
                        help="Add 30s delay between epochs (true real-time simulation)")
    parser.add_argument("--train-first", action="store_true",
                        help="Train and save model before streaming")
    args = parser.parse_args()

    if args.train_first:
        train_model()

    asyncio.run(stream(args.realtime))
