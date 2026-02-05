from __future__ import annotations
import argparse
import glob
import os
import numpy as np
import time

from sleep_bci.model.artifacts import load_bundle
from sleep_bci.features.bandpower import extract_features_epoch


def iter_epochs(processed_dir: str):
    night_files = sorted(
        f for f in glob.glob(os.path.join(processed_dir, "*.npz"))
        if "sleep_edf_all" not in os.path.basename(f)
    )
    if len(night_files) == 0:
        raise ValueError(f"No nightly .npz files found in {processed_dir}")

    for nf in night_files:
        data = np.load(nf, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        if X.shape[-1] == 3001:
            X = X[..., :3000]
        for i in range(X.shape[0]):
            yield nf, X[i], int(y[i])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simulate streaming inference over stored epochs.")
    p.add_argument("--processed_dir", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--realtime", action="store_true", help="Sleep briefly between epochs.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    bundle = load_bundle(args.model_path)

    n = 0
    for nf, epoch, y_true in iter_epochs(args.processed_dir):
        feats = extract_features_epoch(epoch, fs=bundle.fs).reshape(1, -1)
        proba = bundle.pipeline.predict_proba(feats)[0]
        pred = int(np.argmax(proba))
        print(
            f"{os.path.basename(nf)} | true={bundle.label_map.get(y_true,y_true)} "
            f"pred={bundle.label_map.get(pred,pred)} conf={np.max(proba):.2f}"
        )
        n += 1
        if args.realtime:
            time.sleep(0.1)
        if n >= args.max_epochs:
            break


if __name__ == "__main__":
    main()
