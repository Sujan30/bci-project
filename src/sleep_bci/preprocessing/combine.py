from __future__ import annotations
import argparse
import glob
import os
import numpy as np


def combine_nights(processed_dir: str, out_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not os.path.exists(processed_dir):
        raise FileNotFoundError(f"processed_dir does not exist: {processed_dir}")

    night_files = sorted(
        f for f in glob.glob(os.path.join(processed_dir, "*.npz"))
        if "sleep_edf_all" not in os.path.basename(f)
    )
    if len(night_files) == 0:
        sample = os.listdir(processed_dir)[:25]
        raise ValueError(f"No nightly .npz files found in {processed_dir}. Sample: {sample}")

    X_all, y_all, night_ids_all = [], [], []
    for night_idx, f in enumerate(night_files):
        data = np.load(f, allow_pickle=True)
        X = data["X"]
        y = data["y"]

        if X.shape[-1] == 3001:
            X = X[..., :3000]

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatch X/y in {f}: {X.shape[0]} vs {y.shape[0]}")

        X_all.append(X)
        y_all.append(y)
        night_ids_all.append(np.full((len(y),), night_idx, dtype=np.int32))

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    night_ids_all = np.concatenate(night_ids_all, axis=0)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, X=X_all, y=y_all, night_ids=night_ids_all)
    return X_all, y_all, night_ids_all


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Combine per-night NPZ into one dataset with night_ids.")
    p.add_argument("--processed_dir", required=True)
    p.add_argument("--out_path", required=True)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    X, y, night_ids = combine_nights(args.processed_dir, args.out_path)
    print("✅ Combined dataset")
    print("X:", X.shape, "y:", y.shape, "night_ids:", night_ids.shape)
    print("Saved:", args.out_path)


if __name__ == "__main__":
    main()
