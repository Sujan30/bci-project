from __future__ import annotations
import argparse
import glob
import os
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score

from sleep_bci.features.bandpower import extract_features_batch
from sleep_bci.model.artifacts import ModelBundle, save_bundle


DEFAULT_LABEL_MAP = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def load_nightly_npz(processed_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not os.path.exists(processed_dir):
        raise FileNotFoundError(f"processed_dir does not exist: {processed_dir}")

    night_files = sorted(
        f for f in glob.glob(os.path.join(processed_dir, "*.npz"))
        if "sleep_edf_all" not in os.path.basename(f)
    )
    if len(night_files) == 0:
        sample = os.listdir(processed_dir)[:25]
        raise ValueError(f"No nightly .npz files in {processed_dir}. Sample: {sample}")

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

    return (
        np.concatenate(X_all, axis=0),
        np.concatenate(y_all, axis=0),
        np.concatenate(night_ids_all, axis=0),
    )


def train_lda(X: np.ndarray, y: np.ndarray, night_ids: np.ndarray, fs: float, n_splits: int) -> ModelBundle:
    X_feat = extract_features_batch(X, fs=fs)

    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    gkf = GroupKFold(n_splits=n_splits)

    bal_accs, macro_f1s = [], []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_feat, y, groups=night_ids), 1):
        clf.fit(X_feat[train_idx], y[train_idx])
        pred = clf.predict(X_feat[test_idx])

        bal = balanced_accuracy_score(y[test_idx], pred)
        mf1 = f1_score(y[test_idx], pred, average="macro")
        bal_accs.append(bal)
        macro_f1s.append(mf1)

        print(f"Fold {fold}: balanced acc={bal:.3f} | macro F1={mf1:.3f}")
        print(classification_report(y[test_idx], pred, digits=3))

    print("\nOverall:")
    print(f"Balanced Acc mean±std: {np.mean(bal_accs):.4f} ± {np.std(bal_accs):.4f}")
    print(f"Macro F1 mean±std:     {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")

    clf.fit(X_feat, y)
    return ModelBundle(pipeline=clf, label_map=DEFAULT_LABEL_MAP, fs=fs)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train LDA sleep-stage classifier from nightly NPZ.")
    p.add_argument("--processed_dir", required=True)
    p.add_argument("--model_out", required=True)
    p.add_argument("--fs", type=float, default=100.0)
    p.add_argument("--n_splits", type=int, default=5)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    X, y, night_ids = load_nightly_npz(args.processed_dir)
    bundle = train_lda(X, y, night_ids, fs=args.fs, n_splits=args.n_splits)
    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)
    save_bundle(args.model_out, bundle)
    print("\n✅ Saved model:", args.model_out)


if __name__ == "__main__":
    main()
