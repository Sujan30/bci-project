from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
from datetime import datetime
from typing import Optional

import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


def _build_classifier(model_type: str):
    """Construct an sklearn pipeline for the given model type."""
    if model_type == "random_forest":
        return make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        )
    # Default: Linear Discriminant Analysis
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())


def _save_confusion_matrix(
    y_true: list, y_pred: list, out_dir: str, label_names: list
) -> None:
    """Save confusion matrix as confusion_matrix.json and confusion_matrix.png."""
    labels = list(range(len(label_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid divide-by-zero
    cm_norm = (cm / row_sums).round(4)

    cm_dict = {
        "labels": label_names,
        "matrix": cm.tolist(),
        "normalized": cm_norm.tolist(),
    }
    with open(os.path.join(out_dir, "confusion_matrix.json"), "w") as f:
        json.dump(cm_dict, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Fraction of true class")

        tick_marks = np.arange(len(label_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(label_names)
        ax.set_yticklabels(label_names)

        for i in range(len(label_names)):
            for j in range(len(label_names)):
                ax.text(
                    j, i,
                    f"{cm[i, j]}\n({cm_norm[i, j]:.0%})",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=8,
                )

        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        ax.set_title("Confusion Matrix (Cross-Validation, all folds)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Warning: could not save confusion_matrix.png: {e}")


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    night_ids: np.ndarray,
    fs: float,
    n_splits: int,
    model_type: str = "lda",
    out_dir: Optional[str] = None,
    dataset_hash: Optional[str] = None,
) -> tuple[ModelBundle, dict]:
    """Train a sleep-stage classifier with GroupKFold cross-validation.

    Args:
        X: Raw EEG epochs, shape (n_epochs, 1, n_samples).
        y: Integer sleep-stage labels.
        night_ids: Per-epoch night index for GroupKFold splitting.
        fs: Sampling frequency in Hz.
        n_splits: Number of cross-validation folds.
        model_type: "lda" or "random_forest".
        out_dir: If given, save confusion_matrix.json / .png here.
        dataset_hash: Optional hash of input data for provenance tracking.
    """
    X_feat = extract_features_batch(X, fs=fs)

    clf = _build_classifier(model_type)
    gkf = GroupKFold(n_splits=n_splits)

    fold_results = []
    bal_accs, macro_f1s = [], []
    all_y_true, all_y_pred = [], []

    label_names = [DEFAULT_LABEL_MAP[k] for k in sorted(DEFAULT_LABEL_MAP)]

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(X_feat, y, groups=night_ids), 1
    ):
        clf.fit(X_feat[train_idx], y[train_idx])
        pred = clf.predict(X_feat[test_idx])

        all_y_true.extend(y[test_idx].tolist())
        all_y_pred.extend(pred.tolist())

        bal = balanced_accuracy_score(y[test_idx], pred)
        mf1 = f1_score(y[test_idx], pred, average="macro", zero_division=0)
        bal_accs.append(bal)
        macro_f1s.append(mf1)

        report = classification_report(
            y[test_idx], pred, digits=3, output_dict=True, zero_division=0
        )

        # Per-class F1 scores keyed by stage name
        per_class_f1 = {}
        for cls_key, cls_metrics in report.items():
            if isinstance(cls_metrics, dict) and "f1-score" in cls_metrics:
                try:
                    label_idx = int(cls_key)
                    stage = DEFAULT_LABEL_MAP.get(label_idx, str(label_idx))
                    per_class_f1[stage] = round(float(cls_metrics["f1-score"]), 4)
                except (ValueError, TypeError):
                    pass

        fold_results.append({
            "fold": fold,
            "balanced_accuracy": round(bal, 4),
            "macro_f1": round(mf1, 4),
            "per_class_f1": per_class_f1,
            "classification_report": report,
        })

        print(f"Fold {fold}: balanced acc={bal:.3f} | macro F1={mf1:.3f}")
        print(classification_report(y[test_idx], pred, digits=3, zero_division=0))

    print("\nOverall:")
    print(f"Balanced Acc mean±std: {np.mean(bal_accs):.4f} ± {np.std(bal_accs):.4f}")
    print(f"Macro F1 mean±std:     {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")

    # Train final model on all data
    clf.fit(X_feat, y)

    # Resolve git commit for provenance
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    training_config = {
        "model_type": model_type,
        "n_splits": n_splits,
        "fs": fs,
        "class_weight": "balanced" if model_type == "random_forest" else None,
        "n_estimators": 100 if model_type == "random_forest" else None,
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_commit,
    }

    results = {
        "model_type": model_type,
        "n_splits": n_splits,
        "fs": fs,
        "total_epochs": int(X.shape[0]),
        "total_nights": int(len(set(night_ids))),
        "training_config": training_config,
        "dataset_hash": dataset_hash,
        "folds": fold_results,
        "overall": {
            "balanced_accuracy_mean": round(float(np.mean(bal_accs)), 4),
            "balanced_accuracy_std": round(float(np.std(bal_accs)), 4),
            "macro_f1_mean": round(float(np.mean(macro_f1s)), 4),
            "macro_f1_std": round(float(np.std(macro_f1s)), 4),
        },
    }

    if out_dir:
        try:
            _save_confusion_matrix(all_y_true, all_y_pred, out_dir, label_names)
        except Exception as e:
            print(f"Warning: could not save confusion matrix: {e}")

    bundle = ModelBundle(
        pipeline=clf,
        label_map=DEFAULT_LABEL_MAP,
        fs=fs,
        model_type=model_type,
        training_config=training_config,
        dataset_hash=dataset_hash,
    )

    return bundle, results


# Keep backward-compatible alias
def train_lda(
    X: np.ndarray,
    y: np.ndarray,
    night_ids: np.ndarray,
    fs: float,
    n_splits: int,
) -> tuple[ModelBundle, dict]:
    return train_classifier(X, y, night_ids, fs, n_splits, model_type="lda")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a sleep-stage classifier from nightly NPZ files."
    )
    p.add_argument("--processed_dir", required=True)
    p.add_argument("--model_out", required=True)
    p.add_argument("--fs", type=float, default=100.0)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument(
        "--model_type",
        choices=["lda", "random_forest"],
        default="lda",
        help="Classifier algorithm (default: lda)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    X, y, night_ids = load_nightly_npz(args.processed_dir)
    out_dir = os.path.dirname(args.model_out) or "."
    os.makedirs(out_dir, exist_ok=True)
    bundle, results = train_classifier(
        X, y, night_ids,
        fs=args.fs,
        n_splits=args.n_splits,
        model_type=args.model_type,
        out_dir=out_dir,
    )
    save_bundle(args.model_out, bundle)
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved model: {args.model_out}")
    print(f"Saved results: {results_path}")
    if os.path.exists(os.path.join(out_dir, "confusion_matrix.png")):
        print(f"Saved confusion matrix: {os.path.join(out_dir, 'confusion_matrix.png')}")


if __name__ == "__main__":
    main()
