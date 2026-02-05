from __future__ import annotations
import argparse
from sleep_bci.preprocessing.core import preprocess_sleep_edf, PreprocessSpec


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preprocess Sleep-EDF EDF files into per-night NPZ.")
    p.add_argument("--raw_dir", required=True, help="Directory containing *PSG.edf and *Hypnogram.edf files.")
    p.add_argument("--out_dir", required=True, help="Output directory for per-night .npz files.")
    p.add_argument("--dry_run", action="store_true", help="Only discover/match files; do not process.")
    p.add_argument("--max_files", type=int, default=None, help="Limit number of nights for quick tests.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    spec = PreprocessSpec()
    kept, skipped = preprocess_sleep_edf(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        spec=spec,
        dry_run=args.dry_run,
        max_files=args.max_files,
    )
    if args.dry_run:
        return
    print("\n✅ Done preprocessing")
    print("Kept nights:", kept)
    print("Skipped nights:", skipped)


if __name__ == "__main__":
    main()
