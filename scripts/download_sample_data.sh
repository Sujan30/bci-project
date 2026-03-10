#!/usr/bin/env bash
# download_sample_data.sh — Download Sleep-EDF Expanded Cassette subset from PhysioNet
#
# Usage:
#   bash scripts/download_sample_data.sh [N_NIGHTS]
#
# N_NIGHTS: Number of nights to download (default: 3, max: 5)
# Downloads PSG + Hypnogram EDF file pairs into data/raw/
#
# Source: https://physionet.org/content/sleep-edfx/1.0.0/
# License: PhysioNet Credentialed Health Data License 1.5.0 (requires free account)

set -euo pipefail

N_NIGHTS="${1:-3}"
MAX_NIGHTS=5
BASE_URL="https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"
OUT_DIR="data/raw"

# Subject file pairs (PSG + Hypnogram), in order
PSG_FILES=(
  "SC4001E0-PSG.edf"
  "SC4002E0-PSG.edf"
  "SC4011E0-PSG.edf"
  "SC4012E0-PSG.edf"
  "SC4021E0-PSG.edf"
)
HYP_FILES=(
  "SC4001EC-Hypnogram.edf"
  "SC4002EC-Hypnogram.edf"
  "SC4011EH-Hypnogram.edf"
  "SC4012EH-Hypnogram.edf"
  "SC4021EH-Hypnogram.edf"
)

# Validate arguments
if ! [[ "$N_NIGHTS" =~ ^[0-9]+$ ]] || [ "$N_NIGHTS" -lt 1 ] || [ "$N_NIGHTS" -gt "$MAX_NIGHTS" ]; then
  echo "Error: N_NIGHTS must be between 1 and $MAX_NIGHTS (got: $N_NIGHTS)" >&2
  echo "Usage: bash scripts/download_sample_data.sh [N_NIGHTS]" >&2
  exit 1
fi

# Check wget is available
if ! command -v wget &>/dev/null; then
  echo "Error: wget is required but not installed." >&2
  echo "  macOS:  brew install wget" >&2
  echo "  Ubuntu: apt-get install wget" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
echo "Downloading $N_NIGHTS night(s) from Sleep-EDF Expanded (PhysioNet) into $OUT_DIR/"
echo ""

total_files=$(( N_NIGHTS * 2 ))
count=0

for (( i=0; i<N_NIGHTS; i++ )); do
  psg="${PSG_FILES[$i]}"
  hyp="${HYP_FILES[$i]}"
  night=$(( i + 1 ))

  count=$(( count + 1 ))
  echo "[$count/$total_files] Downloading $psg ..."
  wget --continue --no-clobber --quiet --show-progress \
    "$BASE_URL/$psg" -P "$OUT_DIR" || {
    echo "Warning: Failed to download $psg." >&2
    echo "  PhysioNet may require a free account. Visit: https://physionet.org/content/sleep-edfx/1.0.0/" >&2
  }

  count=$(( count + 1 ))
  echo "[$count/$total_files] Downloading $hyp ..."
  wget --continue --no-clobber --quiet --show-progress \
    "$BASE_URL/$hyp" -P "$OUT_DIR" || {
    echo "Warning: Failed to download $hyp." >&2
    echo "  PhysioNet may require a free account. Visit: https://physionet.org/content/sleep-edfx/1.0.0/" >&2
  }
done

echo ""
echo "Downloads complete."
n_edf=$(ls "$OUT_DIR"/*.edf 2>/dev/null | wc -l | tr -d ' ')
dir_size=$(du -sh "$OUT_DIR" 2>/dev/null | cut -f1 || echo "unknown")
echo "Summary: $n_edf .edf files in $OUT_DIR/, total size: $dir_size"
echo ""
echo "Next step:"
echo "  curl -X POST http://localhost:8000/v1/preprocess \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"dataset\": {\"raw_dir\": \"$(pwd)/$OUT_DIR\"}, \"output\": {\"combine\": true}, \"dry_run\": false}'"
