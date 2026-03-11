#!/usr/bin/env bash
# download_sample_data.sh — Download Sleep-EDF Expanded Cassette subset from PhysioNet
#
# Usage:
#   bash scripts/download_sample_data.sh [N_NIGHTS]
#
# N_NIGHTS: Number of nights to download (default: 3, max: 25)
# Downloads PSG + Hypnogram EDF file pairs into data/raw/
#
# Source: https://physionet.org/content/sleep-edfx/1.0.0/
# License: PhysioNet Credentialed Health Data License 1.5.0 (requires free account)
#
# Note: The hypnogram filename suffix (C, H, J) is assigned by PhysioNet per recording
# and cannot be derived from the subject number alone. All 25 pairs below are confirmed
# from the Sleep-EDF Expanded Cassette manifest. If a file fails to download, PhysioNet
# likely requires authentication — visit https://physionet.org/content/sleep-edfx/1.0.0/

set -euo pipefail

N_NIGHTS="${1:-3}"
MAX_NIGHTS=25
BASE_URL="https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"
OUT_DIR="data/raw"

# Confirmed PSG + Hypnogram pairs from Sleep-EDF Expanded Cassette (25 nights)
PSG_FILES=(
  "SC4001E0-PSG.edf"   # Subject 01, night 1
  "SC4002E0-PSG.edf"   # Subject 01, night 2
  "SC4011E0-PSG.edf"   # Subject 02, night 1
  "SC4012E0-PSG.edf"   # Subject 02, night 2
  "SC4021E0-PSG.edf"   # Subject 03, night 1
  "SC4022E0-PSG.edf"   # Subject 03, night 2
  "SC4031E0-PSG.edf"   # Subject 04, night 1
  "SC4032E0-PSG.edf"   # Subject 04, night 2
  "SC4041E0-PSG.edf"   # Subject 05, night 1
  "SC4042E0-PSG.edf"   # Subject 05, night 2
  "SC4051E0-PSG.edf"   # Subject 06, night 1
  "SC4052E0-PSG.edf"   # Subject 06, night 2
  "SC4061E0-PSG.edf"   # Subject 07, night 1
  "SC4062E0-PSG.edf"   # Subject 07, night 2
  "SC4071E0-PSG.edf"   # Subject 08, night 1
  "SC4072E0-PSG.edf"   # Subject 08, night 2
  "SC4081E0-PSG.edf"   # Subject 09, night 1
  "SC4082E0-PSG.edf"   # Subject 09, night 2
  "SC4091E0-PSG.edf"   # Subject 10, night 1
  "SC4092E0-PSG.edf"   # Subject 10, night 2
  "SC4101E0-PSG.edf"   # Subject 11, night 1
  "SC4102E0-PSG.edf"   # Subject 11, night 2
  "SC4111E0-PSG.edf"   # Subject 12, night 1
  "SC4112E0-PSG.edf"   # Subject 12, night 2
  "SC4121E0-PSG.edf"   # Subject 13, night 1
)
HYP_FILES=(
  "SC4001EC-Hypnogram.edf"
  "SC4002EC-Hypnogram.edf"
  "SC4011EH-Hypnogram.edf"
  "SC4012EH-Hypnogram.edf"
  "SC4021EH-Hypnogram.edf"
  "SC4022EJ-Hypnogram.edf"
  "SC4031EC-Hypnogram.edf"
  "SC4032EC-Hypnogram.edf"
  "SC4041EH-Hypnogram.edf"
  "SC4042EH-Hypnogram.edf"
  "SC4051EH-Hypnogram.edf"
  "SC4052EH-Hypnogram.edf"
  "SC4061EH-Hypnogram.edf"
  "SC4062EH-Hypnogram.edf"
  "SC4071EH-Hypnogram.edf"
  "SC4072EH-Hypnogram.edf"
  "SC4081EJ-Hypnogram.edf"
  "SC4082EJ-Hypnogram.edf"
  "SC4091EH-Hypnogram.edf"
  "SC4092EJ-Hypnogram.edf"
  "SC4101EH-Hypnogram.edf"
  "SC4102EH-Hypnogram.edf"
  "SC4111EH-Hypnogram.edf"
  "SC4112EH-Hypnogram.edf"
  "SC4121EH-Hypnogram.edf"
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
echo "Next step — preprocess:"
echo "  curl -X POST http://localhost:8000/v1/preprocess \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"dataset\": {\"raw_dir\": \"$(pwd)/$OUT_DIR\"}, \"output\": {\"combine\": true}, \"dry_run\": false}'"
