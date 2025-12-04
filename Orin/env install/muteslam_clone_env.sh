#!/usr/bin/env bash
set -euo pipefail

SRC_ENV="${1:-coslam}"
DST_ENV="${2:-mute_slam}"

echo "[INFO] Cloning conda env '$SRC_ENV' -> '$DST_ENV' ..."
conda create --name "$DST_ENV" --clone "$SRC_ENV" -y
echo "[OK] Cloned '$SRC_ENV' to '$DST_ENV'."
echo "Activate with: conda activate $DST_ENV"
