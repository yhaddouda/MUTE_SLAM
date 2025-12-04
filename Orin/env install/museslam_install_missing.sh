#!/usr/bin/env bash
set -euo pipefail

TARGET_ENV="${1:-museslam}"
CONDA_MISSING="mute_slam_missing_conda.txt"
PIP_MISSING="mute_slam_missing_pip.txt"

if [[ ! -f "$CONDA_MISSING" || ! -f "$PIP_MISSING" ]]; then
  echo "[ERROR] Missing lists not found. Run museslam_env_diff.sh first."
  echo "Expected: $CONDA_MISSING and $PIP_MISSING"
  exit 3
fi

# Choose mamba if available for speed
if command -v mamba >/dev/null 2>&1; then
  CONDA_BIN="mamba"
else
  CONDA_BIN="conda"
fi

# Install conda packages (batch if any lines present)
if [[ -s "$CONDA_MISSING" ]]; then
  mapfile -t PKGS < <(grep -vE '^\s*$' "$CONDA_MISSING")
  echo "[INFO] Installing conda packages into '$TARGET_ENV': ${#PKGS[@]} pkgs"
  "$CONDA_BIN" install -n "$TARGET_ENV" -y "${PKGS[@]}"
else
  echo "[INFO] No conda packages to install."
fi

# Install pip packages inside the env
if [[ -s "$PIP_MISSING" ]]; then
  echo "[INFO] Installing pip packages into '$TARGET_ENV'..."
  # Use a here-doc to avoid temp files
  conda run -n "$TARGET_ENV" python - <<'PY'
import sys, subprocess, pathlib
req = pathlib.Path("/mnt/data/museslam_missing_pip.txt").read_text().strip().splitlines()
req = [r.strip() for r in req if r.strip()]
if req:
    print("[pip] Installing:", req, flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", *req])
else:
    print("[pip] Nothing to install.")
PY
else
  echo "[INFO] No pip packages to install."
fi

echo "[DONE] Missing packages installed for env '$TARGET_ENV'."
