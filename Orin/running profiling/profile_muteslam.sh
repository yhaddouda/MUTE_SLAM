#!/usr/bin/env bash
set -euo pipefail
CONDA_PREFIX="/home/yh279050/miniforge3/envs/mute_slam"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
exec "$CONDA_PREFIX/bin/python" -W ignore run.py "$@"
