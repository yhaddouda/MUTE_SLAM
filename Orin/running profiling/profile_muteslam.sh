#!/usr/bin/env bash
set -euo pipefail

CONDA_PREFIX="/home/yh279050/miniforge3/envs/mute_slam"
PROJECT_ROOT="/home/yh279050/work/MUTE_SLAM"

# Run from project root
cd "$PROJECT_ROOT"

# Force extensions cache into the repo (shared between user and sudo)
export TORCH_EXTENSIONS_DIR="$PROJECT_ROOT/.torch_extensions"

# Make env binaries visible
export PATH="$CONDA_PREFIX/bin:${PATH-}"

# Libraries & stdc++
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

exec "$CONDA_PREFIX/bin/python" -W ignore run.py "$@"

