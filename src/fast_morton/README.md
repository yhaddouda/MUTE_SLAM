fast_morton – CUDA Morton keys extension
========================================

Small CUDA / PyTorch extension to compute 3D Morton (Z-order) keys for batched points:
- morton3d_keys_cuda(points01, R) where points01 is in [0, 1]^3.

This note is generic (any project / method / platform) and includes an example for Jetson Orin.

1. Folder structure
-------------------

Place the extension inside your project, for example:

project_root/
  fast_morton/
    __init__.py          # Public Python API
    morton_ext.py        # Builds/loads the extension (torch.utils.cpp_extension.load)
    morton_cuda.cu       # CUDA implementation
    morton_cpu.cpp       # (optional) CPU fallback

__init__.py typically re-exports the main function:

from .morton_ext import morton3d_keys_cuda

2. Requirements
---------------

Before the first run:

- Python + PyTorch with CUDA support
- CUDA toolkit and nvcc available in PATH
- C++ compiler compatible with your CUDA version (e.g. g++)
- Recommended: ninja installed (faster builds)

Quick checks:

python -c "import torch; print(torch.cuda.is_available())"
which nvcc

3. Torch extensions cache directory
-----------------------------------

PyTorch JIT-compiled extensions are cached (by default) under:

- ~/.cache/torch_extensions (for the current user)

To keep the cache inside the project and share it between normal runs and sudo (profiling etc.), set:

export TORCH_EXTENSIONS_DIR="/path/to/project_root/.torch_extensions"

The directory is created automatically on first build.

You can put this either in your shell (~/.bashrc) or in your run scripts.

4. Compute capability / architectures
-------------------------------------

To ensure the extension is compiled for the correct GPU architecture, you can set:

export CUDA_ARCH_LIST="X.Y"

where X.Y is your GPU compute capability (e.g. 7.5, 8.6, 8.9, etc.).

Some extensions also accept more specific variables (example):

export TCNN_CUDA_ARCHITECTURES=87  # means compute capability 8.7

Use whatever your morton_ext.py expects; if it doesn’t read any env var, CUDA_ARCH_LIST is sufficient.

5. Example: Jetson Orin
-----------------------

Jetson AGX Orin (Ampere) GPU compute capability: 8.7.

Example run script:

#!/usr/bin/env bash
set -euo pipefail

CONDA_PREFIX="/home/you/miniforge3/envs/your_env"
PROJECT_ROOT="/home/you/work/your_project"

cd "$PROJECT_ROOT"

# Shared extension cache inside the repo
export TORCH_EXTENSIONS_DIR="$PROJECT_ROOT/.torch_extensions"

# Compile for Orin (compute capability 8.7)
export CUDA_ARCH_LIST="8.7"
# or, if your build script uses this:
export TCNN_CUDA_ARCHITECTURES=87

export PATH="$CONDA_PREFIX/bin:${PATH-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH-}"

exec "$CONDA_PREFIX/bin/python" -W ignore run.py "$@"

On the first run this will build fast_morton and cache it under .torch_extensions.

6. First build / smoke test
---------------------------

From the project root:

python -c "from fast_morton import morton3d_keys_cuda; print('fast_morton import OK')"

On the first import you should see build logs (CMake / nvcc / ninja).
On subsequent imports, it should load instantly from cache.

7. Basic usage in Python
------------------------

import torch
from fast_morton import morton3d_keys_cuda

# Points in [0, 1]^3 on CUDA
points01 = torch.rand(100000, 3, device="cuda", dtype=torch.float32)
R = 128  # grid resolution used for Morton encoding

keys = morton3d_keys_cuda(points01, R)  # int32, shape [N]
perm = torch.argsort(keys)
points_sorted = points01[perm]

You can now use perm to reorder any associated tensors before passing them to your encoders / networks.
