#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
nvcc_bin="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
if [[ ! -x "$nvcc_bin" ]]; then
    nvcc_bin="$(command -v nvcc || true)"
fi
if [[ -z "$nvcc_bin" ]]; then
    echo "CUDA compiler not found; set CUDACXX to nvcc." >&2
    exit 1
fi
cuda_arch="${GPU_KERNELS_CUDA_ARCH:-90a}"
[[ "$cuda_arch" != 90 ]] || cuda_arch=90a
extra_options=()
[[ -z "${CUTLASS_DIR:-}" ]] || extra_options+=("-DCUTLASS_DIR=$CUTLASS_DIR")
cmake -S "$repo_root" -B "$repo_root/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$cuda_arch" \
    -DGPU_KERNELS_PYTHON="${GPU_KERNELS_PYTHON:-$repo_root/.venv/bin/python}" \
    "${extra_options[@]}" \
    -DCMAKE_CUDA_COMPILER="$nvcc_bin"
targets=()
(( $# == 0 )) || targets=(--target "$@")
cmake --build "$repo_root/build" -j "${GPU_KERNELS_BUILD_JOBS:-6}" "${targets[@]}"
