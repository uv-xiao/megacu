#!/usr/bin/env bash
set -euo pipefail

build_dir="${MEGACU_BUILD_DIR:-build}"

cmake --build "${build_dir}" \
  --target megacu_cuda_allgather_gemm_correctness \
  --target megacu_cuda_gemm_reduce_scatter_correctness \
  cuda_nvshmem_tiny_decode_correctness
ctest --test-dir "${build_dir}" \
  -R 'cuda_allgather_gemm_correctness|cuda_gemm_reduce_scatter_correctness|tiny_decode_correctness' \
  --output-on-failure
