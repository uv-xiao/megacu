#!/usr/bin/env bash
set -euo pipefail

image="${MEGACU_NVSHMEM_IMAGE:-megacu-nvshmem:cuda12.8}"
gpus="${MEGACU_DOCKER_GPUS:-\"device=5,6\"}"
build_dir="${MEGACU_NVSHMEM_BUILD_DIR:-build-nvshmem}"

docker build \
  -f docker/cuda_nvshmem/Dockerfile \
  -t "${image}" \
  .

docker run --rm \
  --gpus "${gpus}" \
  --ipc=host \
  -v "${PWD}:/workspace/megacu" \
  -w /workspace/megacu \
  "${image}" \
  bash -lc "rm -rf '${build_dir}' && \
    cmake -S . -B '${build_dir}' -G Ninja \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DMEGACU_ENABLE_NVSHMEM_TESTS=ON && \
    cmake --build '${build_dir}' && \
    ctest --test-dir '${build_dir}' \
      -R 'cuda_gemm_allreduce_correctness|nvshmem_two_rank_gemm_ar_correctness_(golden|megacu)' \
      --output-on-failure"
