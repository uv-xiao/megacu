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
    MEGACU_BUILD_DIR='${build_dir}' tools/cuda_nvshmem/run_direct.sh && \
    OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
      MEGACU_BUILD_DIR='${build_dir}' MEGACU_MPI_NP=2 \
      tools/cuda_nvshmem/run_mpi.sh && \
    MEGACU_BUILD_DIR='${build_dir}' MEGACU_SKIP_BUILD=1 \
      torchrun --standalone --nproc_per_node=2 \
      tools/cuda_nvshmem/run_torch.py && \
    ctest --test-dir '${build_dir}' \
      -R 'cuda_gemm_allreduce_correctness|cuda_gemm_reduce_scatter_correctness|cuda_allgather_gemm_correctness|tiny_decode_correctness|nvshmem_two_rank_gemm_ar_correctness_(golden|manual_baseline|megacu)|nvshmem_two_rank_gemm_rs_correctness_megacu|nvshmem_two_rank_ag_gemm_correctness_megacu|nvshmem_two_rank_tiny_decode_correctness_megacu' \
      --output-on-failure"
