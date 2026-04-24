# CUDA + NVSHMEM Examples

This directory contains examples whose first platform is CUDA and whose first
communication backend is NVSHMEM.

## Examples

- `gemm_allreduce/`: phased and overlap GEMM+AllReduce orchestration with CUDA
  numeric kernels and NVSHMEM-backed two-rank validation.

Matching assets:

- `docker/cuda_nvshmem/gemm_allreduce/`
- `tools/cuda_nvshmem/gemm_allreduce/`
