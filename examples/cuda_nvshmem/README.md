# CUDA + NVSHMEM Examples

This directory contains examples whose first platform is CUDA and whose first
communication backend is NVSHMEM.

## Examples

- `gemm_allreduce/`: CUDA+NVSHMEM GEMM+AllReduce family. Shared direct target
  signatures, runtime helpers, and golden code live under the family root. PR
  #4 builds the phased Megacu implementation only.

Matching assets:

- `docker/cuda_nvshmem/`
- `tools/cuda_nvshmem/`
