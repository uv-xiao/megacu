# CUDA + NVSHMEM Examples

This directory contains examples whose first platform is CUDA and whose first
communication backend is NVSHMEM.

## Examples

- `gemm_allreduce/`: CUDA+NVSHMEM GEMM+AllReduce family. Shared descriptors,
  runtime helpers, and golden code live under the family root. Phased and
  overlap variants each separate pure CUDA/NVSHMEM baseline ownership from the
  Megacu implementation.

Matching assets:

- `docker/cuda_nvshmem/`
- `tools/cuda_nvshmem/`
