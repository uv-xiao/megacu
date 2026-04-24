# CUDA + NVSHMEM Docker Assets

Docker assets for examples using CUDA as the platform and NVSHMEM as the
communication backend.

## Examples

- `gemm_allreduce/`: CUDA 12.8 development image with Open MPI, Ninja, and
  NVSHMEM host packages for the two-rank GEMM+AllReduce validation.

Matching paths:

- `examples/cuda_nvshmem/gemm_allreduce/`
- `tools/cuda_nvshmem/gemm_allreduce/`
