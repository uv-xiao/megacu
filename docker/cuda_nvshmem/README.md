# CUDA + NVSHMEM Docker Assets

Docker assets for examples using CUDA as the platform and NVSHMEM as the
communication backend.

Shared CUDA+NVSHMEM images should live here by default. Use an example
subdirectory only when that example needs a unique image or container asset.

## Examples

- `Dockerfile`: shared CUDA 12.8 development image with Open MPI, Ninja, and
  NVSHMEM host packages for CUDA+NVSHMEM validations.

Matching paths:

- `examples/cuda_nvshmem/`
- `tools/cuda_nvshmem/`
