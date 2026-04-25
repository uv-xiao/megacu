# CUDA + NVSHMEM Tools

Helper scripts for examples using CUDA as the platform and NVSHMEM as the
communication backend.

Shared CUDA+NVSHMEM scripts should live here by default. Use an example
subdirectory only when that example needs a unique script.

## Examples

- Target layout: a shared two-card Docker runner for CUDA+NVSHMEM validation
  targets should live directly under `tools/cuda_nvshmem/`.

Matching paths:

- `examples/cuda_nvshmem/`
- `docker/cuda_nvshmem/`
