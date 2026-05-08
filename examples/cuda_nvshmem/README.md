# CUDA + NVSHMEM Examples

This directory contains examples whose first platform is CUDA and whose first
communication backend is NVSHMEM.

## Examples

- `gemm_allreduce/`: CUDA+NVSHMEM GEMM+AllReduce family. Shared direct target
  signatures, runtime helpers, golden code, manual baseline, and Megacu
  operator-task runtime path live under the family root.
- `gemm_reduce_scatter/`: GEMM producer tile tasks feeding EventTensor sync
  tasks and reduce-scatter consumer tile tasks.
- `allgather_gemm/`: all-gather producer tile tasks feeding EventTensor sync
  tasks and GEMM consumer tile tasks.
- `tiny_decode_pipeline/`: small decode-style pipeline composed from multiple
  tile/range operator stages.

Matching assets:

- `docker/cuda_nvshmem/`
- `tools/cuda_nvshmem/`
