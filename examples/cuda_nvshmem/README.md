# CUDA + NVSHMEM Examples

This directory contains examples whose first platform is CUDA and whose first
communication backend is NVSHMEM.

## Examples

- `gemm_allreduce/`: CUDA+NVSHMEM GEMM+AllReduce family. Shared direct target
  signatures, runtime helpers, and golden code live under the family root. PR
  #4 builds the phased Megacu implementation only.
- `gemm_reduce_scatter/`: skeleton for GEMM producer tiles feeding
  reduce-scatter consumer work.
- `allgather_gemm/`: skeleton for all-gather producer work feeding GEMM tile
  consumers.
- `tiny_decode_pipeline/`: skeleton for a small decode-style pipeline composed
  from multiple tile/range operator stages.

Matching assets:

- `docker/cuda_nvshmem/`
- `tools/cuda_nvshmem/`
