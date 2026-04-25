# CUDA + NVSHMEM Examples

This directory contains examples whose first platform is CUDA and whose first
communication backend is NVSHMEM.

## Examples

- `gemm_allreduce_phased/`: phased GEMM+AllReduce orchestration with CUDA
  numeric kernels and NVSHMEM-backed two-rank validation.
- `gemm_allreduce_overlap/`: co-resident persistent GEMM+AllReduce overlap
  orchestration with CUDA numeric kernels and NVSHMEM-backed two-rank
  validation.
- `common/gemm_allreduce/`: shared descriptor, runtime helper, native baseline,
  and Megacu CUDA kernel support used by both examples.

Matching assets:

- `docker/cuda_nvshmem/`
- `tools/cuda_nvshmem/`
