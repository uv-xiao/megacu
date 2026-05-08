# Examples

Examples are organized by platform/backend pair:

```text
examples/<platform>_<backend>/<example>/
```

For example, CUDA plus NVSHMEM GEMM+AllReduce lives at:

```text
examples/cuda_nvshmem/gemm_allreduce/
```

Example families may contain variant directories. The CUDA+NVSHMEM examples
currently include:

```text
examples/cuda_nvshmem/gemm_allreduce/        # golden, manual baseline, Megacu
examples/cuda_nvshmem/gemm_reduce_scatter/   # golden, baseline, Megacu
examples/cuda_nvshmem/allgather_gemm/        # golden, baseline, Megacu
examples/cuda_nvshmem/tiny_decode_pipeline/  # golden, baseline, Megacu
```

Operational assets are shared at the platform/backend level by default:

```text
docker/<platform>_<backend>/
tools/<platform>_<backend>/
```

Use per-example Docker or tool subdirectories only when a specific example needs
unique container assets or unique run scripts.

Every example README must explain the example, file layout, execution path,
simple pseudocode, usage commands, runtime assumptions, and known limitations.
