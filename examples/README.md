# Examples

Examples are organized by platform/backend pair:

```text
examples/<platform>_<backend>/<example>/
```

For example, CUDA plus NVSHMEM GEMM+AllReduce lives at:

```text
examples/cuda_nvshmem/gemm_allreduce/
```

Each example must have matching operational assets when they exist:

```text
docker/<platform>_<backend>/<example>/
tools/<platform>_<backend>/<example>/
```

Every example README must explain the example, file layout, execution path,
simple pseudocode, usage commands, runtime assumptions, and known limitations.
