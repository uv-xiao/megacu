# GEMM+AllReduce Docker Image

This directory contains the Dockerfile used by the CUDA+NVSHMEM
GEMM+AllReduce example validation.

The image installs:

- CUDA 12.8 development base image;
- CMake, Ninja, Git, and C++ build tools;
- Open MPI development/runtime packages;
- `libnvshmem3-dev-cuda-12`.

Use the matching run script:

```sh
tools/cuda_nvshmem/gemm_allreduce/run_two_card_docker.sh
```
