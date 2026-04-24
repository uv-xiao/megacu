# GEMM+AllReduce Tools

## Scripts

- `run_two_card_docker.sh`: builds the matching Docker image and runs the CUDA
  smoke, two-card CUDA smoke, and two-rank NVSHMEM correctness validation.

## Usage

```sh
tools/cuda_nvshmem/gemm_allreduce/run_two_card_docker.sh
```

Optional environment variables:

- `MEGACU_NVSHMEM_IMAGE`: Docker image tag to build and run.
- `MEGACU_DOCKER_GPUS`: GPU selector passed to `docker run --gpus`.
- `MEGACU_NVSHMEM_BUILD_DIR`: build directory used inside the container.
