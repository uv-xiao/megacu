# Launch Adapters And Verification

MPI and Torch adapters are required for the CUDA+NVSHMEM PR environment. They
are not optional future paths. The source-of-truth environment for full
execution is Docker because it owns CUDA, NVSHMEM, MPI, PyTorch, and launch
tool versions.

## Launch Adapters

Direct local launch uses NVSHMEM directly and maps local process facts into the
driver.

MPI launch uses `mpirun`, MPI rank/local-rank facts, and NVSHMEM MPI bootstrap.
After initialization, the target sees the same `cuda_nvshmem::driver_view` as
other launch paths.

Torch launch uses `torchrun` and `torch.distributed` for rank/local-rank facts
and UID exchange. The smoke binaries use NVSHMEM UID bootstrap, then call the
same Megacu orchestrate functions as direct and MPI paths.

Adapters own launch normalization only. They do not choose scheduler order,
dispatcher placement, EventTensor behavior, or operator task granularity.

## Verification Commands

Local build and test:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Docker CUDA+NVSHMEM gate:

```bash
tools/cuda_nvshmem/run_two_card_docker.sh
```

The Docker runner defaults to GPUs `0,1`. Override with
`MEGACU_DOCKER_GPUS` when the machine uses different device ordinals.

Focused adapter runners inside the Docker build:

```bash
MEGACU_BUILD_DIR=build-nvshmem tools/cuda_nvshmem/run_direct.sh
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
  MEGACU_BUILD_DIR=build-nvshmem tools/cuda_nvshmem/run_mpi.sh
MEGACU_BUILD_DIR=build-nvshmem MEGACU_SKIP_BUILD=1 \
  torchrun --standalone --nproc_per_node=2 tools/cuda_nvshmem/run_torch.py
```

The full gate covers build contracts, single-GPU correctness, two-rank
NVSHMEM correctness, MPI adapter smokes, Torch adapter smokes, and tiny decode.

Megacu has no planned co-resilience feature set in this design.
