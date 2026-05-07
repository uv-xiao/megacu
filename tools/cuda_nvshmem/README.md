# CUDA + NVSHMEM Tools

Helper scripts for examples using CUDA as the platform and NVSHMEM as the
communication backend.

Shared CUDA+NVSHMEM scripts should live here by default. Use an example
subdirectory only when that example needs a unique script.

## Examples

- `run_direct.sh`: builds and runs the direct tiny-decode correctness check.
- `run_mpi.sh`: builds `megacu_adapter_contracts`, runs it with `mpirun`,
  then runs the MPI-bootstrapped GEMM-RS Megacu correctness smoke when
  NVSHMEM tests are enabled in the build.
- `run_torch.py`: builds `megacu_adapter_contracts`, checks the Torch launch
  adapter facts from `RANK`, `WORLD_SIZE`, and `LOCAL_RANK`, broadcasts an
  NVSHMEM UID through `torch.distributed`, then runs the UID-bootstrapped
  GEMM-RS Megacu correctness smoke.
- `run_two_card_docker.sh`: shared two-card Docker runner for CUDA+NVSHMEM
  validation targets.

The Docker image for these helpers treats MPI, NVSHMEM, and PyTorch as required
PR dependencies. The MPI and Torch helpers assume those dependencies are
available in the execution image.

Matching paths:

- `examples/cuda_nvshmem/`
- `docker/cuda_nvshmem/`
