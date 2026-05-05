# CUDA + NVSHMEM Tools

Helper scripts for examples using CUDA as the platform and NVSHMEM as the
communication backend.

Shared CUDA+NVSHMEM scripts should live here by default. Use an example
subdirectory only when that example needs a unique script.

## Examples

- `run_direct.sh`: builds and runs the direct tiny-decode correctness check.
- `run_mpi.sh`: builds `megacu_adapter_contracts` and runs it with `mpirun`.
- `run_torch.py`: builds `megacu_adapter_contracts` and runs the Torch launch
  adapter smoke path from `RANK`, `WORLD_SIZE`, and `LOCAL_RANK` environment
  facts.
- `run_two_card_docker.sh`: shared two-card Docker runner for CUDA+NVSHMEM
  validation targets.

The Docker image for these helpers treats MPI, NVSHMEM, and PyTorch as required
PR dependencies. The MPI and Torch helpers assume those dependencies are
available in the execution image.

Matching paths:

- `examples/cuda_nvshmem/`
- `docker/cuda_nvshmem/`
