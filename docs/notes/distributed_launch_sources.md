# Distributed Launch Source Notes

- Date: 2026-04-24 Asia/Shanghai
- Purpose: make the CUDA+NVSHMEM multi-GPU launch and framework-integration
  contract concrete enough for implementation.
- Related design:
  `docs/design/runtime_linked_device_native_layer/04-distributed-runtime.md`

## Sources Read

### NVIDIA NVSHMEM 3.6.5 API Documentation

- Library setup, exit, and query:
  https://docs.nvidia.com/nvshmem/api/gen/api/setup.html
- Team management:
  https://docs.nvidia.com/nvshmem/api/gen/api/teams.html

Useful facts for Megacu:

- NVSHMEM initialization is collective across all processing elements that will
  participate. Each initialization call must be matched by finalization.
- `nvshmemx_init_attr` supports initialization from an existing MPI communicator
  through `NVSHMEMX_INIT_WITH_MPI_COMM`.
- `nvshmemx_init_attr` also supports unique-id initialization through
  `NVSHMEMX_INIT_WITH_UNIQUEID`. The application must distribute the UID before
  calling the initialization function.
- When MPI communicator bootstrap is used, NVSHMEM must be finalized before the
  MPI communicator is destroyed.
- If the CUDA device is not set before NVSHMEM initialization, device
  initialization can be delayed until later NVSHMEM operations. Megacu should
  set the CUDA device before constructing its NVSHMEM runtime view.
- NVSHMEM teams expose team-local PE id and PE count through team query APIs, and
  PE translation between teams is available.
- Team destruction is collective for non-predefined teams.

Megacu design implications:

- `megacu::nvshmem::team_view` must be a view over an already-initialized
  NVSHMEM team; the compiled orchestrate target must not call `nvshmem_init`,
  `nvshmemx_init_attr`, or `nvshmem_finalize`.
- Initialization and finalization belong to launch adapters or caller code, not
  the hot-path orchestrate function.
- MPI integration can use an MPI communicator bootstrap.
- Torch Distributed integration should not require MPI. It can use
  `torch.distributed` to distribute the NVSHMEM UID, then use UID bootstrap.
- The adapter must validate PE id, PE count, CUDA device, and team membership
  before calling a compiled Megacu target.

### PyTorch Distributed Documentation

- `torchrun`:
  https://docs.pytorch.org/docs/2.9/elastic/run.html
- `torch.distributed`:
  https://docs.pytorch.org/docs/2.9/distributed.html

Useful facts for Megacu:

- `torchrun` launches one or more processes per node and provides environment
  variables such as `LOCAL_RANK`, `RANK`, `LOCAL_WORLD_SIZE`, `WORLD_SIZE`,
  `MASTER_ADDR`, and `MASTER_PORT`.
- For GPU use, PyTorch's documentation expects each process to operate on one
  GPU, normally selected from `LOCAL_RANK`.
- `torch.distributed.init_process_group` initializes the distributed package and
  blocks until all processes join.
- PyTorch elastic jobs may restart workers and may change rank/world-size
  assignments between restarts.

Megacu design implications:

- The first Torch adapter should require a fixed-size `torchrun` world for the
  lifetime of a Megacu target invocation. Elastic membership changes are out of
  scope for the first CUDA+NVSHMEM backend.
- The Torch adapter should set the CUDA device from `LOCAL_RANK` before
  initializing NVSHMEM.
- The Torch adapter may use `torch.distributed` only as a control plane for rank,
  world-size, and UID exchange. Device communication for the target remains
  NVSHMEM.
- A Torch process-group rank/world mismatch with NVSHMEM PE id/count must be a
  hard error before kernel launch.
