# Distributed Launch And Framework Integration

This chapter fills the missing CUDA+NVSHMEM multi-GPU running contract.

The compiled orchestrate target remains a direct C++ call. The missing layer is
the host control plane that creates the typed runtime views passed to that call.

## Ownership Split

Megacu has three layers for distributed execution:

| Layer | Owns | Must not own |
| --- | --- | --- |
| Compiled orchestrate target | validation of typed runtime views, slot filling, internal `run(...)`, CUDA/NVSHMEM launch | process launch, rank discovery, NVSHMEM initialization/finalization |
| Launch adapter | process-model integration, CUDA device selection, NVSHMEM bootstrap, team view construction, symmetric allocation helpers | dispatcher/scheduler/backend strategy selection |
| User or framework | choosing `torchrun`, `mpirun`, service runner, tensor/workspace allocation policy | Megacu internal metadata or slot ids |

This keeps the hot path thin while making multi-GPU startup implementable.

## Concrete Runtime Views

The first CUDA+NVSHMEM target should use explicit platform and backend views:

```cpp
namespace megacu::cuda {
struct launch_view {
  cudaStream_t stream;
  int device_ordinal;
};
}

namespace megacu {
struct event_storage_view {
  void *data;
  std::int64_t bytes;
  bool symmetric;
};
}

namespace megacu::nvshmem {
enum class ownership : std::uint8_t {
  external,
  owned_by_adapter
};

struct team_view {
  nvshmem_team_t team;
  int team_my_pe;
  int team_n_pes;
  int world_pe;
  int world_n_pes;
  int cuda_device_ordinal;
  ownership lifetime;
};
}
```

The first compiled target ABI should therefore be:

```cpp
void cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem);
```

`launch` is platform state. `team` is backend state. Keeping both explicit avoids
putting CUDA stream/device policy into the NVSHMEM handle and lets framework
wrappers pass their current stream without changing backend semantics.

## Target Team-Size Envelope

The first GEMM+AllReduce proof should compile for an exact two-PE target:

```cmake
megacu_add_orchestrate_target(
  TARGET cuda_nvshmem_gemm_allreduce_overlap
  PROGRAM gemm_allreduce_overlap_program
  SOURCES gemm_allreduce_orchestrate.cc
  KERNELS gemm_allreduce_kernels.cu
  OPS
    gemm_tile_produce=gemm_tile_produce_kernel
    allreduce_tile_consume=allreduce_tile_consume_kernel
  COMPONENTS cuda_nvshmem_static
  SCHEDULER_MODE co_resident_persistent
  BACKEND_ENVELOPE
    TEAM_SIZE 2
)
```

`TEAM_SIZE 2` is not a rank mapping rule. It is a target envelope used by
backend lowering and verification. The dispatcher still owns participant
placement and backend-peer mapping.

Later targets may support `MAX_TEAM_SIZE` or runtime team-size formulas, but the
first implementation should not. A target compiled for `TEAM_SIZE 2` must reject
a runtime `team.team_n_pes != 2` before launching kernels.

## MPI/NVSHMEM Adapter

The first MPI adapter is a C++ host-control-plane helper, not part of the device
hot path.

Planned public path:

- `include/megacu/launch/mpi_nvshmem.h`
- `src/launch/mpi_nvshmem/`
- `examples/cuda_nvshmem_gemm_allreduce/mpi_main.cc`

Initial API shape:

```cpp
namespace megacu::launch {
struct mpi_nvshmem_options {
  MPI_Comm comm;
  int local_cuda_device;
  bool initialize_mpi;
};

class mpi_nvshmem_session {
 public:
  static mpi_nvshmem_session create(mpi_nvshmem_options options);

  mpi_nvshmem_session(mpi_nvshmem_session const &) = delete;
  mpi_nvshmem_session &operator=(mpi_nvshmem_session const &) = delete;

  ~mpi_nvshmem_session();

  megacu::cuda::launch_view launch(cudaStream_t stream) const;
  megacu::nvshmem::team_view team() const;

  void *symmetric_alloc(std::size_t bytes, std::size_t alignment);
  void symmetric_free(void *ptr);
};
}
```

Creation responsibilities:

1. optionally call `MPI_Init` if `initialize_mpi` is true;
2. query MPI rank and world size from `options.comm`;
3. set the CUDA device to `local_cuda_device`;
4. call `nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, ...)`;
5. query NVSHMEM world PE id and PE count;
6. construct `team_view` from `NVSHMEM_TEAM_WORLD`;
7. validate that MPI rank/world size match NVSHMEM PE id/count.

Destruction responsibilities:

1. synchronize or require the caller to synchronize all streams that used the
   session;
2. release adapter-owned symmetric allocations;
3. call `nvshmem_finalize`;
4. call `MPI_Finalize` only if this session called `MPI_Init`.

The adapter must document that NVSHMEM finalization happens before destroying the
MPI communicator. The compiled orchestrate target never finalizes NVSHMEM.

First MPI launcher command:

```bash
mpirun -np 2 ./cuda_nvshmem_gemm_allreduce_mpi --m 128 --n 128 --k 128
```

## Torch Distributed Adapter

Torch integration should not require MPI. It should use `torchrun` as the
process launcher and `torch.distributed` as a control plane for rank/world-size
and UID exchange. CUDA/NVSHMEM remains the device communication backend.

Planned paths:

- `python/megacu/torch/nvshmem.py`
- `python/megacu/torch/gemm_allreduce.py`
- `src/integrations/torch/`
- `tests/integration/torch/`

Initial Python-facing shape:

```python
session = megacu.torch.NvshmemSession.from_torch_distributed()

partial = session.empty_symmetric_like(c)
events = session.empty_event_storage(cuda_nvshmem_gemm_allreduce.event_bytes())

megacu.torch.gemm_allreduce(
    a,
    b,
    c,
    partial=partial,
    events=events,
    session=session,
    problem=problem,
)
```

Adapter creation responsibilities:

1. require a fixed-size `torchrun` world for the session lifetime;
2. call `torch.distributed.init_process_group` if the caller has not already done
   so;
3. read `LOCAL_RANK`, `RANK`, and `WORLD_SIZE`;
4. set the CUDA device from `LOCAL_RANK`;
5. have rank 0 create an NVSHMEM UID;
6. broadcast that UID through `torch.distributed`;
7. call `nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, ...)` with the current
   rank and world size;
8. validate NVSHMEM PE id/count against Torch rank/world size;
9. expose `team_view` and `launch_view` to the C++ extension.

The adapter may use NCCL, Gloo, or another Torch process-group backend only for
control-plane operations. It must not implement the target's device
communication with Torch collectives.

Torch wrapper responsibilities before calling the compiled target:

- check all tensors are CUDA tensors on the current device;
- check `a`, `b`, and `c` meet the target's dtype and layout envelope;
- check `partial` and `events` are symmetric NVSHMEM allocations from the same
  `NvshmemSession`;
- use the current PyTorch CUDA stream to fill `megacu::cuda::launch_view`;
- call the compiled target directly through the extension binding.

First Torch launcher command:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  tests/integration/torch/test_cuda_nvshmem_gemm_allreduce.py
```

Elastic Torch membership changes are out of scope for the first CUDA/NVSHMEM
backend. If `WORLD_SIZE` changes during a session, the adapter must destroy the
old session and construct a new one before any Megacu target is called again.

## Symmetric Allocation Contract

For the first GEMM+AllReduce target:

- `workspace.a`, `workspace.b`, and `workspace.c` may be ordinary local CUDA
  tensor views if the selected kernels only access the local PE's storage.
- `workspace.partial` must be symmetric or otherwise peer-addressable by the
  selected NVSHMEM reduction primitive.
- `events.data` must be symmetric NVSHMEM-accessible storage because remote PEs
  signal and wait on per-tile readiness.
- the adapter that allocates symmetric memory must also free it after all kernels
  using it are complete.

The backend plan must name which resource slots require symmetric allocation.
The runtime target must reject a nonsymmetric view when the selected backend plan
requires symmetry.

## Runtime Validation

Before `run_static_persistent(...)`, the compiled target must validate:

- `launch.device_ordinal == team.cuda_device_ordinal`;
- `team.team_n_pes` matches the target `TEAM_SIZE` envelope;
- `team.team_my_pe` is in `[0, team.team_n_pes)`;
- `events.bytes` is at least the backend-plan event-storage requirement;
- `events.symmetric == true` for remote events;
- every workspace field required to be symmetric is marked symmetric or is
  represented by an adapter-owned symmetric view;
- all CUDA tensor pointers belong to `launch.device_ordinal`;
- the target is called after NVSHMEM initialization and before finalization.

Validation failures should return or throw host-side errors before kernel
launch. Device assertions are not an acceptable primary validation mechanism for
these conditions.

## Failure Modes

- `torchrun` rank/world-size differs from NVSHMEM PE id/count.
- `mpirun` rank/world-size differs from NVSHMEM PE id/count.
- CUDA device was not set before NVSHMEM device initialization.
- A target compiled for `TEAM_SIZE 2` is called with one or more than two PEs.
- Torch wrapper passes ordinary PyTorch tensors for symmetric event or partial
  storage.
- The PyTorch current stream is ignored and kernels launch on the wrong stream.
- A session finalizes NVSHMEM while a Megacu target still has outstanding work.
- The adapter treats Torch collectives as a substitute for NVSHMEM device
  primitives.

Each of these must have either a unit test, integration test, or documented skip
reason before the first CUDA+NVSHMEM implementation is considered complete.
