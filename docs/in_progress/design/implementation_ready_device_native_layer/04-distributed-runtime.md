# Distributed Runtime

Distributed support is part of the first design. It is not an optional
integration note. CUDA+NVSHMEM must work under single-process, MPI-launched,
and torch-distributed-launched programs without changing the Megacu target
design.

## Ownership

Megacu owns:

- typed runtime views for CUDA launch and NVSHMEM team handles;
- validation that those views match the linked target capability envelope;
- small adapters that construct `team_view` from common launch environments;
- device-side NVSHMEM primitive wrappers used by linked operators.

Megacu does not own:

- MPI process management;
- torch-distributed rendezvous;
- global job launch;
- distributed storage allocation policy outside the symmetric buffers passed to
  the target.

## Single-Process Single-Card

For one GPU:

```text
team_n_pes = 1
team_my_pe = 0
world_n_pes = 1
backend/session ids identify local single-card session
```

The same direct orchestrate target runs. Dispatcher and backend choose local
behavior from the runtime team size.

## Single-Host Multi-Card With NVSHMEM

For two GPUs on one host:

```text
nvshmrun -np 2 ./megacu_nvshmem_two_rank_smoke megacu_overlap
        |
        v
each process initializes NVSHMEM
        |
        v
adapter creates nvshmem::team_view from NVSHMEM PE identity
        |
        v
orchestrate target validates team, CUDA device, symmetric event storage,
and symmetric partial buffer
        |
        v
device-side NVSHMEM path runs
```

Multi-card communication must happen through device-side NVSHMEM in both
baseline and Megacu paths. Host-side reduction callbacks are not sufficient.

## MPI-Launched Runtime

MPI is a launcher and rank-discovery layer. The MPI adapter should produce a
CUDA device assignment and initialize or attach to NVSHMEM:

```cpp
auto session = megacu::launch::mpi_nvshmem_session::create({
    .comm = MPI_COMM_WORLD,
    .local_cuda_device = local_device_from_mpi_rank(),
    .initialize_mpi = false});

auto status = cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
    workspace,
    session.events(),
    session.cuda_launch(stream),
    session.team(),
    problem);
```

The orchestrate target sees the same `launch_view` and `team_view` as any other
caller. It does not depend on MPI headers.

## Torch-Distributed Runtime

Torch-distributed is also a launcher/rendezvous layer. A framework adapter may
construct the same Megacu views:

```python
session = megacu.torch.NvshmemSession.from_torch_distributed()
partial = session.empty_symmetric_like(c)
events = session.empty_events(cuda_nvshmem_gemm_allreduce.event_bytes(problem))
megacu.torch.gemm_allreduce(a, b, c, partial, events, session, problem)
```

The C++ binding should:

- extract CUDA stream/device from tensors;
- validate tensor dtype/layout;
- create `tensor_view` and `symmetric_tensor_view`;
- pass `session.team()` and `session.events()` to the same direct orchestrate
  function;
- return or raise based on `megacu::status`.

The binding must not reimplement dispatcher/scheduler/backend decisions.

## Capability Envelope

The first CUDA+NVSHMEM configuration should state:

```text
platform: CUDA
backend: NVSHMEM
launch modes: direct C++, nvshmrun, MPI wrapper, torch-distributed wrapper
team sizes: 1 and 2 first slice
devices: single host, one process per GPU for multi-card
communication: device-side NVSHMEM
dtypes: f32 first slice
layouts: row-major contiguous first slice
schedulers: phased and overlap targets
unsupported: arbitrary PE counts, multi-node performance claims, dynamic
             scheduler queues, host-side collectives as Megacu communication
```

The same high-level target design must cover single-card and two-card runs.
Runtime team/backend views decide which path is valid for this call.

## Distributed Verification

Required evidence before claiming distributed support:

- CUDA single-card numeric correctness;
- `nvshmrun -np 2` two-card correctness for phased baseline, overlap baseline,
  Megacu phased, and Megacu overlap;
- a negative test for mismatched team/device identity;
- a negative test for non-symmetric storage session mismatch;
- a documented torch-distributed smoke path, even if optional in CI;
- an MPI-launched smoke path or a documented blocker if MPI is unavailable.
