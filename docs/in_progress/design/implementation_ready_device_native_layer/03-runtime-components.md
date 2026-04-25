# Runtime Components

Dispatcher, scheduler, platform, backend, and target runtime are linked C++
components that run inside the orchestrate call. They are not compiler passes.

## Runtime Context

The orchestrate function builds a small context:

```cpp
struct runtime_context {
  megacu::cuda::launch_view launch;
  megacu::nvshmem::team_view team;
  megacu::event_storage_view events;
};
```

The context is cheap to construct and references caller-owned handles. It is
not an executor object and does not own a graph.

## Dispatcher

The dispatcher maps the current problem and team to executable work:

```cpp
struct dispatch_state {
  int tiles_m;
  int tiles_n;
  int team_size;
  int local_rank;
};

struct tile_work {
  int tile_row;
  int tile_col;
  int logical_rank;
  int peer_rank;
};
```

Contract:

- compute tile counts from runtime problem shape;
- map virtual roles such as producer/consumer to local runtime lanes;
- map logical ranks to backend peers from `team_view`;
- avoid heap-heavy tables in the hot path;
- expose enough information for scheduler/operator loops.

The current ad-hoc dispatcher that infers compute/comm role by scanning events
is wrong. The GEMM+AllReduce dispatcher should be explicit and domain-specific
for the first slice, while still implementing a reusable component interface.

## Scheduler

The scheduler owns progress policy at runtime:

```cpp
megacu::status phased_gemm_ar(
    runtime_context ctx,
    dispatch_state dispatch,
    gemm_ar_workspace workspace,
    gemm_ar_problem problem,
    operators::gemm_ar_phased ops);

megacu::status overlap_gemm_ar(
    runtime_context ctx,
    dispatch_state dispatch,
    gemm_ar_workspace workspace,
    gemm_ar_problem problem,
    operators::gemm_ar_overlap ops);
```

Phased scheduler contract:

- launch or invoke compute work;
- allow communication for a tile once that tile is ready;
- avoid false whole-program dependencies when the linked operator supports
  tile readiness;
- fall back to a strict phase boundary only when the capability envelope says
  tile readiness is unavailable.

Overlap scheduler contract:

- run compute and communication with a progress guard;
- prove the linked operator uses co-resident workers or a persistent loop before
  allowing blocking communication waits;
- reject invalid launch envelopes at runtime.

## Backend

The NVSHMEM backend owns runtime team and symmetric-resource validation:

```cpp
megacu::status validate_team(team_view team, capability caps);
megacu::status validate_symmetric_storage(
    event_storage_view events,
    symmetric_tensor_view partial,
    team_view team);
```

It also owns device-side communication primitive wrappers used by operators:

```cpp
__device__ void signal_tile_ready(...);
__device__ void wait_tile_ready(...);
__device__ void reduce_tile_from_peers(...);
```

The exact device API can be private to CUDA/NVSHMEM first, but the ownership is
backend, not scheduler and not program IR.

## Platform

The CUDA platform owns:

- device ordinal validation;
- stream validation;
- launch-shape validation;
- cooperative/persistent launch capability checks if the linked scheduler
  requires them;
- CUDA error conversion to `megacu::status`.

CUDA platform code does not own rank mapping or NVSHMEM storage identity.

## Target Runtime

`src/target/` owns reusable call glue:

- common validation sequencing;
- capability envelope checks;
- status handling;
- repeated fast-path run helpers if needed.

Target runtime should move reusable code out of example-local headers. The
example should keep only target-specific problem/workspace definitions and the
direct orchestrate wrapper.

## No Static Sections

The runtime components must not return static section objects such as
`dispatch_section` or `schedule_section`. If tests need inspection, they should
inspect runtime behavior:

- dispatcher maps a concrete problem/team to expected tile and peer work;
- scheduler calls the expected operator path for phased or overlap;
- backend rejects invalid teams or non-symmetric storage;
- target runtime rejects unsupported capability envelopes.
