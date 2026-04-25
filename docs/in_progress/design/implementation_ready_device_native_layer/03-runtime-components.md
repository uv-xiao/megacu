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
  megacu::target_capability capability;
};
```

The context is cheap to construct and references caller-owned handles. It is
not an executor object and does not own a graph.

## Dispatcher

The dispatcher maps the current problem and team to executable work:

```cpp
struct gemm_ar_dispatch {
  std::int32_t tiles_m;
  std::int32_t tiles_n;
  std::int32_t tile_count;
  std::int32_t team_size;
  std::int32_t local_rank;
};

struct gemm_ar_tile_work {
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

Minimum API shape:

```cpp
megacu::status prepare_gemm_ar(
    gemm_ar_dispatch &out,
    gemm_ar_problem problem,
    megacu::nvshmem::team_view team,
    megacu::target_capability capability);

gemm_ar_tile_work tile_work_at(
    gemm_ar_dispatch dispatch,
    std::int32_t linear_tile,
    std::int32_t peer_rank);
```

The dispatcher should be cheap enough to run per orchestrate call. Tile loops
may be computed on host for launch setup or inside kernels for persistent
execution, but they must derive from runtime problem/team values.

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

The scheduler should not perform numeric correctness checks. Correctness
checking belongs in tests and validation drivers.

The scheduler may call one native fused operator for the first implementation if
that operator internally implements the required progress model. In that case,
the scheduler still owns validation that the operator capability matches the
requested progress model.

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

Backend validation must distinguish:

- single-card local execution, where `team_n_pes == 1` and no remote peer is
  required;
- multi-card execution, where event storage and partial buffers must be
  symmetric and session-matched;
- unsupported team sizes, which return `unsupported` or `invalid_argument`
  before launch.

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

Minimum target runtime API:

```cpp
megacu::status validate_capability(
    megacu::target_capability capability,
    megacu::runtime_context context);

megacu::status validate_common_runtime(
    megacu::runtime_context context,
    gemm_ar_workspace workspace,
    gemm_ar_problem problem);
```

Target-specific workspace/problem validation may live with the example until a
second example proves it reusable.

## No Static Sections

The runtime components must not return static section objects such as
`dispatch_section` or `schedule_section`. If tests need inspection, they should
inspect runtime behavior:

- dispatcher maps a concrete problem/team to expected tile and peer work;
- scheduler calls the expected operator path for phased or overlap;
- backend rejects invalid teams or non-symmetric storage;
- target runtime rejects unsupported capability envelopes.
