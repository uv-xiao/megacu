# Runtime Components

Dispatcher, scheduler, platform, backend, and target runtime are linked C++
components that run inside the orchestrate call. They are not compiler passes.

## PR #4 Proof Boundary

This chapter describes the destination component split. PR #4 only needs a tiny
proof that follows the runtime-linked call shape. It may keep dispatcher or
scheduler logic problem-specific and example-local, provided those shortcuts are
not described as reusable Megacu components.

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

The general dispatcher is a `ConfigureTarget` runtime component. It maps
workload-supplied virtual-participant annotations plus the current problem and
team to executable work:

```cpp
enum class participant_role : std::uint8_t {
  compute,
  communication,
  mixed,
};

enum class placement_scope : std::uint8_t {
  per_rank,
  per_peer,
  per_tile,
  cooperative_group,
};

enum class progress_requirement : std::uint8_t {
  nonblocking,
  may_block,
  requires_co_resident_progress,
};

enum class peer_policy : std::uint8_t {
  local_only,
  local_then_remote,
  all_remote_peers,
  paired_peer,
};

enum class participant_requirement : std::uint16_t {
  none = 0,
  symmetric_storage = 1 << 0,
  co_resident_progress_if_blocking = 1 << 1,
};

struct participant_attrs {
  participant_role role;
  placement_scope placement;
  progress_requirement progress;
  peer_policy peer_policy;
  participant_requirement requires;
};

struct participant_binding {
  participant_slot slot;
  participant_attrs attrs;
};

struct dispatch_request {
  runtime_context context;
  problem_view problem;
  std::span<const participant_binding> participants;
};

struct co_residency_group {
  participant_slot producer;
  participant_slot consumer;
  std::int32_t local_lane_group;
  progress_requirement required_progress;
};

struct dispatch_state;

megacu::status map(dispatch_state &out, dispatch_request request);
```

Contract:

- compute work cursor ranges from runtime problem shape;
- map virtual participants to concrete local lanes or workers;
- map logical peer policies to backend peers from `team_view`;
- retarget the same participant annotations between single-card and multi-card
  execution based on `team_view`;
- enforce annotation constraints such as symmetric storage and co-resident
  progress;
- record mapping constraints, such as co-residency groups, that the scheduler
  must honor;
- avoid heap-heavy tables in the hot path;
- expose enough information for scheduler/operator loops.

The current ad-hoc dispatcher that infers compute/communication role by scanning
events is wrong. The replacement is explicit and annotation-driven, but still
general. GEMM+AllReduce should be one user of the annotated dispatcher, not the
dispatcher implementation itself.

For PR #4, a tiny proof can use a GEMM+AllReduce-specific mapper as an
example-local stand-in. That stand-in must be documented as proof scaffolding
and must not live in the shared dispatcher component as the final contract.

The dispatcher may expose typed cursor helpers for scheduler/operator code:

```cpp
struct tile_work {
  std::int32_t tile_row;
  std::int32_t tile_col;
  std::int32_t local_lane;
  std::int32_t logical_rank;
  std::int32_t peer_rank;
  participant_slot participant;
};

tile_work tile_work_at(
    dispatch_state dispatch,
    std::int32_t linear_tile,
    participant_slot participant);
```

The dispatcher should be cheap enough to run per orchestrate call. Tile loops
may be computed on host for launch setup or inside kernels for persistent
execution, but they must derive from runtime problem/team values.

### Single-Card And Multi-Card Retargeting

The dispatcher owns the first retargeting decision for one `OrchTarget` running
on different team sizes:

```text
participant annotations + problem + team_n_pes == 1
  -> local tile cursors
  -> no remote peer work
  -> communication participants mapped to local reduction/no-op semantics

participant annotations + problem + team_n_pes == 2
  -> local tile cursors
  -> peer tile cursors
  -> communication participants mapped to backend PE identities
```

This is not a scheduler choice. The scheduler sees a `dispatch_state` that
already tells it which participant/lane/peer work exists for this call.

### Co-Resident Mapping Boundary

The dispatcher should not decide whether overlap is globally safe. It only maps
the spatial requirement:

- which compute and communication participants must make progress together;
- which local lanes or worker groups can host them;
- which backend peers each communication participant may block on;
- whether the requested mapping needs a co-resident progress guard.

The scheduler consumes those requirements and decides whether the linked
scheduler/operator/platform envelope can execute them.

The dispatcher must not:

- require CMake syntax such as `MAP producer_lane TO RANK 0`;
- specialize the `ConfigureTarget` to one example such as GEMM+AllReduce;
- create static `dispatch_section` metadata;
- depend on string lookup for participant names.

The programming surface provides participant attributes; the dispatcher provides
the algorithm that interprets them for the linked platform/backend/scheduler
configuration.

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
- consume dispatcher co-residency groups and reject mappings that cannot be
  served by the linked scheduler/operator capability;
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

Backend does not choose single-card versus multi-card mapping. It reports
native capability and validates resources for the mapping selected by the
dispatcher.

## Platform

The CUDA platform owns:

- device ordinal validation;
- stream validation;
- launch-shape validation;
- cooperative/persistent launch capability checks if the linked scheduler
  requires them;
- CUDA error conversion to `megacu::status`.

CUDA platform code does not own rank mapping or NVSHMEM storage identity.
It may reject an overlap dispatch if the scheduler asks for a persistent or
cooperative launch shape that the current device/stream cannot support.

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

- dispatcher maps participant annotations plus a concrete problem/team to
  expected tile, lane, and peer work;
- scheduler calls the expected operator path for phased or overlap;
- backend rejects invalid teams or non-symmetric storage;
- target runtime rejects unsupported capability envelopes.
