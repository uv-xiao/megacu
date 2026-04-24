# Dispatcher, Scheduler, And Kernel Lowering

This chapter separates three things that were previously too entangled.

## Dispatcher

The dispatcher has two build-graph parts:

1. a reusable compiled implementation
2. an orchestrate-target lowering step on one concrete program

Together they map logical program work to execution-level placements.

Inputs from the program model:

- named op
- logical work domain
- virtual participant placement
- resource usage
- explicit event dependencies
- optional mapping hints

Dispatcher responsibilities:

- decide how logical work is partitioned
- decide placement groups or execution lanes
- produce the execution-level work units consumed by the scheduler

The dispatcher is independent from the scheduler. This matches the user's point:
mapping strategy and scheduling strategy are not the same thing.

Implementation owner: `src/dispatcher/`.

Participant placement belongs here. CMake may select the dispatcher target, but
CMake should not expose ad hoc `compute_lane -> rank 0` mapping syntax in the
public Megacu API. A dispatcher implementation or typed dispatcher policy owns
that rule and emits a participant table during target lowering.

First dispatcher contract:

- input: lowered program facts from the public builder
- output: dispatch table mapping logical domain instances to execution
  placements
- output: participant table mapping virtual participants at each domain point
  to backend-native peers, ranks, lanes, or CTAs
- no ownership of event wait/signal semantics
- no ownership of execution order
- no public authoring API beyond optional placement hints

First dispatcher output shape, planned path
`include/megacu/detail/dispatch_plan.h`:

```cpp
namespace megacu::detail {
enum class placement_role : std::uint8_t { compute, comm };

struct domain_tile_2d {
  std::uint32_t m;
  std::uint32_t n;
};

struct dispatch_entry {
  domain_tile_2d tile;
  placement_role role;
  std::uint16_t participant_slot;
  std::uint16_t logical_rank;
  std::uint16_t worker_index;
};

struct participant_entry {
  domain_tile_2d tile;
  std::uint16_t participant_slot;
  std::uint16_t logical_rank;
  std::uint16_t backend_peer;
};

struct dispatch_plan {
  std::span<const dispatch_entry> work;
  std::span<const participant_entry> participants;
};
}
```

For GEMM+AllReduce, lowering may keep this plan compact by storing ranges
instead of one entry per tile. The observable plan must still answer two
questions without string lookup:

- which compute or communication worker owns this logical output tile;
- which backend peer corresponds to each rank participant for this tile.

For the first static CUDA/NVSHMEM GEMM+AllReduce example, the dispatcher can
implement a fixed compute/communication policy:

- `compute_lane` resolves to GEMM-producing placements for the current output
  tile point;
- `reduce_lane` resolves to communication/reduction placements for the same
  output tile point;
- rank/team participants resolve to backend-native peers for each partial tile;
- the resulting participant table is consumed by kernel lowering and backend
  event resolution.

If a future workload needs a different compute/communication relation, it
should use a different dispatcher policy or dispatcher configuration, not CMake
one-off mapping lines and not runtime scheduler selection.

## Scheduler

The scheduler also has two build-graph parts:

1. a reusable compiled engine
2. an orchestrate-target payload specialized for one concrete program

The scheduler consumes dispatcher output. It does not define the author's public
API.

Examples:

- static persistent schedule
- queue-backed schedule
- scheduler-CTA model
- cluster-synchronous model

Implementation owner: `src/scheduler/`.

First scheduler contract:

- input: dispatcher output plus explicit event dependencies
- output: schedule payload for the selected execution strategy
- no public scheduler base class
- no runtime scheduler selection
- static persistent scheduling is the first implemented strategy

First scheduler output shape, planned path
`include/megacu/detail/schedule_plan.h`:

```cpp
namespace megacu::detail {
enum class schedule_action : std::uint8_t {
  launch_gemm_tile_produce,
  launch_allreduce_tile_consume
};

enum class progress_model : std::uint8_t {
  phased,
  co_resident_persistent
};

enum class residency_role : std::uint8_t {
  compute,
  comm
};

enum class event_wait_mode : std::uint8_t {
  none,
  blocking_device_wait,
  phase_satisfied
};

struct schedule_entry {
  std::uint32_t order;
  std::uint32_t dispatch_entry_index;
  schedule_action action;
  std::uint16_t required_event_slot;
  std::uint16_t released_event_slot;
  std::uint16_t phase;
  std::uint16_t residency_group;
  residency_role role;
  event_wait_mode wait_mode;
};

struct residency_group {
  std::uint16_t group;
  std::uint16_t min_compute_workers;
  std::uint16_t min_comm_workers;
  bool all_workers_must_be_launched_together;
};

struct cuda_residency_envelope {
  std::uint16_t persistent_grid_blocks;
  std::uint16_t threads_per_block;
  std::uint16_t min_sm_count;
  std::uint16_t max_blocks_per_sm;
  bool requires_cooperative_launch;
};

struct schedule_plan {
  progress_model progress;
  std::uint16_t num_compute_workers;
  std::uint16_t num_comm_workers;
  std::span<const residency_group> residency_groups;
  cuda_residency_envelope residency;
  std::span<const schedule_entry> entries;
};
}
```

For the first static persistent scheduler, `entries` may describe deterministic
loops rather than every tile instance. It must still be inspectable: tests
should be able to see that GEMM tile production releases `partial_ready_event`
and AllReduce tile consumption acquires it.

## Communication Progress Guard

Not every communication edge requires compute/communication co-residency. Megacu
must distinguish two cases:

- **phased progress**: producers finish a phase, then consumers run a later phase;
  correctness does not require producer and consumer CTAs to be resident at the
  same time.
- **co-resident progress**: a consumer may wait while a producer is expected to
  keep making progress in the same launch; correctness requires both roles to be
  resident together.

The scheduler owns this distinction. It must materialize a progress model in
`schedule_plan`, and kernel lowering must reject a target that uses a blocking
wait without a valid progress guard.

For `progress_model::co_resident_persistent`, the schedule plan must prove:

- every blocking acquire has a producer role and consumer role in the same
  `residency_group`;
- the group has at least one compute worker and one communication worker;
- all workers in the group are launched by the same persistent entrypoint;
- `cuda_residency_envelope::persistent_grid_blocks` is no larger than
  `min_sm_count * max_blocks_per_sm`, so those workers are resident worker loops,
  not unbounded CUDA blocks waiting for blocks that may never be scheduled;
- runtime CUDA validation checks the current device has at least `min_sm_count`
  SMs and supports cooperative launch when `requires_cooperative_launch` is true;
- no blocking wait crosses to a different residency group unless that wait is
  known to be satisfied by a completed earlier phase.

This is the guard against deadlock in overlap kernels. A consumer task must not
spin on an event whose producer task is merely queued behind it in the same CUDA
grid or in a later launch. If the scheduler cannot prove co-residency, it must
choose `progress_model::phased` or fail target lowering.

For `progress_model::phased`, the scheduler may use separate launches or
persistent phases. The event dependency is still present for metadata and
validation, but the runtime does not rely on simultaneous producer/consumer
progress.

## Kernel Lowering

Kernel lowering also splits into reusable compiled logic plus orchestrate-target
specialization.

It decides how named kernels are turned into executable device behavior for the
chosen scheduler.

Examples:

- keep named kernels as separate launches inside the orchestrate-controlled
  execution path
- stitch several named kernels into one persistent kernel
- link helper implementations around backend/platform interaction

This is also where fine-grained overlap needs careful treatment.

Implementation owner: `src/lowering/`.

First kernel-lowering contract:

- input: named op table, typed resource table, event table, dispatch table, and
  selected scheduler payload
- input: participant table from dispatcher and backend event-layout rules
- output: CUDA/NVSHMEM executable path plus inspection metadata
- owns stitching and launch payload construction
- does not invent public fragment-op APIs
- does not emit new C++/CUDA source in the normal path

First kernel-lowering output shape, planned path
`include/megacu/detail/kernel_plan.h`:

```cpp
namespace megacu::detail {
struct kernel_symbol {
  std::string_view op_name;
  std::uint16_t op_slot;
  std::uint16_t entrypoint_role;
  std::uint16_t symbol_id;
};

struct cuda_launch_shape {
  dim3 grid;
  dim3 block;
  std::uint32_t dynamic_smem_bytes;
  bool cooperative;
};

struct kernel_plan {
  std::span<const kernel_symbol> symbols;
  cuda_launch_shape launch;
  bool stitched_persistent;
};
}
```

The first persistent lowering may launch one stitched persistent entrypoint
implemented by Megacu and call linked op bodies through a small op table. That
lowering requires callable bodies or lowering-provided trampolines. A different
CUDA lowering may instead launch one or more `__global__` kernels directly and
only require launchable-kernel entrypoints. The op table is built from CMake
`OPS` entries and follows the role-based op implementation ABI in
`10-implementation-architecture.md`.

It may also temporarily use separate launches while the persistent engine is
being built. In both cases, the normal path links existing C++/CUDA symbols and
metadata; it does not emit a new program-specific `.cu` file.

## Fine-Grained Compute/Communication Overlap

The user concern is correct: fine-grained overlap is important, and a model that
treats every op as a fully opaque launched kernel would be too weak.

The chosen solution is:

- keep a single public named-op abstraction
- allow backend primitives inside named kernel bodies
- let kernel lowering stitch named kernels when the chosen target strategy
  supports it

So fine-grained overlap comes from two places:

1. in-kernel backend/platform primitives used inside named kernels
2. reusable engines plus orchestrate-target stitching/composition by the
   lowering pipeline

Megacu does not need a public fragment taxonomy to support this.

## Build-Graph Ownership

Dispatcher, scheduler, and kernel lowering implementations are compiled first,
then chosen and specialized during orchestrate-target lowering.

Runtime orchestration must not choose among them.

## Backend Resolution Contract

Backend adapters own the final translation from Megacu slots to native backend
objects.

For CUDA/NVSHMEM first, `src/backends/nvshmem/` must provide:

- `megacu::nvshmem::team_view`: typed runtime handle passed into the
  orchestrate program;
- event-storage layout rules for `remote_event`;
- peer resolution from dispatcher participant slots to NVSHMEM PE ids;
- signal/wait helpers callable from lowered kernels;
- host/runtime validation that the supplied team and event storage match the
  target envelope.

The CUDA platform adapter under `src/platform/cuda/` must provide
`megacu::cuda::launch_view`, stream/device validation, and launch payload
construction. The NVSHMEM backend adapter must not hide CUDA stream policy inside
the team handle.

Initial device-side API shape:

```cpp
namespace megacu::nvshmem {
struct event_endpoint;

__device__ void signal(
    megacu::cuda::kernel_context ctx,
    event_endpoint endpoint,
    std::uint64_t value);

__device__ void wait(
    megacu::cuda::kernel_context ctx,
    event_endpoint endpoint,
    std::uint64_t value);
}
```

`event_endpoint` is created by lowering and accessed through
`kernel_context::event<tag>(...)`. User kernels should not compute NVSHMEM
signal addresses directly unless they intentionally bypass Megacu for a
handwritten baseline.

First backend plan shape, planned path
`include/megacu/detail/backend_plan.h`:

```cpp
namespace megacu::detail {
struct event_layout_entry {
  std::uint16_t event_slot;
  std::uint16_t domain_rank;
  std::uint32_t byte_offset;
  megacu::memory_scope scope;
};

struct nvshmem_backend_plan {
  std::span<const event_layout_entry> events;
  bool requires_symmetric_partial_buffer;
  bool may_use_multimem_reduce;
};
}
```

Runtime validation for this plan must check:

- `team.team_n_pes` matches the rank-domain extent used by target lowering;
- `launch.device_ordinal` matches the CUDA device selected before NVSHMEM
  initialization;
- `events.buffer.bytes` is large enough for every `event_layout_entry`;
- `events.buffer.backend` and `events.buffer.session` match the launched
  `team_view`;
- `events.buffer` is symmetric when any lowered event has remote scope;
- `workspace.partial.buffer.backend` and `workspace.partial.buffer.session`
  match the launched `team_view` when the selected NVSHMEM primitive requires
  remote access;
- multimem-specific paths are only enabled when the backend reports support.
- overlap schedules only launch when the current CUDA device satisfies the
  materialized `cuda_residency_envelope`.
- linked runtime metadata passes magic, version, size, checksum, and table
  presence validation before backend plans are read.

## Kernel Context Contract

Lowered kernels receive a platform-specific context. For CUDA first:

```cpp
namespace megacu::cuda {
enum class capability : std::uint8_t {
  multimem_reduce
};

struct backend_capabilities {
  __device__ bool supports(capability capability) const;
};

struct kernel_context {
  template <class DomainTag>
  __device__ domain_point<DomainTag> domain_point() const;

  __device__ logical_rank local_rank() const;

  template <class DomainTag>
  __device__ team_view<DomainTag> team() const;

  template <class ParticipantTag, class DomainPoint>
  __device__ backend_peer peer(DomainPoint point) const;

  template <class EventTag, class DomainPoint>
  __device__ megacu::nvshmem::event_endpoint event(
      DomainPoint point,
      backend_peer peer) const;

  __device__ backend_capabilities backend() const;
};
}
```

The context is target-specific. It may be a compact pointer to materialized
metadata, inline constants, or registers populated by linked lowering
implementations. Its observable contract is typed lookup by tag, not string
lookup by name.

For the first implementation, `kernel_context` should be a trivially copyable
CUDA device struct containing only:

- pointer to target metadata in device-accessible memory;
- current dispatch entry index or packed tile coordinate;
- local rank and team size;
- backend plan pointer.

Anything larger must be justified by a concrete kernel lookup requirement.

## Internal Records

The orchestrate target lowering may create internal records, but those records
are not public authoring APIs.

Minimum internal records for the first slice:

- op table: named op symbol, implementation symbol, domain reference
- resource table: typed view slots used by lowered code
- event table: event storage, scope, release/acquire dependencies
- dispatch table: logical domain to execution placement
- participant table: virtual participant to backend peer/rank/lane mapping
- schedule payload: static persistent execution order
- kernel payload: named kernel entrypoints and stitched execution metadata
- backend payload: NVSHMEM handles and signal/wait metadata needed by kernels

Each record must have one owner:

- public program builders collect semantic facts;
- dispatcher owns placement;
- scheduler owns execution order;
- lowering owns kernel stitching and launch payloads;
- platform/backend adapters own native handles and primitive calls.

No record should duplicate a concept already owned by another layer.

## Implementation Architecture

The implementation should likely organize these as separate internal components:

- dispatcher implementation
- scheduler implementation
- kernel-lowering implementation
- target-lowering driver that applies them to one concrete orchestrate program
- CUDA platform adapter under `src/platform/cuda/`
- NVSHMEM backend adapter under `src/backends/nvshmem/`

But they are not separate public authoring layers. Their composition is part of
the CMake-managed build graph that produces one compiled orchestrate target.
