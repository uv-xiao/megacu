# Programming Surface

Megacu user code is ordinary C++/CUDA linked into a target. The user should not
author an IR. The user should not call a materializer. The user should not pass
through a generic runtime graph object.

The user-facing surface has two layers:

1. direct target ABI used by application/framework code;
2. virtual-participant annotations and native operator entrypoints linked into
   that target.

Component-facing runtime APIs live under `include/megacu/detail/` or
component-specific headers until they are proven reusable.

## PR #4 Proof Boundary

PR #4 only needs one tiny problem-specific proof example. The proof example may
hard-code a narrow problem shape, dtype, operator, or mapping helper when that
keeps the architecture readable. It must not present those shortcuts as the
general Megacu programming surface.

Reusable APIs in this document are architecture direction. The future general
implementation work in `docs/todo/concrete_impl/` must prove them with stronger
generality evidence before they become accepted shared implementation.

## Direct Orchestrate ABI

The full CUDA+NVSHMEM design can expose direct functions such as:

```cpp
megacu::status cuda_nvshmem_gemm_allreduce_phased_orchestrate(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem);

megacu::status cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem);
```

The ABI is explicit because framework bindings and native callers need a stable
call surface. The target name encodes the linked platform/backend/scheduler
configuration. Runtime parameters provide data and team handles; they do not
select a different backend or scheduler.

For PR #4, one direct function is enough if it proves the corrected call path.
Phased and overlap variants can remain architecture examples or future
implementation work.

The direct ABI should be callable from:

- a normal C++ executable;
- an MPI-launched C++ executable after an adapter creates CUDA/NVSHMEM views;
- a PyTorch extension after a torch-distributed adapter creates the same views.

## Runtime Views

The public reusable view layer stays small:

```cpp
namespace megacu {
enum class dtype : std::uint16_t;
enum class status_code : std::uint8_t;
struct tensor_view;
struct symmetric_buffer_view;
struct symmetric_tensor_view;
struct event_storage_view;
struct status;

struct target_capability {
  std::uint16_t max_team_size;
  dtype supported_dtype;
  bool requires_symmetric_partial;
  bool supports_single_card;
  bool supports_multi_card;
  bool requires_co_resident_progress;
};

struct runtime_context {
  cuda::launch_view launch;
  nvshmem::team_view team;
  event_storage_view events;
  target_capability capability;
};
}

namespace megacu::cuda {
struct launch_view;
}

namespace megacu::nvshmem {
struct team_view;
}
```

These are value views over storage and native handles. They are not descriptor
records. They are not serialized metadata. They are the values the target needs
to run now.

`target_capability` may be linked as a small `constexpr` object in the target
source or passed through compile definitions into a handwritten source file. It
must not be generated from a materializer and must not contain per-tile or
per-task schedules.

## Orchestrator Code

An orchestrator is normal C++ code that calls runtime components. It supplies
workload-specific virtual-participant attributes to the general dispatcher
linked by the `ConfigureTarget`:

```cpp
namespace gemm_ar_tags {
struct gemm_producer;
struct reduce_consumer;
}

constexpr auto participants = megacu::participant_list{
    megacu::participant<gemm_ar_tags::gemm_producer>(
        "gemm_producer",
        megacu::participant_attrs{
            .role = megacu::participant_role::compute,
            .placement = megacu::placement_scope::per_rank,
            .progress = megacu::progress_requirement::nonblocking,
            .peer_policy = megacu::peer_policy::local_then_remote,
            .requires = megacu::participant_requirement::none}),
    megacu::participant<gemm_ar_tags::reduce_consumer>(
        "reduce_consumer",
        megacu::participant_attrs{
            .role = megacu::participant_role::communication,
            .placement = megacu::placement_scope::per_peer,
            .progress = megacu::progress_requirement::may_block,
            .peer_policy = megacu::peer_policy::all_remote_peers,
            .requires = megacu::participant_requirement::
                co_resident_progress_if_blocking})};
```

The annotations are small typed declarations. They let the linked dispatcher
choose concrete rank, lane, peer, and work ownership at runtime from the current
team and problem. They are not a program builder, not a static schedule, and
not metadata materialized by a compiler pass.

The orchestrate call then uses the `ConfigureTarget` dispatcher through a
generic API:

```cpp
megacu::status cuda_nvshmem_gemm_allreduce_overlap_orchestrate(
    gemm_ar_workspace ws,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem) {
  auto ctx = megacu::runtime_context{
      .launch = launch,
      .team = team,
      .events = events,
      .capability = cuda_nvshmem_gemm_allreduce_overlap_capability()};

  auto validation = megacu::target::validate(ctx, ws, problem);
  if (validation.code != megacu::status_code::ok) {
    return validation;
  }

  megacu::dispatch_state dispatch;
  validation = megacu::dispatcher::map(
      dispatch,
      megacu::dispatch_request{
          .context = ctx,
          .problem = megacu::problem_view::from(problem),
          .participants = participants});
  if (validation.code != megacu::status_code::ok) {
    return validation;
  }

  return megacu::scheduler::run_overlap_gemm_ar(
      ctx,
      dispatch,
      ws,
      problem,
      megacu::operators::gemm_ar_overlap{});
}
```

This sketch shows the intended ownership. Exact names can change during
implementation, and PR #4 may replace the generic dispatcher call with an
example-local mapper. The direction cannot revert to compiler-like IR building.

The orchestrator must stay readable top-to-bottom. It should not hide runtime
state in a generic payload blob.

## Virtual Participant Annotation API

The public programming surface should expose only the minimum needed for an
orchestrate target to describe logical actors:

```cpp
namespace megacu {
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

template <class Tag>
constexpr participant_ref<Tag> participant(
    std::string_view debug_name,
    participant_attrs attrs);
}
```

`debug_name` is for diagnostics. Runtime mapping must not depend on string
lookup. Type tags and the `participant_attrs` values are the semantic input.

The dispatcher may reject annotations that are unsupported by the linked
`ConfigureTarget`, such as a blocking communication participant without a
co-resident progress-capable scheduler.

PR #4 does not need to implement this whole reusable annotation API. It only
needs to keep any narrower proof helper out of public shared API unless the
design can defend it.

## Kernel Operator Code

Kernel/operator code is native CUDA/C++:

```cpp
extern "C" megacu::status megacu_cuda_gemm_allreduce_overlap_f32(
    gemm_ar_workspace workspace,
    megacu::event_storage_view events,
    megacu::cuda::launch_view launch,
    megacu::nvshmem::team_view team,
    gemm_ar_problem problem);
```

Operators may call backend primitives such as device-side NVSHMEM waits,
signals, gets, puts, or reductions. Megacu links those primitives; it does not
rewrite the operator body.

Operator entrypoints should return `megacu::status` for host-visible failures.
Device-side failures that cannot return synchronously should be surfaced through
CUDA errors, status buffers, or explicit validation before launch.

## Task And Event Programming Model

The public programming model should not expose a task descriptor builder. For
the first slice, tasks and events are visible as runtime concepts:

```cpp
struct gemm_ar_tile {
  std::int32_t tile_m;
  std::int32_t tile_n;
  std::int32_t peer_rank;
};

struct tile_ready_events {
  megacu::event_storage_view storage;
};
```

Schedulers and operators use those runtime concepts directly:

```text
dispatcher yields tile work
  -> compute operator writes partial tile
  -> backend event primitive marks tile ready
  -> scheduler/backend waits for tile readiness
  -> communication operator reduces tile
```

There is no event name resolution at runtime. Event identity is determined by
the linked target and the tile/rank cursor.

## What Replaces Program IR

Instead of program records, the target owns:

- a direct function signature;
- a linked component configuration;
- runtime component calls;
- native operator symbols.

If a component needs a small data object, it should be a runtime view or cursor,
for example:

```cpp
struct tile_work {
  int tile_m;
  int tile_n;
  int logical_rank;
  int peer_rank;
};
```

Such objects are runtime execution state, not compiler IR.

## Public Surface Acceptance Criteria

- A user can understand the target ABI by reading one example header.
- The normal hot path does not allocate a graph, plan, or materialized table.
- The same ABI accepts single-card and two-card runtime team views.
- A framework adapter only converts framework objects into Megacu views and
  calls the direct ABI.
- Unsupported runtime inputs return `megacu::status`, not undefined behavior.
