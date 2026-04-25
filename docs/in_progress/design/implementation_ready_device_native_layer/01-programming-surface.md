# Programming Surface

Megacu user code is ordinary C++/CUDA linked into a target. The user should not
author an IR. The user should not call a materializer. The user should not pass
through a generic runtime graph object.

## Direct Orchestrate ABI

The first CUDA+NVSHMEM target exposes direct functions:

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

## Runtime Views

The public reusable view layer stays small:

```cpp
namespace megacu {
struct tensor_view;
struct symmetric_buffer_view;
struct symmetric_tensor_view;
struct event_storage_view;
struct status;
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

## Orchestrator Code

An orchestrator is normal C++ code that calls runtime components:

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
      .events = events};

  auto validation = megacu::cuda_nvshmem::validate(ctx, ws, problem);
  if (validation.code != megacu::status_code::ok) {
    return validation;
  }

  auto dispatch = megacu::dispatcher::cuda_nvshmem_gemm_ar(problem, team);
  return megacu::scheduler::overlap_gemm_ar(
      ctx,
      dispatch,
      ws,
      problem,
      megacu::operators::gemm_ar_overlap{});
}
```

This sketch shows the intended ownership. Exact names can change during
implementation, but the direction cannot revert to compiler-like IR building.

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
